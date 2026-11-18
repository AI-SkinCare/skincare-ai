"""추천 모델 추론. 서버와 평가 스크립트가 공통으로 사용한다.

torch/transformers는 load() 안에서만 import한다.
→ GPU 없는 환경(백엔드 팀원 PC, CI)에서도 서버·테스트가 mock 모드로 돌아간다.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from threading import RLock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .fallback import RuleBasedRecommender
from .schema import RecInput, build_messages, parse_output


@dataclass
class RecResult:
    parsed: dict[str, Any]      # {"reasoning": [...], "answer": str}
    model_version: str
    used_fallback: bool
    attempts: int
    raw_text: str
    output_tokens: int
    latency_ms: int


class Recommender(Protocol):
    version: str

    def recommend(self, x: RecInput) -> RecResult: ...


class MockRecommender:
    """GPU 없이 API 연동을 개발·테스트하기 위한 대체 모델 (규칙 기반)."""

    def __init__(self, top_path: str | Path | None = None):
        self.rule = RuleBasedRecommender(top_path)
        self.version = "mock_" + self.rule.version

    def recommend(self, x: RecInput) -> RecResult:
        t0 = time.perf_counter()
        return RecResult(self.rule.recommend(x), self.version, False, 1, "", 0,
                         int((time.perf_counter() - t0) * 1000))


class SLLMRecommender:
    """베이스 모델 + LoRA 어댑터를 4-bit로 올려 추천을 생성한다."""

    def __init__(
        self,
        base_model: str,
        adapter_dir: str | None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 2400,
        version: str | None = None,
        top_path: str | Path | None = None,
        serving_guidance: str | None = None,
    ):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.version = version or (Path(adapter_dir).parent.name if adapter_dir else "base_" + base_model.split("/")[-1])
        self.fallback = RuleBasedRecommender(top_path)
        self.serving_guidance = serving_guidance
        self.model = None
        self.tok = None
        self._generation_lock = RLock()

    def load(self) -> "SLLMRecommender":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cuda = torch.cuda.is_available()
        bf16 = cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
        kw: dict[str, Any] = {"torch_dtype": dtype}
        if self.load_in_4bit and cuda:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            )
            kw["device_map"] = {"": 0}
        tok_src = self.adapter_dir if self.adapter_dir and Path(self.adapter_dir, "tokenizer_config.json").exists() else self.base_model
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **kw)
        if self.adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_dir)
        if not cuda:
            model = model.to("cpu")
        model.eval()
        self.model = model
        return self

    def generate_raw(self, x: RecInput, sample: bool = False, trusted_context: str | None = None) -> tuple[str, int]:
        import torch

        assert self.model is not None, "load()를 먼저 호출하세요"
        # 학습(train_qlora.encode)과 같은 프롬프트여야 한다. Qwen2.5 템플릿은 enable_thinking을 무시한다.
        messages = build_messages(x)
        if self.serving_guidance:
            messages[0]["content"] += "\n\n" + self.serving_guidance
        if trusted_context:
            # Evidence mode is an extractive task, not the generic beauty-advice
            # persona. Both paired arms receive the same task instructions.
            messages[0]["content"] = (self.serving_guidance or '') + '\n\n' + trusted_context
            messages[1]['content'] += '\n\n서버 참고 자료와 규약:\n' + trusted_context
            messages[1]['content'] += ('\n\n서버 출력 조건: 첫 단계는 다음 한 문장만 쓰세요: 사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다. '
                '두 번째 단계는 성분을 최대 두 개 선택하세요. 근거 자료가 있으면 각 성분의 effect 문장을 그대로 사용하고 끝에 정확한 [E_...] ID를 붙이세요. 추가 효능 문장을 만들지 마세요. '
                '세 번째 단계에는 무향 완제품, 제품 표시대로 사용, 자극 시 중단 조건을 반드시 포함하세요. 선크림을 골랐다면 광범위 SPF30 이상 내수성 표시 조건도 쓰세요. '
                '연구 수치·고농도·임의의 횟수나 대기시간을 권장하지 마세요. 단일 성분으로 질환 치료·항염·항균 효과를 단정하지 마세요. JSON 외에는 쓰지 마세요.')
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        # 일부 토크나이저가 내보내는 token_type_ids 등은 generate()가 거부하므로 필요한 것만 전달
        enc = {k: enc[k].to(self.model.device) for k in ("input_ids", "attention_mask") if k in enc}
        gen_kw: dict[str, Any] = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                      repetition_penalty=1.05)
        if sample:
            gen_kw.update(do_sample=True, temperature=0.3, top_p=0.9)
        else:
            gen_kw.update(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), int(new.shape[0])

    def recommend(self, x: RecInput) -> RecResult:
        return self._recommend(x)

    def recommend_with_context(self, x: RecInput, trusted_context: str, *, disable_adapter: bool = False) -> RecResult:
        """Request-local server evidence; never mutate shared serving guidance."""
        result = self._recommend(x, trusted_context, disable_adapter)
        if disable_adapter and hasattr(self.model, 'disable_adapter'):
            result.model_version = 'base_' + self.base_model.split('/')[-1] + '__source_context_v1'
        return result

    def _recommend(self, x: RecInput, trusted_context: str | None = None, disable_adapter: bool = False) -> RecResult:
        # PEFT's temporary switch is model-wide. Serialize all recommendations,
        # including the legacy path, so concurrent HTTP calls cannot observe it.
        with self._generation_lock:
            ctx = self.model.disable_adapter() if disable_adapter and hasattr(self.model, 'disable_adapter') else nullcontext()
            with ctx:
                return self._run_recommend(x, trusted_context)

    def _run_recommend(self, x: RecInput, trusted_context: str | None = None) -> RecResult:
        t0 = time.perf_counter()
        raw, n_tok, attempts = "", 0, 0
        for sample in (False, True):  # 1차 greedy, 실패 시 1회 샘플링 재생성
            attempts += 1
            raw, n = (self.generate_raw(x, sample=sample, trusted_context=trusted_context)
                      if trusted_context else self.generate_raw(x, sample=sample))
            n_tok += n
            parsed = parse_output(raw)
            if parsed:
                return RecResult(parsed, self.version, False, attempts, raw, n_tok,
                                 int((time.perf_counter() - t0) * 1000))
        return RecResult(self.fallback.recommend(x), self.fallback.version, True, attempts, raw, n_tok,
                         int((time.perf_counter() - t0) * 1000))
