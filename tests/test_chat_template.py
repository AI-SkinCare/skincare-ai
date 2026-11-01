"""Qwen3 생각 모드가 학습·추론 프롬프트에 새지 않는지, Qwen2.5 프롬프트는 그대로인지 확인한다.

토크나이저가 로컬 캐시에 없으면 건너뛴다(네트워크 없이도 테스트가 돌도록).
"""
import pytest

from src.llm.schema import RecInput, build_messages
from src.llm.train_qlora import encode

transformers = pytest.importorskip("transformers")


def _tok(name):
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:  # noqa: BLE001 — 캐시 없음
        pytest.skip(f"{name} 토크나이저가 캐시에 없음")


def _row():
    x = RecInput(gender="여성", age=30, skin_type="복합성", concerns=["모공"])
    return {"input": x.to_dict(), "target": '{"reasoning": [], "answer": "a"}'}


def test_qwen3_prompt_has_empty_think_block():
    tok = _tok("Qwen/Qwen3-8B")
    e = encode(_row(), tok, 4096)
    n_prompt = sum(1 for lab in e["labels"] if lab == -100)
    prompt = tok.decode(e["input_ids"][:n_prompt])
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")   # 모델은 JSON부터 쓴다
    assert e["input_ids"][-1] == tok.eos_token_id and e["labels"][-1] == tok.eos_token_id


def test_qwen25_prompt_unchanged_by_flag():
    tok = _tok("Qwen/Qwen2.5-0.5B-Instruct")
    x = RecInput.from_dict(_row()["input"])
    msgs = build_messages(x)
    a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert a == b                                   # 기존 v1/v2 어댑터의 추론 프롬프트가 그대로
    assert a.endswith("<|im_start|>assistant\n")


def test_auxiliary_uses_a_distinct_prompt_and_only_answer_tokens_have_loss():
    from src.llm.prepare_auxiliary_data import dictionary_tasks
    tok = _tok("Qwen/Qwen3-8B")
    row = dictionary_tasks([{'inci_name':'glycerin','function':'Humectant'}],
                          {'license':'CC-BY-4.0'})[0]
    e = encode(row,tok,2048)
    n_prompt = sum(v==-100 for v in e['labels'])
    prompt = tok.decode(e['input_ids'][:n_prompt])
    assert '제공된 성분 사전' in prompt and '주요 고민' not in prompt
    assert e['labels'][n_prompt:]==e['input_ids'][n_prompt:]
    assert 'not_established' in tok.decode(e['input_ids'][n_prompt:])
