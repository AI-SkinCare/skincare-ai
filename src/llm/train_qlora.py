"""성분 추천 sLLM QLoRA 파인튜닝.

사용 (PowerShell, 레포 루트에서):
  python -m src.llm.train_qlora --smoke      # 0.5B 모델로 10스텝, 파이프라인 검증 (수 분)
  python -m src.llm.train_qlora              # 본 학습 (configs/sllm_qlora.yaml)

설계 포인트
  - 손실은 '답변 토큰'에만 건다. 프롬프트까지 학습하면 모델이 질문 형식을 외우는 데 용량을 쓴다.
  - max_seq_len을 넘는 샘플은 잘라 쓰지 않고 제외한다. JSON 중간에서 잘린 정답을 학습하면
    '닫는 괄호 없이 끝내는 법'을 배우기 때문이다.
  - Windows: DataLoader 워커 0 (멀티프로세싱 이슈 회피).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from .schema import RecInput, build_messages

ROOT = Path(__file__).resolve().parents[2]


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def encode(row: dict, tok, max_len: int) -> dict | None:
    x = RecInput.from_dict(row["input"])
    prompt = tok.apply_chat_template(build_messages(x), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(row["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(t_ids) > max_len:
        return None
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/sllm_qlora.yaml"))
    ap.add_argument("--smoke", action="store_true", help="작은 모델·소량 데이터로 파이프라인만 검증")
    ap.add_argument("--base", help="베이스 모델 덮어쓰기")
    ap.add_argument("--no-4bit", action="store_true", help="4-bit 양자화 끄기 (CPU 테스트용)")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 이어서 학습")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    d, m, lcfg, t = cfg["data"], cfg["model"], cfg["lora"], cfg["train"]
    base = args.base or (cfg["smoke"]["base"] if args.smoke else m["base"])
    out_dir = ROOT / (cfg["smoke"]["output_dir"] if args.smoke else t["output_dir"])
    n_train = cfg["smoke"]["max_train_samples"] if args.smoke else None
    n_eval = cfg["smoke"]["max_eval_samples"] if args.smoke else None

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments, set_seed)

    set_seed(d["seed"])
    cuda = torch.cuda.is_available()
    use_4bit = m["load_in_4bit"] and not args.no_4bit and cuda
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    print(f"[env] cuda={cuda} gpu={torch.cuda.get_device_name(0) if cuda else '-'} bf16={bf16} 4bit={use_4bit}")
    print(f"[model] {base}")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    proc = ROOT / d["processed_dir"]
    if not (proc / "train.jsonl").exists():
        sys.exit("학습 데이터가 없습니다. 먼저 python -m src.llm.prepare_data 를 실행하세요.")
    train_rows = read_jsonl(proc / "train.jsonl", n_train)
    eval_rows = read_jsonl(proc / "val_eval.jsonl", n_eval)
    max_len = m["max_seq_len"]
    train_ds = [e for e in (encode(r, tok, max_len) for r in train_rows) if e]
    eval_ds = [e for e in (encode(r, tok, max_len) for r in eval_rows) if e]
    skipped = len(train_rows) - len(train_ds)
    lens = sorted(len(e["input_ids"]) for e in train_ds)
    print(f"[data] train {len(train_ds)} (길이 초과 제외 {skipped}) / eval {len(eval_ds)} / "
          f"토큰 길이 p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")

    kw = {"torch_dtype": dtype}
    if use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    model.config.use_cache = False  # gradient checkpointing과 충돌
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cuda:
        model.enable_input_require_grads()  # 비양자화 + gradient checkpointing + LoRA 조합에 필요
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs or t["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if args.smoke else -1,
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=1 if args.smoke else t["grad_accum"],
        learning_rate=t["lr"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=1 if args.smoke else t["logging_steps"],
        eval_strategy="steps",
        eval_steps=5 if args.smoke else t["eval_steps"],
        save_strategy="steps",
        save_steps=10 if args.smoke else t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=bf16,
        fp16=cuda and not bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=d["seed"],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=Collator(tok.pad_token_id))
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    metrics = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    info = {
        "base_model": base, "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "train_samples": len(train_ds), "skipped_too_long": skipped, "max_seq_len": max_len,
        "eval_loss": metrics.get("eval_loss"), "train_minutes": round((time.time() - t0) / 60, 1),
        "lora": lcfg, "4bit": use_4bit,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] 어댑터 저장: {adapter_dir}\n{json.dumps(info, ensure_ascii=False, indent=1)}")


if __name__ == "__main__":
    main()
