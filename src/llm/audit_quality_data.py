"""생성한 v3 데이터 무결성·성분 보존·분할 겹침을 GPU 없이 검사한다."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from .ingredients import IngredientExtractor
from .prepare_quality_data import CLINICAL, ROOT, TITLES, digest, normalize_steps, read_rows, target_text, validate_splits
from .schema import RecInput, build_messages


def audit(proc: Path, src: Path, tokenizer=None, max_len: int = 2048) -> dict:
    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    splits = {s: read_rows(proc / f"{s}.jsonl") for s in ("train", "val_eval", "test")}
    ext = IngredientExtractor.load(proc / "ingredient_dict.json")
    assert digest(ext.lexicon) == manifest["lexicon_hash"], "lexicon changed"
    assert hashlib.sha256((ROOT / "configs/filter_rules.yaml").read_bytes()).hexdigest() == manifest["filter_hash"], "filter rules changed"
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == expected, f"source changed: {name}"
    for name, expected in manifest["builder_source_sha256"].items():
        assert hashlib.sha256((ROOT / "src/llm" / name).read_bytes()).hexdigest() == expected, f"builder changed: {name}"
    originals = {r["id"]: r for name in ("train", "val") for r in read_rows(src / f"{name}.jsonl") if "#dup" not in r["id"]}
    validate_splits(splits)
    checks = collections.Counter()
    lengths, too_long = {}, {}
    for split, rows in splits.items():
        assert digest(rows) == manifest["split_hashes"][split], f"split changed: {split}"
        assert len({r["id"] for r in rows}) == len(rows), f"duplicate IDs: {split}"
        sizes, skipped = [], collections.Counter()
        for r in rows:
            x = RecInput.from_dict(r["input"])
            x.validate()
            assert x.user_note is None and x.condition_codes is None
            assert r["target_concern"] in x.concerns
            t = json.loads(r["target"])
            assert tuple(s["title"] for s in t["reasoning"]) == TITLES
            assert len(t["reasoning"]) == 3 and all(s["content"].strip() for s in t["reasoning"]) and t["answer"].strip()
            gold = ext.extract(target_text(t))
            assert gold and gold == r["gold_ingredients"], f"gold mismatch: {r['id']}"
            original = json.loads(originals[r["id"].split("#dup")[0]]["target"])
            before = set(ext.extract(target_text(normalize_steps(original))))
            assert before <= set(gold), f"ingredient deleted: {r['id']}"
            assert not CLINICAL.search(" ".join(s["content"] for s in t["reasoning"]) + " " + t["answer"])
            checks["checked_rows"] += 1
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(build_messages(x), tokenize=False, add_generation_prompt=True)
                n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + len(tokenizer(r["target"], add_special_tokens=False)["input_ids"]) + 1
                sizes.append(n)
                if n > max_len:
                    skipped[r["target_concern"]] += 1
        if sizes:
            sizes.sort()
            lengths[split] = {"n": len(sizes), "p50": sizes[len(sizes) // 2],
                              "p95": sizes[math.ceil(len(sizes) * .95) - 1], "max": sizes[-1]}
            too_long[split] = {"total": sum(skipped.values()), "per_concern": dict(skipped)}
    return {"dataset_id": manifest["dataset_id"], "checks": dict(checks), "split_input_overlap": 0,
            "split_target_overlap": 0, "ingredient_loss_rows": 0, "invalid_gold_rows": 0,
            "tokenizer": getattr(tokenizer, "name_or_path", None), "max_seq_len": max_len,
            "token_lengths": lengths, "too_long": too_long}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/sllm_v3")
    ap.add_argument("--src", type=Path, default=ROOT / "data/processed/sllm")
    ap.add_argument("--tokenizer", help="캐시된 토크나이저로 길이 검사. 네트워크/GPU는 사용하지 않음")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args(argv)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = audit(args.data, args.src, tok, args.max_seq_len)
    (args.data / "audit_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
