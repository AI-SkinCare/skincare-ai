"""개발 문항의 모델별 생성문을 나란히 모아 전문가 내용 검토에 제공한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_quality_data import ROOT, read_rows
from .review_evidence import DEFAULT_OUT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "model_content_review.jsonl")
    args = ap.parse_args(argv)
    cases = {r["id"]: r for r in read_rows(ROOT / "data/processed/sllm_v3/val_eval.jsonl")}
    evidence = {r["id"]: r for r in read_rows(DEFAULT_OUT / "claim_review_queue.jsonl")}
    by_id, comparison_keys = {}, set()
    for path in sorted(args.eval_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in result or "items" not in result:
            continue
        summary = result["summary"]
        comparison_keys.add(summary["comparison_key"])
        for item in result["items"]:
            identifier = item["id"]
            case = cases[identifier]
            row = by_id.setdefault(identifier, {"id": identifier, "input": case["input"],
                "single_concern_reference_only": json.loads(case["target"]),
                "bibliography_review": evidence[identifier]["bibliography_review"], "models": {}})
            row["models"][summary["model"]] = {"raw_text": item["raw_text"], "cleaned": item["cleaned"],
                "fallback": item["fallback"], "recognized_ingredients": item["pred"],
                "human_review": {key: None for key in ("all_concerns_addressed", "claims_supported", "personalization", "usage_appropriate")},
                "reviewer_id": None, "review_notes": None}
    if len(comparison_keys) != 1:
        raise ValueError("compatible evaluation artifacts required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(by_id.values(), key=lambda r: (-len(r["input"]["concerns"]), r["id"]))), encoding="utf-8")
    print(f"{len(by_id)} cases exported to {args.out}; human review pending")


if __name__ == "__main__":
    main()
