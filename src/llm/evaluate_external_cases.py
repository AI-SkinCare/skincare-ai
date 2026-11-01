"""수집·전문가 검토·모델 선택 후 봉인한 신규 외부 사례 평가. 기존 시험으로 대체하지 않는다."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path

from .ingredients import IngredientExtractor
from .inference import SLLMRecommender
from .prepare_followup_review import validate_external_cases
from .prepare_quality_data import ROOT, digest, read_rows, serving_key
from .review_evidence import DEFAULT_OUT
from .schema import RecInput


def external_metrics(pred: list[str], gold_by_concern: dict, fallback: bool) -> dict:
    per = {c: len(set(pred) & set(g["acceptable_ingredients"])) / len(set(g["acceptable_ingredients"]))
           for c, g in gold_by_concern.items()}
    acceptable = {i for g in gold_by_concern.values() for i in g["acceptable_ingredients"]}
    avoid = {i for g in gold_by_concern.values() for i in g.get("avoid_ingredients", [])}
    return {"recognized_name_precision_at_3": len(set(pred[:3]) & acceptable) / 3 if not fallback else None,
            "per_concern_ingredient_recall": per if not fallback else None,
            "all_concerns_have_ingredient_match": all(v > 0 for v in per.values()) if not fallback else False,
            "avoid_ingredient_matches": sorted(set(pred) & avoid),
            "semantic_concern_coverage": "requires_human_review", "factual_correctness": "requires_human_review"}


def seal_manifest(path: Path, value: dict):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("sealed dataset/model changed; do not retune against this test")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True, help="개발 검증에서 선택을 끝낸 어댑터")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "independent_test")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)
    rows = read_rows(args.cases)
    blocked = {serving_key(r) for s in ("train", "val") for r in read_rows(ROOT / f"data/processed/sllm/{s}.jsonl")}
    gate = validate_external_cases(rows, blocked)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["ready"]:
        print("외부 시험 보류: 신규 수집·전문가 검토·분포 조건을 만족해야 합니다.")
        return 2
    if args.validate_only:
        return 0
    if (args.out / "items.jsonl").exists():
        raise ValueError("existing test outputs preserved; use a new output directory for an explicitly recorded rerun")
    adapter_config = args.adapter / "adapter_config.json"
    model_files = sorted(args.adapter.glob("adapter_model.*"))
    if not adapter_config.exists() or not model_files:
        raise ValueError("selected trained adapter missing")
    files = [adapter_config, *model_files]
    manifest = {"cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "selected_adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "protocol": "external_multi_concern_v1", "collector_declared_provenance": gate["provenance_status"]}
    seal_manifest(args.out / "sealed_manifest.json", manifest)
    # 외부 정답의 명시적 허용 성분만 별도 사전으로 사용. 시험 사전이 학습/개발에 유입되지 않는다.
    lex = {i: i for r in rows for g in r["gold_by_concern"].values()
           for i in g["acceptable_ingredients"] + g.get("avoid_ingredients", [])}
    ext = IngredientExtractor(lex)
    for r in rows:
        for g in r["gold_by_concern"].values():
            for name in g["acceptable_ingredients"] + g.get("avoid_ingredients", []):
                if ext.extract(name) != [name]:
                    raise ValueError(f"ambiguous or unsupported canonical ingredient name: {name}")
    base = json.loads(adapter_config.read_text(encoding="utf-8"))["base_model_name_or_path"]
    model = SLLMRecommender(base, str(args.adapter)).load()
    items = []
    for index, row in enumerate(rows, 1):
        result = model.recommend(RecInput.from_dict(row["input"]))
        pred = ext.extract("\n".join(s["content"] for s in result.parsed["reasoning"]) + "\n" + result.parsed["answer"])
        item = {"id": row["id"], "input": row["input"], "gold_by_concern": row["gold_by_concern"],
                "raw_text": result.raw_text, "parsed": result.parsed, "pred": pred,
                "fallback": result.used_fallback, "attempts": result.attempts,
                "metrics": external_metrics(pred, row["gold_by_concern"], result.used_fallback),
                "human_review": {k: None for k in ("all_concerns_addressed", "claims_supported", "personalization", "usage_appropriate")}}
        items.append(item)
        with (args.out / "items.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{index}/{len(rows)}", flush=True)
    precisions = [r["metrics"]["recognized_name_precision_at_3"] for r in items if r["metrics"]["recognized_name_precision_at_3"] is not None]
    summary = {"n": len(items), "model_generated_n": len(precisions),
               "recognized_name_precision_at_3": statistics.mean(precisions) if precisions else None,
               "all_concerns_have_ingredient_match_rate": sum(r["metrics"]["all_concerns_have_ingredient_match"] for r in items) / len(items),
               "fallback_rate": sum(r["fallback"] for r in items) / len(items),
               "semantic_review_status": "pending", "manifest_hash": digest(manifest),
               "metric_limitation": "only reviewed reference vocabulary is recognized; unregistered names require human review; not overall recommendation precision",
               "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
