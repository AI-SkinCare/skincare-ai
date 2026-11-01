"""Replay prior outputs, or exercise the actual Qwen adapter through FastAPI.

All outputs stay local. Replay is NOT fresh inference. Live cases explicitly
distinguish labelled input images from synthetic classification/survey fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.adapters import load_mapping
from server.disease_pipeline import build_disease_rec_input, load_disease_mapping
from server.recommendation_response import prepare_recommendation, QUALITY_VERSION
from server.schemas import DiseasePrediction, Survey
from src.llm.inference import RecResult
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path, data):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def summarize(records):
    bodies = [r["response"] for r in records if r["http_status"] == 200]
    recs = [b["recommendation"] for b in bodies if b.get("recommendation")]
    ingredients = [i for r in recs for i in r["ingredients"]]
    sets = Counter(tuple(sorted(i["name"] for i in r["ingredients"])) for r in recs)
    generations = [r["generation"]["result"] for r in records if r.get("generation")]
    # Rejected/format-fallback generations cost time too. Never drop them from latency.
    latencies = [r["latency_ms"] / 1000 for r in generations]
    return {
        "requests": len(records), "http_errors": sum(r["http_status"] != 200 for r in records),
        "statuses": dict(Counter(b["recommendation_status"] for b in bodies)),
        "quality_statuses": dict(Counter(r["quality_status"] for r in recs)),
        "quality_rejected": sum(b.get("recommendation_status") == "quality_rejected" for b in bodies),
        "rule_fallback": sum(r["used_fallback"] for r in recs),
        "ingredient_items": len(ingredients), "missing_purpose": sum(not i["purpose"] for i in ingredients),
        "purpose_sources": dict(Counter(i["purpose_source"] for i in ingredients)),
        "delivered_empty_ingredients": sum(not r["ingredients"] for r in recs),
        "delivered_empty_routine": sum(not r["routine"] for r in recs),
        "rejected_no_ingredients": sum("no_supported_ingredients" in b.get("quality_issues", []) for b in bodies),
        "rejected_no_routine": sum("no_supported_routine" in b.get("quality_issues", []) for b in bodies),
        "issue_counts": dict(Counter(k for b in bodies for k in b.get("quality_issues", []))),
        "unique_ingredient_sets": len(sets), "largest_repeated_set_count": max(sets.values(), default=0),
        "generation_p50_s": statistics.median(latencies) if latencies else None,
        "generation_max_s": max(latencies) if latencies else None,
        "scope": "Output structure and rule checks only; not clinical efficacy accuracy.",
        "actual_generation_calls": sum(r.get("generation") is not None for r in records),
        "raw_format_fallbacks": sum(bool(r["generation"]["result"]["used_fallback"]) for r in records if r.get("generation")),
    }


def replay(args):
    ext = IngredientExtractor.load(args.processed_dir / "ingredient_dict.json")
    records = []
    for row in read_rows(args.baseline / "recommendations.jsonl"):
        before = row["response"]["recommendation"]
        body = row["response"].copy()
        findings = []
        if before:
            req = row["request"]
            disease = DiseasePrediction.model_validate(req["disease"])
            x, notes, _ = build_disease_rec_input(Survey(**req["survey"]), req.get("metrics"), disease,
                                                 load_mapping(), load_disease_mapping(), .6)
            titles = ["문제 분석", "성분 선택", "사용법"]
            parsed = {"reasoning": [{"step": i+1, "title": titles[min(i, 2)], "content": s}
                                    for i, s in enumerate(before["reasoning"])], "answer": before["answer"]}
            result = RecResult(parsed, before["model_version"], before["used_fallback"], 1, "", 0, before["inference_ms"])
            rec, findings = prepare_recommendation(result, x, notes, ext, OutputFilter.from_yaml(), disease)
            body["quality_issues"] = rec.quality_issues
            body["recommendation"] = None if rec.quality_status == "withheld" else rec.model_dump()
            if rec.quality_status == "withheld":
                body["recommendation_status"] = "quality_rejected"
        record = {"case_id": row["rel_path"], "mode": "previously_filtered_output_replay",
                  "http_status": 200, "response": body, "findings": findings,
                  "before_ingredients": before["ingredients"] if before else []}
        append(args.output_dir / "responses.jsonl", record)
        records.append(record)
    summary = summarize(records)
    summary["mode"] = "offline_replay_not_new_inference"
    summary["baseline_sha256"] = digest(args.baseline / "recommendations.jsonl")
    summary["generation_p50_s"] = summary["generation_max_s"] = None
    write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def live(args):
    from fastapi.testclient import TestClient
    from server import main
    from server.config import Settings
    from src.llm.inference import SLLMRecommender
    from evaluate_disease_pipeline import classification, recommendation_cases

    captured = []

    class RecordingRecommender(SLLMRecommender):
        def recommend(self, x):
            result = super().recommend(x)
            captured.append({"input": x.to_dict(), "result": asdict(result)})
            return result

    main.SLLMRecommender = RecordingRecommender
    token = "local-quality-test"
    cfg = Settings(internal_token=token, rec_backend="sllm", rec_adapter_dir=str(args.adapter.resolve()),
                   processed_dir=args.processed_dir.resolve(), disease_backend="densenet" if args.image_root else "disabled")
    records = []
    source_files = list((ROOT / "server").glob("*.py")) + [ROOT / "src/llm/inference.py", ROOT / "src/llm/ingredients.py"]
    write(args.output_dir / "provenance.json", {
        "adapter": str(args.adapter), "adapter_sha256": digest(args.adapter / "adapter_model.safetensors"),
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        "processed_dir": str(args.processed_dir), "quality_version": QUALITY_VERSION,
        "test_transport": "FastAPI TestClient; real GPU Qwen, with real CPU DenseNet only for actual_image cases",
        "survey": "Synthetic explicit test profiles, not inferred demographics.",
        "model_input": "12 synthetic classification fixtures, plus 4 real image predictions if --image-root is supplied. Image ground truth never used as recommender input.",
    })
    started = time.perf_counter()
    with TestClient(main.create_app(cfg)) as client:
        print("[ready] " + args.adapter.parent.name, flush=True)
        write(args.output_dir / "model_info.json", client.get("/model-info", headers={"X-Internal-Token": token}).json())

        def request(case_id, kind, path, **kw):
            if args.cases and case_id not in args.cases:
                return
            before = len(captured)
            t0 = time.perf_counter()
            response = client.post(path, headers={"X-Internal-Token": token}, **kw)
            row = {"case_id": case_id, "kind": kind, "http_status": response.status_code,
                   "elapsed_s": time.perf_counter()-t0, "response": response.json(),
                   "generation": captured[-1] if len(captured) > before else None}
            if "json" in kw:
                row["request"] = kw["json"]
            append(args.output_dir / "responses.jsonl", row)
            records.append(row)
            summary = summarize(records)
            summary.update(phase="running", elapsed_s=time.perf_counter()-started)
            write(args.output_dir / "summary.json", summary)
            print(f"[{len(records)}] {case_id}: {response.status_code} {row['response'].get('recommendation_status')} {row['elapsed_s']:.1f}s", flush=True)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP failure: {case_id}, {response.status_code}")
            if case_id in {"uncertain_gate", "normal_gate"}:
                expected = "uncertain" if case_id == "uncertain_gate" else "no_concern"
                if row["response"].get("recommendation_status") != expected or row["generation"] is not None:
                    raise RuntimeError(f"Gate invoked generation or returned wrong status: {case_id}")

        for case_id, label, survey in recommendation_cases():
            request(case_id, "synthetic_profile_and_classification", "/infer/disease-recommend",
                    json={"disease": classification(label), "survey": survey})
        for label, conf, case in [("atopic", .4, "uncertain_gate"), ("normal", .9, "normal_gate")]:
            request(case, "synthetic_gate", "/infer/disease-recommend",
                    json={"disease": classification(label, conf), "survey": {"gender": "female", "age": 30, "skinType": "normal"}})
        if args.image_root:
            old = read_rows(args.baseline / "recommendations.jsonl")
            selected = [next(r for r in old if r["request"]["disease"]["label_en"] == "acne" and r["classification_correct"]),
                        next(r for r in old if r["rel_path"] == "아토피/정면/H5_1010_P1_L0.png"),
                        next(r for r in old if r["response"]["recommendation_status"] == "uncertain"),
                        next(r for r in old if r["response"]["recommendation_status"] == "no_concern")]
            for row in selected:
                path = (args.image_root / row["rel_path"]).resolve()
                if not path.is_relative_to(args.image_root.resolve()):
                    raise ValueError("Unsafe image path")
                with path.open("rb") as f:
                    request(row["rel_path"], "actual_image_and_hypothetical_survey", "/infer/disease-analyze",
                            files={"image": (path.name, f, "image/png")},
                            data={"survey": json.dumps(row["request"]["survey"])})
    if args.cases and set(args.cases) != {r["case_id"] for r in records}:
        raise RuntimeError("Requested cases were not all executed; check names and --image-root")
    summary = summarize(records)
    summary.update(phase="complete", elapsed_s=time.perf_counter()-started)
    write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replay", "live"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/disease_images_test_sample_96_20261016")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/sllm")
    parser.add_argument("--adapter", type=Path, default=ROOT / "outputs/rec_sllm_v2/adapter")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--cases", nargs="+", help="Run only named live cases for a focused regression")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "responses.jsonl").exists():
        raise SystemExit("Use a new output directory; existing results are preserved.")
    try:
        (replay if args.mode == "replay" else live)(args)
    except Exception as exc:
        write(args.output_dir / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
