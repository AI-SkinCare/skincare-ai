"""Replay saved ingredient extraction without models, filters, or production edits.

Run from the AI repository:
    .venv/Scripts/python.exe -B scripts/audit_ingredient_extraction.py
    .venv/Scripts/python.exe -B scripts/audit_ingredient_extraction.py --check

The committed old parser and current parser share the CURRENT extractor/dictionary.
Stored API results are a separate historical baseline. The input text has already
passed the historical serving filters; this cannot recover pre-filter generations.
Counts describe extraction/completeness, never clinical efficacy or safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

# Even a run without -B must not create caches in frozen production directories.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import postprocess as current_parser
from src.llm.ingredients import IngredientExtractor

BASELINE = ROOT / "outputs/disease_images_test_sample_96_20261016"
DEFAULT_OUTPUT = ROOT / "outputs/recommendation_quality_20261021/extraction_audit.json"
TITLES = ("문제 분석 및 원인 파악", "성분 선택 및 근거 제시", "사용법 및 관리방안")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def old_parser(ref: str):
    revision = git("rev-parse", "--verify", ref + "^{commit}").decode("ascii").strip()
    source = git("show", revision + ":server/postprocess.py")
    module = types.ModuleType("_extraction_audit_old_parser")
    sys.modules[module.__name__] = module
    exec(compile(source.decode("utf-8"), revision + ":server/postprocess.py", "exec"),
         module.__dict__)
    return module, revision, source


def to_parsed(rec: dict) -> dict:
    if len(rec["reasoning"]) != 3:
        raise ValueError("Baseline must have three reasoning steps for faithful reconstruction")
    steps = []
    for index, step in enumerate(rec["reasoning"]):
        steps.append(copy.deepcopy(step) if isinstance(step, dict) else {
            "step": index + 1, "title": TITLES[index], "content": step,
        })
    return {"reasoning": steps, "answer": rec["answer"]}


def metrics(cases: list[dict], stage: str) -> dict:
    counts, sources = Counter(), Counter()
    for case in cases:
        result = case[stage]
        ingredients = result["ingredients"]
        missing = sum(item["purpose"] is None for item in ingredients)
        counts.update({
            "responses": 1, "ingredient_items": len(ingredients),
            "missing_purpose": missing, "cases_with_missing_purpose": int(missing > 0),
            "empty_ingredients": int(not ingredients),
            "empty_routine": int(not result["routine_count"]),
            "empty_answer": int(not result["answer_nonempty"]),
            "empty_field_hold_candidates": int(
                not ingredients or not result["routine_count"] or not result["answer_nonempty"]),
        })
        sources.update(item.get("purpose_source", "legacy_unspecified") for item in ingredients)
    return {**dict(counts),
            "missing_purpose_rate": round(counts["missing_purpose"] / counts["ingredient_items"], 6)
            if counts["ingredient_items"] else None,
            "purpose_sources": dict(sources)}


def name_key(name: str, extractor: IngredientExtractor) -> str:
    found = extractor.extract(name)
    return found[0] if len(found) == 1 else "".join(name.split()).upper()


def indexed(items: list[dict], extractor: IngredientExtractor) -> dict:
    result = {}
    for item in items:
        key = name_key(item["name"], extractor)
        if key in result:
            raise ValueError("Ambiguous canonical ingredient names in one response: " + key)
        result[key] = item
    return result


def transitions(cases: list[dict], stage: str, extractor: IngredientExtractor) -> dict:
    counts = Counter()
    for case in cases:
        before = indexed(case[stage]["ingredients"], extractor)
        after = indexed(case["new_current"]["ingredients"], extractor)
        for name, item in before.items():
            old_status = "missing" if item["purpose"] is None else "provided"
            new_status = ("removed" if name not in after else
                          "missing" if after[name]["purpose"] is None else "provided")
            counts[old_status + "->" + new_status] += 1
        for name in after.keys() - before.keys():
            counts["added->" + ("missing" if after[name]["purpose"] is None else "provided")] += 1
    return dict(counts)


def build_report(baseline: Path, dictionary: Path, baseline_ref: str | None) -> dict:
    recommendations = baseline / "recommendations.jsonl"
    annotations_path = baseline / "manual_review_findings.json"
    provenance_path = baseline / "provenance.json"
    provenance = read_json(provenance_path)
    ref = baseline_ref or provenance["code_revision"]
    old, revision, old_source = old_parser(ref)
    paths = [Path(__file__).resolve(), ROOT / "server/postprocess.py",
             ROOT / "src/llm/ingredients.py", dictionary,
             recommendations, annotations_path, provenance_path]
    before_hashes = {relative(path): sha256(path.read_bytes()) for path in paths}
    extractor = IngredientExtractor.load(dictionary)
    # The old parser imports the same current extractor class; assert the control.
    if old.IngredientExtractor is not IngredientExtractor:
        raise ValueError("Old/current parser comparison does not share the current extractor")
    rows = [json.loads(line) for line in recommendations.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cases, seen, statuses = [], set(), Counter()
    for row in rows:
        case_id = row["rel_path"]
        if case_id in seen:
            raise ValueError("Duplicate baseline case: " + case_id)
        seen.add(case_id)
        response = row["response"]
        statuses[response["recommendation_status"]] += 1
        rec = response.get("recommendation")
        if rec is None:
            continue
        if response["recommendation_status"] != "generated":
            raise ValueError("Unexpected recommendation on a non-generated baseline case")
        parsed = to_parsed(rec)
        old_items = old.structure_ingredients(copy.deepcopy(parsed), extractor, limit=5)
        new_items = current_parser.structure_ingredients(
            copy.deepcopy(parsed), extractor, limit=5, include_routine=False)
        old_routine = old.structure_routine(copy.deepcopy(parsed))
        new_routine = current_parser.structure_routine(copy.deepcopy(parsed))
        cases.append({
            "case_id": case_id,
            "stored_baseline": {"ingredients": rec["ingredients"],
                                "routine_count": len(rec["routine"]),
                                "answer_nonempty": bool(rec["answer"].strip())},
            "old_current": {"ingredients": old_items, "routine_count": len(old_routine),
                            "answer_nonempty": bool(parsed["answer"].strip())},
            "new_current": {"ingredients": new_items, "routine_count": len(new_routine),
                            "answer_nonempty": bool(parsed["answer"].strip())},
        })

    by_id = {case["case_id"]: case for case in cases}
    annotated, outcomes = [], Counter()
    annotations = read_json(annotations_path)["manual_confirmed_malformed_purpose_items"]
    for item in annotations:
        case = by_id[item["file"]]
        key = name_key(item["name"], extractor)
        stored = indexed(case["stored_baseline"]["ingredients"], extractor).get(key)
        if stored is None or stored["purpose"] != item["purpose"]:
            raise ValueError("Annotation does not match stored baseline: " + item["file"])
        previous = indexed(case["old_current"]["ingredients"], extractor).get(key)
        new_item = indexed(case["new_current"]["ingredients"], extractor).get(key)
        outcome = ("candidate_removed" if new_item is None else
                   "unavailable" if new_item["purpose"] is None else
                   "known_malformed_retained" if new_item["purpose"] == item["purpose"] else
                   "replaced_model_explanation")
        outcomes[outcome] += 1
        annotated.append({"case_id": item["file"], "name": item["name"],
                          "annotated_baseline_purpose": item["purpose"],
                          "old_current": previous, "new_current": new_item, "outcome": outcome})

    after_hashes = {relative(path): sha256(path.read_bytes()) for path in paths}
    if before_hashes != after_hashes:
        raise RuntimeError("Audit inputs or production sources changed during replay; rerun")
    return {
        "audit_version": "ingredient_extraction_audit_v1",
        "scope": {
            "mode": "extraction_only_from_previously_filtered_saved_api_text",
            "new_generation_calls": 0, "gpu_jobs": 0,
            "additional_output_filters_applied": False, "include_routine": False,
            "ingredient_limit": 5, "response_assembler_applied": False,
            "clinical_efficacy_or_safety_assessed": False,
            "interpretation": (
                "Missing purposes measure completeness. Changed annotated purposes measure "
                "known-defect recurrence, not overall attribution or clinical accuracy. "
                "Full serving replay with filters and include_routine=True is a separate result."),
        },
        "provenance": {
            "baseline_directory": relative(baseline), "dictionary": relative(dictionary),
            "baseline_parser_requested_ref": ref, "baseline_parser_commit": revision,
            "baseline_parser_sha256": sha256(old_source),
            "current_repository_head": git("rev-parse", "HEAD").decode("ascii").strip(),
            "old_and_new_share_current_extractor_and_dictionary": True,
            "source_sha256": before_hashes,
            "reproduce": (
                ".\\.venv\\Scripts\\python.exe -B scripts/audit_ingredient_extraction.py "
                "--baseline-ref " + revision),
        },
        "requests": len(rows), "baseline_statuses": dict(statuses),
        "metrics": {stage: metrics(cases, stage)
                    for stage in ("stored_baseline", "old_current", "new_current")},
        "transitions": {
            "stored_baseline_to_new_current": transitions(cases, "stored_baseline", extractor),
            "old_current_to_new_current": transitions(cases, "old_current", extractor),
        },
        "known_malformed": {
            "annotated_items": len(annotations),
            "annotated_cases": len({item["file"] for item in annotations}),
            "outcomes": dict(outcomes), "items": annotated,
        },
        "cases": cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE)
    ap.add_argument("--dictionary", type=Path, default=ROOT / "data/processed/sllm/ingredient_dict.json")
    ap.add_argument("--baseline-ref", help="Defaults to the commit recorded in baseline provenance.json")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--check", action="store_true", help="Verify byte-identical existing output; do not write")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Restrict output to local result storage, not production or baseline inputs.
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.is_relative_to(args.baseline_dir.resolve()):
        ap.error("--output must be under outputs and outside the baseline directory")
    report = build_report(args.baseline_dir, args.dictionary, args.baseline_ref)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
               + "\n").encode("utf-8")
    if args.check:
        if not output.exists() or output.read_bytes() != encoded:
            raise SystemExit("CHECK FAILED: saved report differs from current inputs/code")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    print(json.dumps({"output": relative(output), "verified_only": args.check,
                      "metrics": report["metrics"],
                      "known_malformed_outcomes": report["known_malformed"]["outcomes"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
