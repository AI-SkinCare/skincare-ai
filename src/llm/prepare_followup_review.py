"""공개 성분 사전 대조와 신규 사례 수집 준비. 자동 생성 사례에 정답을 붙이지 않는다."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import csv
import itertools
import json
import re
import urllib.request
from pathlib import Path

from .prepare_quality_data import ROOT, read_rows, serving_key
from .schema import CONCERNS, GENDERS, SKIN_TYPES, RecInput
from .review_evidence import DEFAULT_OUT

HF_REPO = "cureskin/indian-skincare-ingredient-flags"
HF_REVISION = "e34203b35153fd1cd9de6de7c938165c722c003f"


def name_key(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def acquire_dictionary(out: Path) -> tuple[list[dict], dict]:
    dest = out / "external/cureskin" / HF_REVISION
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename in ("ingredient-flags.csv", "README.md", "LICENSE"):
        path = dest / filename
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{filename}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read()
            path.write_bytes(raw)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    # CC BY로 제공된 사전. 재사용 시 이름·버전·변경 여부를 함께 보존한다.
    card = (dest / "README.md").read_text(encoding="utf-8")
    if "cc-by-4.0" not in card.lower():
        raise ValueError("expected license declaration missing")
    rows = list(csv.DictReader(io.StringIO((dest / "ingredient-flags.csv").read_text(encoding="utf-8-sig"))))
    if not rows or "inci_name" not in rows[0]:
        raise ValueError("unexpected dictionary schema")
    metadata = {"repository": HF_REPO, "revision": HF_REVISION, "license": "CC-BY-4.0", "rows": len(rows),
                "attribution": "CureSkin, Indian Skincare Ingredient Flags v2026.07", "files_sha256": hashes,
                "source_url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REVISION}",
                "usage": "exact_identity_lookup_only; no personalized labels; blank flags mean unknown"}
    (dest / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, metadata


def dictionary_matches(lexicon: dict, external: list[dict]) -> list[dict]:
    by_name = collections.defaultdict(list)
    for row in external:
        by_name[name_key(row["inci_name"])].append(row)
    grouped = collections.defaultdict(set)
    for alias, canonical in lexicon.items():
        if re.search(r"[A-Za-z]", alias):
            grouped[canonical].add(name_key(alias))
        else:
            grouped[canonical]  # 영어 별칭 없는 대표명도 누락하지 않는다.
    return [{"ingredient": name, "english_aliases": sorted(aliases),
             "exact_matches": [r for a in sorted(aliases) for r in by_name.get(a, [])],
             "claim_status": "not_expert_verified"} for name, aliases in sorted(grouped.items())]


def diagnostic_inputs(blocked: set[str]) -> list[dict]:
    """8종×피부타입×성별 + 모든 고민 쌍. 합성·입력 전용이며 독립 시험으로 세지 않는다."""
    rows = []
    combinations = [(c,) for c in CONCERNS] + list(itertools.combinations(CONCERNS, 2))
    for concerns in combinations:
        profiles = itertools.product(SKIN_TYPES, GENDERS) if len(concerns) == 1 else [("복합성", "여성"), ("건성", "남성")]
        for skin, gender in profiles:
            for age in range(25, 66):
                x = RecInput(gender=gender, age=age, skin_type=skin, concerns=list(concerns))
                key = serving_key({"input": x.to_dict()})
                if key not in blocked:
                    break
            else:
                raise ValueError("no unused diagnostic profile")
            blocked.add(key)
            rows.append({"id": "SYN_DIAG_" + key[:16], "input": x.to_dict(), "source_origin": "synthetic_design",
                         "usage": "diagnostic_only", "expert_review_status": "pending", "gold_by_concern": None,
                         "is_independent_human_test": False})
    return rows


def validate_external_cases(rows: list[dict], blocked: set[str], min_per_concern=20, min_multi=40) -> dict:
    """신규 시험 선언·검토·겹침 게이트. 실제 수집/검토 여부는 팀의 책임 있는 확인이 필요하다."""
    errors, seen_ids, seen_inputs = [], set(), set()
    counts = collections.Counter()
    multi = 0
    for r in rows:
        identifier = r.get("id")
        try:
            x = RecInput.from_dict(r["input"])
            x.validate()
            if len(set(x.concerns)) != len(x.concerns):
                raise ValueError("duplicate concerns")
            key = serving_key(r)
            if not identifier or identifier in seen_ids or key in seen_inputs or key in blocked:
                raise ValueError("missing/duplicate ID or historical/duplicate input")
            seen_ids.add(identifier)
            seen_inputs.add(key)
            if r.get("source_origin") != "new_human_collection" or not r.get("collection_batch_id"):
                raise ValueError("new human provenance required")
            review = r.get("review", {})
            if review.get("status") != "approved" or not all(review.get(k) for k in ("reviewer_id", "qualification", "reviewed_at")):
                raise ValueError("review approval and attribution required")
            if review.get("claim_evidence_verified") is not True:
                raise ValueError("claim-to-source review required")
            gold = r.get("gold_by_concern", {})
            if set(gold) != set(x.concerns):
                raise ValueError("every input concern needs a reviewed gold set")
            for concern in x.concerns:
                entry = gold[concern]
                acceptable = entry.get("acceptable_ingredients", [])
                citations = entry.get("evidence_urls", [])
                if not acceptable or not all(isinstance(s, str) and s.strip() for s in acceptable):
                    raise ValueError("nonempty reviewed ingredients required")
                if not citations or not all(isinstance(u, str) and u.startswith("https://") for u in citations):
                    raise ValueError("source URLs required")
            counts.update(x.concerns)
            multi += int(len(x.concerns) > 1)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append({"id": identifier, "reason": str(exc)})
    for c in CONCERNS:
        if counts[c] < min_per_concern:
            errors.append({"concern": c, "reason": f"need {min_per_concern}, got {counts[c]}"})
    if multi < min_multi:
        errors.append({"reason": f"need {min_multi} multi-concern cases, got {multi}"})
    return {"ready": not errors, "rows": len(rows), "counts": dict(counts), "multi_concern": multi,
            "errors": errors, "provenance_status": "declared_by_collectors_not_independently_certified"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/sllm_v3")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--download-dictionary", action="store_true")
    ap.add_argument("--download-cosing", action="store_true", help="비공식 공개 사본의 이름만 대조. 효능/규제 승인으로 쓰지 않음")
    ap.add_argument("--external-cases", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    # 모든 과거 원본 프로필을 차단. 정제 때 제외된 원본도 과거 노출 기록에 포함된다.
    original = [r for s in ("train", "val") for r in read_rows(ROOT / f"data/processed/sllm/{s}.jsonl")]
    blocked = {serving_key(r) for r in original}
    reviews = read_rows(args.out / "claim_review_queue.jsonl")
    rare = {c for c, n in json.loads((args.data / "quality_report.json").read_text(encoding="utf-8"))["concern_train_unique"].items() if n < 150}
    candidates = [r for r in reviews if r["concern"] in rare or len(r["input"]["concerns"]) > 1]
    candidates.sort(key=lambda r: (r["concern"] not in rare, not bool(r["reasons"]), r["id"]))
    (args.out / "rare_multi_review_candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in candidates), encoding="utf-8")
    diagnostic = diagnostic_inputs(set(blocked))
    (args.out / "synthetic_diagnostic_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in diagnostic), encoding="utf-8")
    template = {"id": "NEW_CASE_0001", "collection_batch_id": "", "source_origin": "new_human_collection",
                "input": RecInput(gender="여성", age=30, skin_type="건성", concerns=[CONCERNS[0], CONCERNS[3]]).to_dict(),
                "gold_by_concern": {c: {"acceptable_ingredients": [], "evidence_urls": [], "rationale": "", "avoid_ingredients": []}
                                    for c in (CONCERNS[0], CONCERNS[3])},
                "review": {"status": "pending", "reviewer_id": "", "qualification": "", "reviewed_at": "", "claim_evidence_verified": False}}
    (args.out / "external_case_template.json").write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"rare_multi_existing_candidates": len(candidates), "rare_concerns": sorted(rare),
               "synthetic_input_only_cases": len(diagnostic), "new_reviewed_human_cases": 0,
               "independent_test_status": "waiting_for_new_collection_and_expert_labels", "training_data_modified": False}
    if args.download_dictionary:
        external, metadata = acquire_dictionary(args.out)
        lex = json.loads((args.data / "ingredient_dict.json").read_text(encoding="utf-8"))["lexicon"]
        matches = dictionary_matches(lex, external)
        (args.out / "external_ingredient_matches.json").write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["external_dictionary"] = metadata
        summary["dictionary_canonical_names"] = len(matches)
        summary["dictionary_exact_matched_names"] = sum(bool(r["exact_matches"]) for r in matches)
    if args.download_cosing:
        revision = "2978cd39d1856dc05b286b138c46fe65fd85ef6d"
        url = f"https://huggingface.co/datasets/nad949/CosIng/resolve/{revision}/COSING_Ingredients-Fragrance%20Inventory_v2.csv"
        dest = args.out / "external/cosing"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "ingredients.csv"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        external = [{**r, "inci_name": r["INCI name"].strip()} for r in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))]
        lex = json.loads((args.data / "ingredient_dict.json").read_text(encoding="utf-8"))["lexicon"]
        matches = dictionary_matches(lex, external)
        (args.out / "cosing_ingredient_matches.json").write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
        provenance = {"repository": "nad949/CosIng", "revision": revision, "source_url": url,
                      "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(external),
                      "license_status": "mirror_has_no_license_declaration; local_identity_review_only; not_redistributed",
                      "official_source": "https://single-market-economy.ec.europa.eu/sectors/cosmetics/cosmetic-ingredient-database_en",
                      "limitations": "unofficial_snapshot; INCI listing is not use approval or efficacy evidence"}
        (dest / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["cosing_dictionary"] = provenance
        summary["cosing_exact_matched_names"] = sum(bool(r["exact_matches"]) for r in matches)
    if args.external_cases:
        result = validate_external_cases(read_rows(args.external_cases), blocked)
        (args.out / "external_case_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["external_case_validation"] = result
        summary["new_reviewed_human_cases"] = len(read_rows(args.external_cases)) if result["ready"] else 0
    (args.out / "collection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
