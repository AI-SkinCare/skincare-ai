"""원본 processed/sllm에서 격리·중복 검사 후 별도 v3 데이터를 만든다.

python -m src.llm.prepare_quality_data
기존 데이터/어댑터는 보존하며 test는 새 실험의 최종 평가까지 열지 않는다.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
from pathlib import Path

from .clean_targets import TargetCleaner
from .ingredients import INGREDIENT_SCHEMA_VERSION, IngredientExtractor, harvest, save_lexicon
from .output_filter import OutputFilter
from .prepare_data import load_split
from .schema import CONCERNS, RecInput

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sllm_quality_v3.1"
TITLES = ("문제 분석 및 원인 파악", "성분 선택 및 근거 제시", "사용법 및 관리방안")
PLACEHOLDER = re.compile(r"12345678|10\.xxxx|xxxxx|10\.1234/example|placeholder", re.I)
# 이런 문장은 단어만 '관리'로 바꿔서 사실 검증을 대신할 수 없다.
CLINICAL = re.compile(r"치료|완치|처방|투약|복용|의약품|질환|질병|아토피|피부염|건선|습진|지루성|한포진|백반증|바르는\s*보톡스")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def serving_key(row: dict) -> str:
    """순서만 다른 같은 프로필도 동일 그룹. 코드·추가 질문은 기본 서빙에서 쓰지 않는다."""
    x = copy.deepcopy(row["input"])
    x["condition_codes"] = x["user_note"] = None
    x["concerns"] = sorted(x["concerns"])
    x["external_factors"] = sorted(x.get("external_factors", []), key=lambda e: (e["factor"], e.get("details", "")))
    return digest(x)


def target_text(target: dict) -> str:
    steps = target["reasoning"]
    # 원본 제목의 변형보다 학습 규약의 두 번째 단계를 우선한다.
    return (steps[1]["content"] if len(steps) > 1 else "") + "\n" + target["answer"]


def target_key(row: dict) -> str:
    return digest(re.sub(r"\s+", "", row["target"]))


def normalize_steps(target: dict) -> dict:
    target = copy.deepcopy(target)
    steps = target["reasoning"]
    if len(steps) < 3 or any(not s.get("content", "").strip() for s in steps) or not target["answer"].strip():
        raise ValueError("empty_or_short_target")
    # 추가 단계의 내용은 3단계에 합쳐 보존한다.
    steps[2]["content"] = "\n".join(s["content"] for s in steps[2:])
    target["reasoning"] = [{"step": i + 1, "title": title, "content": steps[i]["content"]}
                           for i, title in enumerate(TITLES)]
    return target


def build_dataset(train: list[dict], val: list[dict], old_dev_ids: set[str],
                  annotations: dict[str, dict] | None = None) -> dict:
    annotations = annotations or {}
    # 과거 oversampling 사본을 먼저 제거. 사전/빈도에 복제 횟수가 영향을 주지 않는다.
    originals = sorted((r for r in train if "#dup" not in r["id"]), key=lambda r: r["id"])
    if len({r["id"] for r in originals}) != len(originals):
        raise ValueError("duplicate source IDs")
    lex, freq = harvest([target_text(json.loads(r["target"])) for r in originals])
    ext = IngredientExtractor(lex)
    cleaner = TargetCleaner(OutputFilter.from_yaml(), lex)
    counters = collections.Counter()
    original_counts = collections.Counter(r["target_concern"] for r in originals)
    quarantined, review = [], []

    def clean_row(source):
        row = copy.deepcopy(source)
        reasons, flags = [], []
        try:
            RecInput.from_dict(row["input"]).validate()
            if row["target_concern"] not in row["input"]["concerns"]:
                reasons.append("target_concern_not_in_input")
            raw = json.loads(row["target"])
            full = "\n".join(s["content"] for s in raw["reasoning"]) + "\n" + raw["answer"]
            if CLINICAL.search(full):
                reasons.append("clinical_or_botox_claim_review")
            conflicts = cleaner.conflicting_pairs(full)
            if conflicts:
                reasons.append("conflicting_bilingual_names")
            target = normalize_steps(raw)
            before = set(ext.extract(target_text(target)))
            target = json.loads(cleaner.clean(json.dumps(target, ensure_ascii=False), counters))
            if any(not s["content"].strip() for s in target["reasoning"]) or not target["answer"].strip():
                reasons.append("empty_after_cleaning")
            after = ext.extract(target_text(target))
            lost = sorted(before - set(after))
            if lost:
                reasons.append("ingredient_lost_after_cleaning")
            if not after:
                reasons.append("empty_gold")
        except (ValueError, KeyError, TypeError) as e:
            reasons.append("invalid_structure:" + str(e))
            lost, conflicts = [], []
            target, after = {}, []
        evidence = annotations.get(row["id"], {}).get("evidence_sources")
        if evidence is None:
            flags.append("source_metadata_unavailable")
        elif PLACEHOLDER.search(json.dumps(evidence, ensure_ascii=False)):
            flags.append("placeholder_or_known_unrelated_reference")
        # 자리표시자 외의 출처도 이 처리에서 효능을 검증한 것은 아니다.
        flags.append("efficacy_not_expert_verified")
        if len(row["input"]["concerns"]) > 1:
            flags.append("multiple_concerns_single_target_label")
        if original_counts[row["target_concern"]] < 150:
            flags.append("rare_concern_review")
        entry = {"id": row["id"], "source_split": row["split"], "concern": row["target_concern"],
                 "reasons": reasons, "flags": flags, "evidence_sources": evidence,
                 "lost_ingredients": lost, "conflicting_names": conflicts, "original_target": row["target"],
                 "input": copy.deepcopy(row["input"]),
                 "review_priority": 0 if reasons or "rare_concern_review" in flags else 1 if len(flags) > 1 else 2}
        if reasons:
            quarantined.append(entry)
            return None
        row["target"] = json.dumps(target, ensure_ascii=False)
        row["gold_ingredients"] = after
        row["input"]["condition_codes"] = row["input"]["user_note"] = None
        row["quality_flags"] = flags
        if len(before) != len(set(source.get("gold_ingredients", []))) or set(after) != set(source.get("gold_ingredients", [])):
            counters["gold_rows_changed"] += 1
        review.append(entry)
        return row

    cleaned_train = [r for source in originals if (r := clean_row(source)) is not None]
    cleaned_val = [r for source in sorted(val, key=lambda r: r["id"]) if (r := clean_row(source)) is not None]
    excluded = []

    def unique(rows, split, blocked_inputs=(), blocked_targets=()):
        inputs, targets = set(blocked_inputs), set(blocked_targets)
        result = []
        for row in rows:
            ik, tk = serving_key(row), target_key(row)
            reason = "input_overlap" if split != "train" and ik in inputs else "target_duplicate" if tk in targets else None
            if reason:
                excluded.append({"id": row["id"], "candidate_split": split, "reason": reason})
                continue
            row["split"] = split
            result.append(row)
            inputs.add(ik)
            targets.add(tk)
        return result

    train_unique = unique(cleaned_train, "train")
    # 과거 학습에 사용됐던 입력도 최종 시험에 재사용하지 않도록 원본 전체를 차단.
    historical_train_inputs = {serving_key(r) for r in originals}
    historical_dev_inputs = {serving_key(r) for r in val if r["id"] in old_dev_ids}
    historical_train_targets = {target_key(r) for r in cleaned_train}
    dev = unique([r for r in cleaned_val if r["id"] in old_dev_ids], "val_eval",
                 historical_train_inputs, historical_train_targets)
    test = unique([r for r in cleaned_val if r["id"] not in old_dev_ids], "test",
                  historical_train_inputs | historical_dev_inputs,
                  historical_train_targets | {target_key(r) for r in cleaned_val if r["id"] in old_dev_ids})
    # v1/v2 모델 개발에는 원본 검증 전체가 이미 노출됐으므로 완전한 신규 외부 테스트는 아니다.
    groups = collections.defaultdict(list)
    for row in train_unique:
        groups[serving_key(row)].append(row)
    ambiguous = [rs for rs in groups.values() if len({tuple(sorted(r["gold_ingredients"])) for r in rs}) > 1]
    tops = collections.defaultdict(collections.Counter)
    for row in train_unique:
        tops[row["target_concern"]].update(row["gold_ingredients"])
    count = collections.Counter(r["target_concern"] for r in train_unique)
    train_final = list(train_unique)
    # 복제는 다양성을 늘리지 않으므로 v3은 최대 2배, 최소 50까지만 사용한다.
    for concern, n in sorted(count.items()):
        pool = [r for r in train_unique if r["target_concern"] == concern]
        for i in range(max(0, min(50, n * 2) - n)):
            duplicate = copy.deepcopy(pool[i % n])
            duplicate["id"] += f"#dup{i}"
            train_final.append(duplicate)
    splits = {"train": train_final, "val_eval": dev, "test": test}
    validate_splits(splits)
    report = {
        "version": VERSION, "ingredient_schema": INGREDIENT_SCHEMA_VERSION,
        "original_train": len(originals), "old_train_with_duplicates": len(train), "original_val": len(val),
        "train_unique": len(train_unique), "train_with_duplicates": len(train_final), "development": len(dev), "reserved_test": len(test),
        "quarantined": len(quarantined), "quarantine_reasons": dict(collections.Counter(reason for r in quarantined for reason in r["reasons"])),
        "quarantine_by_source_split": dict(collections.Counter(r["source_split"] for r in quarantined)),
        "excluded": dict(collections.Counter(r["candidate_split"] + ":" + r["reason"] for r in excluded)),
        "source_placeholder_records": sum("placeholder_or_known_unrelated_reference" in r["flags"] for r in review + quarantined),
        "cleaning": dict(counters), "ingredient_vocab": len(set(lex.values())),
        "conflicting_gold_input_groups": len(ambiguous), "conflicting_gold_input_rows": sum(map(len, ambiguous)),
        "concern_counts": {s: dict(collections.Counter(r["target_concern"] for r in rows)) for s, rows in splits.items()},
        "concern_train_unique": dict(count),
        "missing_test_concerns": [c for c in CONCERNS if not any(r["target_concern"] == c for r in test)],
        "limitations": ["efficacy_not_expert_verified", "original_validation_seen_in_previous_experiments", "rare_concerns_and_multiple_concerns_need_new_labels"],
    }
    return {"splits": splits, "lexicon": lex, "freq": freq,
            "top": {c: [name for name, _ in tops[c].most_common(10)] for c in CONCERNS},
            "report": report, "quarantine": quarantined, "excluded": excluded,
            "review": review, "ambiguous_input_groups": [[r["id"] for r in rs] for rs in ambiguous]}


def validate_splits(splits: dict[str, list[dict]]) -> None:
    sets = {s: ({serving_key(r) for r in rows}, {target_key(r) for r in rows}) for s, rows in splits.items()}
    for a, pair in sets.items():
        for b, other in sets.items():
            if a < b and (pair[0] & other[0] or pair[1] & other[1]):
                raise ValueError(f"split leakage: {a} / {b}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/processed/sllm")
    ap.add_argument("--dst", type=Path, default=ROOT / "data/processed/sllm_v3")
    ap.add_argument("--raw-root", type=Path, default=Path("D:/03.스킨케어 성분-효능 추천 데이터/3.개방데이터/2.데이터(NIA)"))
    args = ap.parse_args(argv)
    if args.src.resolve() == args.dst.resolve():
        raise ValueError("source and destination must differ")
    annotations = {}
    for split in ("Training", "Validation"):
        for r in load_split(args.raw_root / split / "02.라벨링데이터"):
            annotations[r["info"]["id"]] = r["info"]
    data = build_dataset(read_rows(args.src / "train.jsonl"), read_rows(args.src / "val.jsonl"),
                         {r["id"] for r in read_rows(args.src / "val_eval.jsonl")}, annotations)
    manifest = {"version": VERSION, "ingredient_schema": INGREDIENT_SCHEMA_VERSION,
                "source_sha256": {name: hashlib.sha256((args.src / name).read_bytes()).hexdigest()
                                  for name in ("train.jsonl", "val.jsonl", "val_eval.jsonl")},
                "source_annotations_hash": digest(annotations),
                "builder_source_sha256": {name: hashlib.sha256((ROOT / "src/llm" / name).read_bytes()).hexdigest()
                                          for name in ("prepare_quality_data.py", "prepare_data.py", "ingredients.py", "clean_targets.py", "schema.py", "output_filter.py")},
                "lexicon_hash": digest(data["lexicon"]), "filter_hash": hashlib.sha256((ROOT / "configs/filter_rules.yaml").read_bytes()).hexdigest(),
                "split_hashes": {s: digest(rows) for s, rows in data["splits"].items()},
                "test_status": "reserved_for_new_v3_experiments; historical_validation_not_blind_external_test"}
    manifest["dataset_id"] = digest(manifest)
    manifest_path = args.dst / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("existing dataset differs; choose a new --dst to preserve test membership")
    args.dst.mkdir(parents=True, exist_ok=True)
    for name, rows in {**data["splits"], "quarantine": data["quarantine"], "review_queue": data["review"], "excluded": data["excluded"]}.items():
        (args.dst / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    save_lexicon(args.dst / "ingredient_dict.json", data["lexicon"], data["freq"])
    for name, value in (("concern_top_ingredients", data["top"]), ("quality_report", data["report"]),
                        ("manifest", manifest), ("ambiguous_input_groups", data["ambiguous_input_groups"])):
        (args.dst / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data["report"], ensure_ascii=False, indent=2))
    print(f"[saved] {args.dst}")


if __name__ == "__main__":
    main()
