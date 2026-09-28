"""AI허브 「스킨케어 성분-효능 추천 데이터」 라벨(zip) → 학습용 JSONL.

사용 (PowerShell):
  python -m src.llm.prepare_data --raw-root "D:\\03.스킨케어 성분-효능 추천 데이터\\3.개방데이터\\2.데이터(NIA)"

산출물 (data/processed/sllm/, git 제외):
  train.jsonl / val.jsonl / val_eval.jsonl, ingredient_dict.json,
  concern_top_ingredients.json, stats.json
  + docs/label_spec.md (실제 데이터에서 확인한 라벨 명세, 커밋 대상)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import zipfile
from pathlib import Path

import yaml

from .ingredients import IngredientExtractor, harvest, save_lexicon
from .schema import CONCERNS, ExternalFactor, RecInput, build_target

ROOT = Path(__file__).resolve().parents[2]


def _fix_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except UnicodeError:
        return info.filename


def load_split(label_dir: Path) -> list[dict]:
    recs = []
    for z in sorted(label_dir.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.is_dir() or not _fix_name(info).endswith(".jsonl"):
                    continue
                for line in zf.read(info).decode("utf-8-sig").splitlines():
                    if line.strip():
                        recs.append(json.loads(line))
    return recs


def to_input(r: dict, keep_codes: bool, user_note: str | None) -> RecInput:
    m = r["meta"]
    x = RecInput(
        gender=m["gender"],
        age=int(m["age"]),
        skin_type=m["skin_type"],
        concerns=list(m["skin_concerns"]),
        condition_codes=m.get("initial_skin_condition") if keep_codes else None,
        external_factors=[
            ExternalFactor(e["factor"], e.get("details", ""))
            for e in sorted(r.get("external", []), key=lambda e: e.get("priority", 99))
        ],
        user_note=user_note,
    )
    x.validate()
    return x


def ingredient_text(r: dict) -> str:
    """정답 성분 추출 대상: 2단계(성분 선택) + 최종 답변."""
    steps = r["chain_of_thought"]
    step2 = next((s["content"] for s in steps if "성분" in s.get("title", "")), steps[1]["content"] if len(steps) > 1 else "")
    return step2 + "\n" + r["info"]["answer"]


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/sllm_qlora.yaml"))
    ap.add_argument("--raw-root", help="…/3.개방데이터/2.데이터(NIA) 경로 (config 값 덮어쓰기)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    raw_root = Path(args.raw_root or dcfg["raw_root"])
    out = ROOT / dcfg["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(dcfg["seed"])

    train_raw = load_split(raw_root / "Training" / "02.라벨링데이터")
    val_raw = load_split(raw_root / "Validation" / "02.라벨링데이터")
    print(f"원본 로드: train {len(train_raw)} / val {len(val_raw)}")
    if not train_raw:
        sys.exit(f"라벨을 찾지 못했습니다: {raw_root}")

    # 1) 성분 사전: train 말뭉치로만 구축 (val 정보가 새지 않도록)
    lexicon, freq = harvest([ingredient_text(r) for r in train_raw])
    save_lexicon(out / "ingredient_dict.json", lexicon, freq)
    ext = IngredientExtractor(lexicon)

    def gold(r):
        return ext.extract(ingredient_text(r))

    cover = sum(bool(gold(r)) for r in train_raw) / len(train_raw)
    print(f"성분 사전 {len(freq)}종 / 정답 성분 추출 커버리지(train) {cover:.1%}")

    # 2) 고민별 빈출 성분 (규칙 기반 폴백 + 베이스라인)
    by_concern: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train_raw:
        for g in gold(r):
            by_concern[r["info"]["target_concern"]][g] += 1
    top = {c: [n for n, _ in by_concern[c].most_common(10)] for c in CONCERNS}
    (out / "concern_top_ingredients.json").write_text(json.dumps(top, ensure_ascii=False, indent=1), encoding="utf-8")

    # 베이스라인: "고민별 빈출 성분 Top-3"을 그대로 추천했을 때의 일치율
    hits = n = 0
    for r in val_raw:
        g = set(gold(r))
        if not g:
            continue
        pred = top.get(r["info"]["target_concern"], [])[:3]
        hits += len(set(pred) & g)
        n += len(pred)
    baseline_p3 = hits / n if n else 0.0
    print(f"베이스라인(고민별 빈출 Top-3) precision@3 = {baseline_p3:.1%}")

    # 3) 학습 레코드 생성
    def rec(r, split, keep_codes, note):
        x = to_input(r, keep_codes, note)
        return {
            "id": r["info"]["id"],
            "split": split,
            "target_concern": r["info"]["target_concern"],
            "input": x.to_dict(),
            "target": build_target(r["chain_of_thought"], r["info"]["answer"]),
            "gold_ingredients": gold(r),
        }

    train = []
    for r in train_raw:
        keep = rng.random() >= dcfg["condition_code_dropout"]
        note = r["info"]["question"] if rng.random() < dcfg["user_note_prob"] else None
        train.append(rec(r, "train", keep, note))

    # 희소 고민 오버샘플링 (복제 상한을 둬서 6건짜리 클래스를 수십 번 외우지 않게)
    cnt = collections.Counter(t["target_concern"] for t in train)
    extra = []
    for c, n_c in cnt.items():
        want = min(dcfg["oversample_min"], n_c * dcfg["oversample_max_factor"])
        pool = [t for t in train if t["target_concern"] == c]
        for i in range(max(0, want - n_c)):
            dup = dict(pool[i % len(pool)])
            dup["id"] = f"{dup['id']}#dup{i}"
            extra.append(dup)
    train += extra
    rng.shuffle(train)

    # 검증: 서빙과 같은 조건(추가 요청 없음). 상태 코드는 원본 유지, 서빙 정책은 평가 시 적용
    val = [rec(r, "val", True, None) for r in val_raw]

    # 빠른 평가용 층화 샘플: 희소 고민은 전부, 나머지는 비율대로
    by_c = collections.defaultdict(list)
    for v in val:
        by_c[v["target_concern"]].append(v)
    size = dcfg["val_eval_size"]
    rare = [v for c, vs in by_c.items() if len(vs) <= 20 for v in vs]
    common = {c: vs for c, vs in by_c.items() if len(vs) > 20}
    tot_common = sum(len(vs) for vs in common.values())
    val_eval = list(rare)
    for c, vs in common.items():
        k = round((size - len(rare)) * len(vs) / tot_common)
        val_eval += rng.sample(vs, min(k, len(vs)))

    for name, rows in (("train", train), ("val", val), ("val_eval", val_eval)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "train_raw": len(train_raw),
        "train_after_oversample": len(train),
        "val": len(val),
        "val_eval": len(val_eval),
        "concern_train_raw": dict(cnt),
        "concern_train_final": dict(collections.Counter(t["target_concern"] for t in train)),
        "concern_val": {c: len(vs) for c, vs in by_c.items()},
        "concern_val_eval": dict(collections.Counter(v["target_concern"] for v in val_eval)),
        "ingredient_vocab": len(freq),
        "gold_ingredient_coverage_train": round(cover, 4),
        "baseline_precision_at_3": round(baseline_p3, 4),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    write_label_spec(train_raw + val_raw, stats, top)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def write_label_spec(recs: list[dict], stats: dict, top: dict) -> None:
    C = collections.Counter
    lines = ["# 라벨 명세 — 스킨케어 성분-효능 추천 데이터", "",
             "> `python -m src.llm.prepare_data`가 실제 데이터에서 자동 생성한 문서입니다. 직접 수정하지 마세요.", "",
             f"- 전체 {len(recs):,}건 (train {stats['train_raw']:,} / val {stats['val']:,})", ""]
    lines += ["## 피부 타입 (4종)", "", "| 값 | 건수 |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in C(r["meta"]["skin_type"] for r in recs).most_common()]
    lines += ["", "## 피부 고민 (8종, target_concern 기준)", "", "| 값 | 전체 | 학습(원본) | 검증 |", "|---|---|---|---|"]
    tc = C(r["info"]["target_concern"] for r in recs)
    for k, v in tc.most_common():
        lines.append(f"| {k} | {v:,} | {stats['concern_train_raw'].get(k, 0):,} | {stats['concern_val'].get(k, 0):,} |")
    lines += ["", "> 민감성·피부처짐·과각질은 표본이 매우 적습니다(검증셋 0~2건). 이 고민들의 성능 수치는 통계적으로 의미가 없으니 발표 시 별도 표기해야 합니다.", ""]
    lines += ["## 외부 요인 (7종)", "", "| 값 | 건수 |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in C(e["factor"] for r in recs for e in r["external"]).most_common()]
    lines += ["", "## initial_skin_condition (8자리 코드)", "",
              "데이터 설명서에 코드 정의가 없어, 고민 라벨과의 상관(lift)으로 의미를 검증했습니다.", "",
              "| 자리 | 값 | 추정 의미 | 근거 | 상태 |", "|---|---|---|---|---|",
              "| 1 | NS/S | 미상 (민감/피지?) | 지성·복합성에서 S 비율 높음, 민감성 고민과 무관(1.06) | 미확인 |",
              "| 2 | P0/P1/P2 | 색소침착 등급으로 추정 | P2가 미백 고민에서 2.77배, **모공 고민에서는 0.34배** → 모공 아님 | 추정 |",
              "| 3 | W0/W1/W2 | 주름 등급 | W2가 주름 고민에서 4.61배 | 확인 |",
              "| 4 | NA/A | 여드름 | A가 여드름 고민에서 4.56배 | 확인 |",
              "| 5 | VP/NVP | 미상 | 어떤 고민과도 상관 없음(0.9~1.1) | 미확인 |",
              "| 6 | SA/NSA | 미상 (노화/처짐?) | 주름 1.86배, 처짐 1.64배, 건성에서 높음 | 미확인 |",
              "| 7 | ND/D | 건조/각질 | D가 과각질 고민에서 43.5배 | 확인 |",
              "| 8 | NR/R | 홍조 | R이 붉어짐 고민에서 3.14배 | 확인 |", "",
              "**결론**: 서빙 시 측정값으로 이 코드를 추측해서 채우지 않습니다(`configs/serving_mapping.yaml`의 `use_condition_codes: false`). "
              "학습 시 30% 확률로 코드를 빼서, 코드 없이도 추천하도록 학습합니다.", ""]
    lines += ["## 고민별 빈출 성분 Top-5 (규칙 기반 폴백에 사용)", ""]
    for c in CONCERNS:
        lines.append(f"- **{c}**: {', '.join(top.get(c, [])[:5]) or '(없음)'}")
    lines += ["", "## 데이터 품질 이슈", "",
              "- `evidence_sources`의 약 75%가 `PMID:12345678`, `DOI:10.xxxx/xxxxx` 같은 자리표시자 → 학습 정답에서 제외 (모델이 가짜 출처를 생성하지 않도록).",
              "- 성분 표기가 `영문(한글)`, `한글(영문)`, `**굵게**`, 한글 단독 등으로 혼재 → 사전 기반 추출기 사용 (`src/llm/ingredients.py`).",
              f"- 정답 성분 추출 커버리지: {stats['gold_ingredient_coverage_train']:.1%}", ""]
    p = ROOT / "docs" / "label_spec.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
