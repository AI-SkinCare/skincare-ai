"""(설문 + 피부 측정값) → 추천 모델 입력(RecInput) 변환.

학습 데이터에는 '측정값'이 없고 '피부 타입·고민 텍스트'만 있다(두 데이터셋의 피험자가 다름).
그래서 서빙 때 측정값을 학습 데이터 어휘(8종 고민 등)로 번역하는 층이 반드시 필요하고,
그 규칙을 이 파일과 configs/serving_mapping.yaml 한 곳에 모아 둔다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.llm.schema import CONCERNS, ExternalFactor, RecInput

from .schemas import Survey

ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = ("moisture", "elasticity", "pigmentation", "pore")


def load_mapping(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "configs" / "serving_mapping.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def summarize_metrics(metrics: dict[str, dict[str, float]] | None) -> dict[str, float] | None:
    """full_face가 있으면 그 값을, 없으면 ROI 평균을 쓴다."""
    if not metrics:
        return None
    if "full_face" in metrics:
        return {k: float(metrics["full_face"][k]) for k in METRIC_KEYS if k in metrics["full_face"]}
    acc: dict[str, list[float]] = {}
    for roi in metrics.values():
        for k in METRIC_KEYS:
            if k in roi:
                acc.setdefault(k, []).append(float(roi[k]))
    return {k: sum(v) / len(v) for k, v in acc.items()} or None


def _match(text: str, table: dict[str, list[str]]) -> str | None:
    t = text.strip().lower()
    for label, words in table.items():
        if t == label.lower() or any(w.lower() in t for w in words):
            return label
    return None


def build_rec_input(survey: Survey, metrics: dict[str, dict[str, float]] | None,
                    mapping: dict[str, Any]) -> tuple[RecInput, list[str]]:
    notes: list[str] = []  # 응답에 포함해 백엔드/프론트가 추정 여부를 알 수 있게 함
    m = summarize_metrics(metrics)

    gender = {"male": "남성", "female": "여성"}.get(survey.gender, survey.gender)

    if survey.age:
        age = survey.age
    elif survey.ageBand and survey.ageBand in mapping["age_band_midpoint"]:
        age = mapping["age_band_midpoint"][survey.ageBand]
        notes.append(f"나이는 연령대({survey.ageBand}) 중간값 {age}세로 추정")
    else:
        age = 30
        notes.append("나이 정보 없음 → 30세로 가정")

    skin_type = _match(survey.skinType, mapping["skin_type_synonyms"]) if survey.skinType else None
    if not skin_type:
        thr = mapping["skin_type_fallback"]["moisture_dry_below"]
        skin_type = "건성" if m and m.get("moisture", 1.0) <= thr else "중성"
        notes.append(f"피부 타입 미입력 → 측정값 기반 추정({skin_type}, 정확도 낮음)")

    # 고민: 사용자 자가응답 우선, 다음으로 측정값에서 심각도 순
    concerns: list[str] = []
    for c in survey.selfReportedConcerns:
        label = _match(c, mapping["concern_synonyms"])
        if label and label not in concerns:
            concerns.append(label)
    if m:
        scored = []
        for key, rule in mapping["metric_concerns"].items():
            if key not in m:
                continue
            v, thr = m[key], rule["threshold"]
            sev = (v - thr) if rule["direction"] == "high" else (thr - v)
            if sev > 0:
                scored.append((sev, rule["concern"]))
        for _, label in sorted(scored, reverse=True):
            if label not in concerns:
                concerns.append(label)
                notes.append(f"측정값 기반 고민 추가: {label}")
    if not concerns:
        if m:  # 임계값을 넘은 게 없으면 상대적으로 가장 나쁜 지표
            worst = max(mapping["metric_concerns"].items(),
                        key=lambda kv: (m.get(kv[0], 0) if kv[1]["direction"] == "high" else 1 - m.get(kv[0], 1)))
            concerns = [worst[1]["concern"]]
        else:
            concerns = ["모공"]
        notes.append(f"뚜렷한 고민 없음 → '{concerns[0]}'로 기본 설정")
    concerns = [c for c in concerns if c in CONCERNS][: mapping["max_concerns"]]

    ext: list[ExternalFactor] = []
    rules = mapping["external_rules"]
    for fld in ("sunExposureLevel", "stressLevel", "waterIntakeLevel"):
        val = getattr(survey, fld)
        if val and fld in rules and val in rules[fld]:
            factor, details = rules[fld][val]
            ext.append(ExternalFactor(factor, details))
    if survey.sleepHours is not None and survey.sleepHours < rules["sleepHours_below"]["value"]:
        factor, details = rules["sleepHours_below"]["factor"]
        ext.append(ExternalFactor(factor, details))
    if survey.currentProducts:
        ext.append(ExternalFactor("화장품", "현재 사용: " + ", ".join(survey.currentProducts[:5])))

    x = RecInput(gender=gender, age=int(age), skin_type=skin_type, concerns=concerns,
                 condition_codes=None, external_factors=ext[:4], user_note=survey.note)
    x.validate()
    return x, notes
