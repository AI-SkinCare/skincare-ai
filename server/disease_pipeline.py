"""분류 결과 → 기존 추천 모델 입력. 질환명은 진단이 아닌 참고 정보로 전달한다."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.llm.schema import CONCERNS, RecInput
from src.llm.output_filter import OutputFilter

from .adapters import build_rec_input, normalize_concerns
from .schemas import DISEASE_LABELS, DiseasePrediction, Survey

ROOT = Path(__file__).resolve().parents[1]
_SAFETY_FILTER = OutputFilter({
    "drop_sentence_terms": ["안전하게 사용", "안전합니다", "부작용이 없", "자극이 없", "알레르기가 없"],
})


def filter_disease_output(parsed: dict) -> tuple[dict, bool]:
    """질환용 효능/안전성 검증을 받지 않은 어댑터의 무조건적 안전 보장을 제거한다."""
    cleaned, report = _SAFETY_FILTER.apply(parsed)
    cleaned["disclaimer"] = parsed["disclaimer"]
    return cleaned, report.changed


def load_disease_mapping(path: Path | None = None) -> dict:
    data = yaml.safe_load((path or ROOT / "configs/disease_mapping.yaml").read_text(encoding="utf-8"))
    if set(data["labels"]) != set(DISEASE_LABELS):
        raise ValueError("질환 매핑은 분류기의 6개 라벨과 일치해야 합니다")
    for rule in data["labels"].values():
        if any(c not in CONCERNS for c in rule["concerns"]):
            raise ValueError("질환 매핑에 Qwen이 지원하지 않는 고민이 있습니다")
    return data["labels"]


def build_disease_rec_input(survey: Survey, metrics: dict | None, disease: DiseasePrediction,
                            mapping: dict, disease_mapping: dict, threshold: float
                            ) -> tuple[RecInput | None, list[str], str]:
    if disease.uncertain or disease.confidence < threshold:
        return None, ["분류가 불확실해 자동 성분 추천을 보류했습니다. 재촬영 또는 피부과 전문의 상담을 권합니다."], "uncertain"
    rule = disease_mapping[disease.label_en]
    # 질환에서 도출한 고민을 앞에 배치: 기존 max_concerns 제한 때문에 밀려나지 않게 한다.
    concerns = list(dict.fromkeys(rule["concerns"] + normalize_concerns(survey.selfReportedConcerns, mapping)))
    if not concerns and not metrics:
        return None, ["정상으로 분류됐으며 선택한 고민이 없어 성분 추천을 생략했습니다. 필요하면 고민을 선택해 주세요."], "no_concern"
    sv = survey.model_copy(update={"selfReportedConcerns": concerns})
    x, notes = build_rec_input(sv, metrics, mapping)
    context = (
        f"이미지 분류 참고(진단 아님): {disease.label}, 분류 점수 {disease.confidence:.3f}. "
        "이 점수는 질환이 있을 임상적 확률이 아닙니다. "
        "질환의 치료 성분이나 처방이 아닌 화장품 수준의 보조 관리만 안내하세요. "
        + rule["care_note"]
    )
    x.user_note = context + (f"\n사용자 추가 요청: {survey.note}" if survey.note else "")
    notes.insert(0, f"이미지 분류 '{disease.label}'를 스킨케어 고민으로 변환: {', '.join(rule['concerns']) or '자가응답 사용'}")
    return x, notes, "generated"


def disease_guidance(disease: DiseasePrediction) -> list[str]:
    out = ["이미지 모델의 6종 분류 참고 결과이며 의학적 진단이 아닙니다. 분류 점수만으로 다른 질환을 배제할 수 없습니다."]
    if disease.label_en != "normal":
        out.append("성분 추천은 화장품 수준의 보조 관리 정보입니다. 질환 확인과 치료는 피부과 전문의와 상담하세요.")
    return out
