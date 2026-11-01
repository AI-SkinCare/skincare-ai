"""AI 서버 요청/응답 스키마 (개발명세서 §5 기준)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Metric4 = dict[str, float]  # {"moisture","elasticity","pigmentation","pore"} → 0~1


class Survey(BaseModel):
    """개발명세서 §4.2 설문 + 추천 품질에 필요한 선택 필드(age, skinType, stressLevel, note)."""
    model_config = ConfigDict(extra="ignore")

    gender: Literal["male", "female", "남성", "여성"]
    ageBand: str | None = None
    age: int | None = Field(None, ge=10, le=100)
    skinType: str | None = None
    sleepHours: float | None = None
    waterIntakeLevel: Literal["low", "medium", "high"] | None = None
    sunExposureLevel: Literal["low", "medium", "high"] | None = None
    stressLevel: Literal["low", "medium", "high"] | None = None
    currentProducts: list[str] = []
    selfReportedConcerns: list[str] = []
    note: str | None = Field(None, max_length=300)


class RecommendRequest(BaseModel):
    metrics: dict[str, Metric4] | None = None
    survey: Survey


class Ingredient(BaseModel):
    name: str
    purpose: str | None = None
    priority: int
    purpose_status: Literal["provided", "unavailable"] = "unavailable"
    purpose_source: Literal["model", "model_shared", "unavailable"] = "unavailable"


class RoutineStep(BaseModel):
    step: int
    time: Literal["morning", "evening", "daily", "weekly", "anytime"]
    action: str


class RecommendResponse(BaseModel):
    skin_type: str
    concerns: list[str]
    ingredients: list[Ingredient]
    routine: list[RoutineStep]
    reasoning: list[str]
    answer: str
    disclaimer: str
    filtered: bool
    used_fallback: bool
    input_notes: list[str]
    model_version: str
    inference_ms: int
    quality_status: Literal["ok", "limited", "withheld"] = "ok"
    quality_issues: list[str] = Field(default_factory=list)
    postprocess_version: str = "recommendation_quality_v1"


class SkinMetricsResponse(BaseModel):
    roi_detected: list[str]
    metrics: dict[str, Metric4]
    confidence: float
    calibration_applied: bool
    is_stub: bool
    model_version: str
    inference_ms: int


DISEASE_LABELS = {
    "psoriasis": "건선", "atopic": "아토피", "acne": "여드름", "normal": "정상",
    "rosacea": "주사", "seborrheic": "지루",
}
DiseaseLabel = Literal["건선", "아토피", "여드름", "정상", "주사", "지루"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DiseasePrediction(BaseModel):
    """분류기 원본 JSON + 추적 정보. 점수는 임상적 질환 확률이 아니다."""
    label: DiseaseLabel
    label_en: Literal["psoriasis", "atopic", "acne", "normal", "rosacea", "seborrheic"]
    confidence: Probability
    uncertain: bool
    probabilities: dict[DiseaseLabel, Probability]
    model_version: str = "external"
    inference_ms: int = Field(0, ge=0)

    @model_validator(mode="after")
    def consistent_classification(self):
        if DISEASE_LABELS[self.label_en] != self.label:
            raise ValueError("label과 label_en이 일치하지 않습니다")
        if set(self.probabilities) != set(DISEASE_LABELS.values()):
            raise ValueError("6종 클래스의 확률이 모두 필요합니다")
        if abs(sum(self.probabilities.values()) - 1) > 0.001:
            raise ValueError("클래스 확률의 합이 1이어야 합니다")
        if abs(self.probabilities[self.label] - self.confidence) > 0.0002:
            raise ValueError("confidence와 선택 클래스의 확률이 일치하지 않습니다")
        if max(self.probabilities.values()) > self.confidence + 0.0002:
            raise ValueError("선택 클래스가 가장 높은 확률이어야 합니다")
        return self


class DiseaseRecommendRequest(BaseModel):
    disease: DiseasePrediction
    survey: Survey
    metrics: dict[str, Metric4] | None = None


class DiseaseAnalyzeResponse(BaseModel):
    disease: DiseasePrediction
    recommendation: RecommendResponse | None
    recommendation_status: Literal["generated", "uncertain", "no_concern", "quality_rejected"]
    guidance: list[str]
    quality_issues: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    skin: SkinMetricsResponse
    recommendation: RecommendResponse | None
    disease: DiseasePrediction | None = None
    recommendation_status: Literal["generated", "uncertain", "no_concern", "quality_rejected"] = "generated"
    guidance: list[str] = Field(default_factory=list)
    quality_issues: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rec_loaded: bool
    skin_loaded: bool
    rec_backend: str
    skin_backend: str
    disease_loaded: bool = False
    disease_backend: str = "disabled"


class ModelInfo(BaseModel):
    rec_model_version: str
    skin_model_version: str
    rec_base_model: str | None
    rec_adapter_dir: str | None
    disease_model_version: str | None = None
