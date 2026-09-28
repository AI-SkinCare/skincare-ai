"""AI 서버 요청/응답 스키마 (개발명세서 §5 기준)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class SkinMetricsResponse(BaseModel):
    roi_detected: list[str]
    metrics: dict[str, Metric4]
    confidence: float
    calibration_applied: bool
    is_stub: bool
    model_version: str
    inference_ms: int


class AnalyzeResponse(BaseModel):
    skin: SkinMetricsResponse
    recommendation: RecommendResponse


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rec_loaded: bool
    skin_loaded: bool
    rec_backend: str
    skin_backend: str


class ModelInfo(BaseModel):
    rec_model_version: str
    skin_model_version: str
    rec_base_model: str | None
    rec_adapter_dir: str | None
