"""피부 상태 회귀 모델 인터페이스 (담당: 박성근).

박성근 모델이 준비되면 SkinRegressor 규약(version, predict)만 맞춘 클래스를
src/models/ 아래에 만들고 load_skin_regressor()에서 반환하도록 연결하면 된다.
그 전까지는 StubSkinRegressor가 '가짜임을 명시한' 값을 돌려준다(is_stub=True).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

ROIS = ("full_face", "forehead", "glabella", "eye_left", "eye_right", "cheek_left", "cheek_right", "lip", "chin")


@dataclass
class SkinPrediction:
    roi_detected: list[str]
    metrics: dict[str, dict[str, float]]   # ROI → {moisture, elasticity, pigmentation, pore} (0~1)
    confidence: float
    calibration_applied: bool
    is_stub: bool


class SkinRegressor(Protocol):
    version: str

    def predict(self, image_bytes: bytes, device_type: str) -> SkinPrediction: ...


class StubSkinRegressor:
    """연동 개발용. 같은 이미지엔 항상 같은 값을 주도록 해시에서 값을 만든다."""

    version = "skin_reg_stub"

    def predict(self, image_bytes: bytes, device_type: str) -> SkinPrediction:
        h = hashlib.sha256(image_bytes).digest()
        base = {k: round(0.25 + (h[i] / 255) * 0.5, 3)
                for i, k in enumerate(("moisture", "elasticity", "pigmentation", "pore"))}
        return SkinPrediction(["full_face"], {"full_face": base}, 0.0, False, True)


def load_skin_regressor(backend: str) -> SkinRegressor:
    if backend == "stub":
        return StubSkinRegressor()
    if backend == "model":
        # TODO(박성근): from src.models.skin_regressor import SkinRegressionModel; return SkinRegressionModel(...).load()
        raise NotImplementedError("회귀 모델 미연결: SKIN_BACKEND=stub 으로 실행하세요")
    raise ValueError(f"알 수 없는 SKIN_BACKEND: {backend}")
