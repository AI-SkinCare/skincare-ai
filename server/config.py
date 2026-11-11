"""환경변수 기반 서버 설정. (.env.example 참고)"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    internal_token: str | None = field(default_factory=lambda: _env("AI_INTERNAL_TOKEN"))
    rec_backend: str = field(default_factory=lambda: _env("REC_BACKEND", "mock"))          # mock | sllm
    rec_base_model: str = field(default_factory=lambda: _env("REC_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    rec_adapter_dir: str | None = field(default_factory=lambda: _env("REC_ADAPTER_DIR", str(ROOT / "outputs/rec_sllm_v2/adapter")))
    rec_max_new_tokens: int = field(default_factory=lambda: int(_env("REC_MAX_NEW_TOKENS", "2400")))
    skin_backend: str = field(default_factory=lambda: _env("SKIN_BACKEND", "stub"))         # stub | model
    # disabled | densenet; experimental image models remain outside this PR.
    disease_backend: str = field(default_factory=lambda: _env("DISEASE_BACKEND", "disabled"))
    disease_weights: Path = field(default_factory=lambda: ROOT / _env("DISEASE_WEIGHTS", "model/weights/densenet121_s2.pt"))
    disease_device: str = field(default_factory=lambda: _env("DISEASE_DEVICE", "cpu"))
    disease_threshold: float = field(default_factory=lambda: float(_env("DISEASE_THRESHOLD", "0.6")))
    disease_knowledge_path: Path = field(default_factory=lambda: ROOT / _env("DISEASE_KNOWLEDGE", "data/processed/disease_20261107/medical_knowledge.json"))
    processed_dir: Path = field(default_factory=lambda: ROOT / _env("PROCESSED_DIR", "data/processed/sllm"))
    # GPU 생성은 사실상 직렬 → 추천 동시 처리 1, 대기열 초과 시 503
    rec_concurrency: int = field(default_factory=lambda: int(_env("REC_CONCURRENCY", "1")))
    skin_concurrency: int = field(default_factory=lambda: int(_env("SKIN_CONCURRENCY", "4")))
    queue_limit: int = field(default_factory=lambda: int(_env("QUEUE_LIMIT", "20")))
    rec_timeout_s: float = field(default_factory=lambda: float(_env("REC_TIMEOUT_S", "120")))
    # Auto enables the bounded source path for real SLLM; mock stays unchanged.
    rec_evidence_mode: str = field(default_factory=lambda: _env("REC_EVIDENCE_MODE", "auto"))
    rec_evidence_path: Path = field(default_factory=lambda: ROOT / _env("REC_EVIDENCE_PATH", "configs/ingredient_evidence.json"))
    rec_evidence_use_adapter: bool = field(default_factory=lambda: _env("REC_EVIDENCE_USE_ADAPTER", "false").lower() == "true")
    max_image_mb: float = field(default_factory=lambda: float(_env("MAX_IMAGE_MB", "10")))

    def __post_init__(self):
        if self.rec_concurrency < 1 or self.skin_concurrency < 1 or self.queue_limit < 0:
            raise ValueError('Concurrency must be positive; QUEUE_LIMIT must be nonnegative')
        if not math.isfinite(self.rec_timeout_s) or self.rec_timeout_s <= 0:
            raise ValueError('REC_TIMEOUT_S must be finite and positive')
        if not math.isfinite(self.max_image_mb) or self.max_image_mb <= 0:
            raise ValueError('MAX_IMAGE_MB must be finite and positive')
        if self.rec_evidence_mode == 'auto':
            self.rec_evidence_mode = 'enforce' if self.rec_backend == 'sllm' else 'off'


settings = Settings()
