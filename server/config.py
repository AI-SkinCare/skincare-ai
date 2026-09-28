"""환경변수 기반 서버 설정. (.env.example 참고)"""
from __future__ import annotations

import os
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
    processed_dir: Path = field(default_factory=lambda: ROOT / _env("PROCESSED_DIR", "data/processed/sllm"))
    # GPU 생성은 사실상 직렬 → 추천 동시 처리 1, 대기열 초과 시 503
    rec_concurrency: int = field(default_factory=lambda: int(_env("REC_CONCURRENCY", "1")))
    skin_concurrency: int = field(default_factory=lambda: int(_env("SKIN_CONCURRENCY", "4")))
    queue_limit: int = field(default_factory=lambda: int(_env("QUEUE_LIMIT", "20")))
    rec_timeout_s: float = field(default_factory=lambda: float(_env("REC_TIMEOUT_S", "120")))
    max_image_mb: float = field(default_factory=lambda: float(_env("MAX_IMAGE_MB", "10")))


settings = Settings()
