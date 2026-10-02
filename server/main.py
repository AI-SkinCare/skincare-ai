"""스킨케어 AI 추론 서버.

실행 (PowerShell, 레포 루트):
  $env:AI_INTERNAL_TOKEN="<백엔드와 공유할 비밀값>"
  $env:REC_BACKEND="mock"     # 학습 전: 규칙 기반 / 학습 후: "sllm"
  uvicorn server.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.llm.inference import MockRecommender, SLLMRecommender
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter

from .adapters import build_rec_input, load_mapping
from .config import Settings, settings as default_settings
from .postprocess import structure_ingredients, structure_routine
from .schemas import (AnalyzeResponse, HealthResponse, ModelInfo, RecommendRequest, RecommendResponse,
                      SkinMetricsResponse, Survey)
from .skin_regressor import load_skin_regressor

log = logging.getLogger("skincare-ai")


# ── 에러 (개발명세서 §2.2, §2.3) ──
class AppError(Exception):
    def __init__(self, code: str, status: int, message: str, detail=None):
        self.code, self.status, self.message, self.detail = code, status, message, detail


def _err(code: str, status: int, message: str, detail=None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"success": False, "error": {"code": code, "message": message, "detail": detail}})


# ── 동시성 제어: GPU는 한 번에 한 생성만 효율적으로 처리 ──
class Gate:
    def __init__(self, concurrency: int, queue_limit: int):
        self.sem = asyncio.Semaphore(concurrency)
        self.waiting = 0
        self.queue_limit = queue_limit

    async def run(self, fn, *args, timeout: float | None = None):
        if self.waiting >= self.queue_limit:
            raise AppError("E3003", 503, "AI 서버 대기열이 가득 찼습니다. 잠시 후 다시 시도하세요.")
        self.waiting += 1
        try:
            await self.sem.acquire()
        finally:
            self.waiting -= 1
        task = asyncio.ensure_future(run_in_threadpool(fn, *args))
        # 타임아웃이 나도 GPU 작업은 끝까지 돈다 → 끝날 때 슬롯을 반납해야 다음 요청과 GPU를 다투지 않음
        task.add_done_callback(lambda _: self.sem.release())
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            raise AppError("E3002", 504, "추론 시간이 초과되었습니다.")


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or default_settings
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dict_path = cfg.processed_dir / "ingredient_dict.json"
        state["extractor"] = IngredientExtractor.load(dict_path) if dict_path.exists() else None
        top = cfg.processed_dir / "concern_top_ingredients.json"
        if cfg.rec_backend == "sllm":
            log.info("sLLM 로드 중: %s + %s", cfg.rec_base_model, cfg.rec_adapter_dir)
            state["rec"] = await run_in_threadpool(
                lambda: SLLMRecommender(cfg.rec_base_model, cfg.rec_adapter_dir,
                                        max_new_tokens=cfg.rec_max_new_tokens, top_path=top).load())
        elif cfg.rec_backend == "mock":
            state["rec"] = MockRecommender(top)
            if state["extractor"] is None:
                names = (name for names in state["rec"].rule.top.values() for name in names)
                state["extractor"] = IngredientExtractor(
                    {"".join(name.split()).upper(): name for name in names})
        else:
            raise ValueError(f"알 수 없는 REC_BACKEND: {cfg.rec_backend}")
        if state["extractor"] is None:
            log.warning("ingredient_dict.json 없음 → 성분 구조화 비활성 (prepare_data 실행 필요)")
        state["skin"] = load_skin_regressor(cfg.skin_backend)
        state["filter"] = OutputFilter.from_yaml()
        state["mapping"] = load_mapping()
        state["rec_gate"] = Gate(cfg.rec_concurrency, cfg.queue_limit)
        state["skin_gate"] = Gate(cfg.skin_concurrency, cfg.queue_limit)
        if not cfg.internal_token:
            log.warning("AI_INTERNAL_TOKEN 미설정 → 모든 보호 엔드포인트가 401을 반환합니다")
        yield
        state.clear()

    app = FastAPI(title="Skincare AI Inference Server", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(AppError)
    async def _app_err(_: Request, e: AppError):
        return _err(e.code, e.status, e.message, e.detail)

    @app.exception_handler(RequestValidationError)
    async def _val_err(_: Request, e: RequestValidationError):
        return _err("E4002", 422, "요청 형식이 올바르지 않습니다.", json.loads(json.dumps(e.errors(), default=str)))

    def auth(x_internal_token: str | None = Header(default=None)):
        # 토큰 미설정 시에도 닫힌 상태로 동작 (fail-closed)
        if not cfg.internal_token or not x_internal_token or not hmac.compare_digest(x_internal_token, cfg.internal_token):
            raise AppError("E1001", 401, "내부 인증 토큰이 없거나 올바르지 않습니다.")

    async def _read_image(image: UploadFile) -> bytes:
        data = await image.read()
        if len(data) > cfg.max_image_mb * 1024 * 1024:
            raise AppError("E2003", 413, f"이미지 용량은 {cfg.max_image_mb:.0f}MB 이하여야 합니다.")
        if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
            raise AppError("E2004", 415, "JPEG 또는 PNG 이미지만 지원합니다.")
        return data  # 디스크에 저장하지 않음 → 요청 종료와 함께 폐기

    async def _skin(data: bytes, device_type: str) -> SkinMetricsResponse:
        t0 = time.perf_counter()
        pred = await state["skin_gate"].run(state["skin"].predict, data, device_type, timeout=30)
        return SkinMetricsResponse(
            roi_detected=pred.roi_detected, metrics=pred.metrics, confidence=pred.confidence,
            calibration_applied=pred.calibration_applied, is_stub=pred.is_stub,
            model_version=state["skin"].version, inference_ms=int((time.perf_counter() - t0) * 1000))

    async def _recommend(req: RecommendRequest) -> RecommendResponse:
        try:
            x, notes = build_rec_input(req.survey, req.metrics, state["mapping"])
        except ValueError as e:
            raise AppError("E4002", 422, "설문/측정값을 해석할 수 없습니다.", str(e))
        res = await state["rec_gate"].run(state["rec"].recommend, x, timeout=cfg.rec_timeout_s)
        cleaned, rep = state["filter"].apply(res.parsed)
        if rep.changed:
            log.info("필터 개입: 문장 제거 %d, 출처 제거 %d, 완화 %d",
                     len(rep.dropped_sentences), rep.removed, rep.softened)
        return RecommendResponse(
            skin_type=x.skin_type, concerns=x.concerns,
            ingredients=structure_ingredients(cleaned, state["extractor"]),
            routine=structure_routine(cleaned),
            reasoning=[s["content"] for s in cleaned["reasoning"]],
            answer=cleaned["answer"], disclaimer=cleaned["disclaimer"],
            filtered=rep.changed, used_fallback=res.used_fallback, input_notes=notes,
            model_version=res.model_version, inference_ms=res.latency_ms)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        rec_ok, skin_ok = "rec" in state, "skin" in state
        return HealthResponse(status="ok" if rec_ok and skin_ok else "degraded", rec_loaded=rec_ok,
                              skin_loaded=skin_ok, rec_backend=cfg.rec_backend, skin_backend=cfg.skin_backend)

    @app.get("/model-info", response_model=ModelInfo, dependencies=[Depends(auth)])
    async def model_info():
        return ModelInfo(rec_model_version=state["rec"].version, skin_model_version=state["skin"].version,
                         rec_base_model=cfg.rec_base_model if cfg.rec_backend == "sllm" else None,
                         rec_adapter_dir=cfg.rec_adapter_dir if cfg.rec_backend == "sllm" else None)

    @app.post("/infer/skin-metrics", response_model=SkinMetricsResponse, dependencies=[Depends(auth)])
    async def skin_metrics(image: UploadFile = File(...), device_type: str = Form("smartphone")):
        return await _skin(await _read_image(image), device_type)

    @app.post("/infer/recommend", response_model=RecommendResponse, dependencies=[Depends(auth)])
    async def recommend(req: RecommendRequest):
        return await _recommend(req)

    @app.post("/infer/analyze", response_model=AnalyzeResponse, dependencies=[Depends(auth)])
    async def analyze(image: UploadFile = File(...), survey: str = Form(...),
                      device_type: str = Form("smartphone")):
        try:
            sv = Survey.model_validate_json(survey)
        except ValidationError as e:
            raise AppError("E4002", 422, "survey JSON 형식이 올바르지 않습니다.", json.loads(e.json()))
        skin = await _skin(await _read_image(image), device_type)
        rec = await _recommend(RecommendRequest(metrics=skin.metrics, survey=sv))
        return AnalyzeResponse(skin=skin, recommendation=rec)

    return app


app = create_app()
