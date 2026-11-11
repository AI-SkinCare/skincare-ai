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
from .disease_classifier import DenseNetDiseaseClassifier, InvalidDiseaseImage
from .medical_context import load_medical_knowledge, medical_reference, reference_prompt
from .ingredient_evidence import (EVIDENCE_GUIDANCE, load_evidence, extend_extractor,
                                  retrieve_evidence, evidence_prompt, evidence_review, enforce_evidence)
from .grounded_generation import generate_grounded
from .disease_pipeline import build_disease_rec_input, disease_guidance, load_disease_mapping
from .recommendation_quality import SERVING_GUIDANCE
from .recommendation_response import prepare_recommendation
from .schemas import (AnalyzeResponse, DiseaseAnalyzeResponse, DiseasePrediction, DiseaseRecommendRequest,
                      HealthResponse, ModelInfo, RecommendRequest, RecommendResponse, SkinMetricsResponse, Survey)
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
        if concurrency < 1 or queue_limit < 0:
            raise ValueError('concurrency must be positive and queue_limit nonnegative')
        self.sem = asyncio.Semaphore(concurrency)
        self.waiting = 0
        self.queue_limit = queue_limit

    async def run(self, fn, *args, timeout: float | None = None):
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        remaining = lambda: None if deadline is None else max(0, deadline - loop.time())
        try:
            if self.sem.locked():
                if self.waiting >= self.queue_limit:
                    raise AppError("E3003", 503, "AI 서버 대기열이 가득 찼습니다. 잠시 후 다시 시도하세요.")
                self.waiting += 1
                try:
                    await asyncio.wait_for(self.sem.acquire(), remaining())
                finally:
                    self.waiting -= 1
            else:
                await self.sem.acquire()
            if deadline is not None and remaining() <= 0:
                self.sem.release()
                raise asyncio.TimeoutError
            task = asyncio.ensure_future(run_in_threadpool(fn, *args))

            def completed(done):
                # A timed-out/cancelled HTTP request cannot stop its GPU thread.
                # Retain its slot until actual completion and consume late errors.
                self.sem.release()
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(completed)
            return await asyncio.wait_for(asyncio.shield(task), remaining())
        except asyncio.TimeoutError:
            raise AppError("E3002", 504, "추론 시간이 초과되었습니다.")


def create_app(cfg: Settings | None = None, *, disease_classifier_factory=None) -> FastAPI:
    cfg = cfg or default_settings
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dict_path = cfg.processed_dir / "ingredient_dict.json"
        state["extractor"] = IngredientExtractor.load(dict_path) if dict_path.exists() else None
        top = cfg.processed_dir / "concern_top_ingredients.json"
        if cfg.rec_backend == "sllm":
            if state["extractor"] is None:
                raise ValueError("ingredient_dict.json이 없어 성분 추천을 구성할 수 없습니다. PROCESSED_DIR을 확인하세요.")
            log.info("sLLM 로드 중: %s + %s", cfg.rec_base_model, cfg.rec_adapter_dir)
            state["rec"] = await run_in_threadpool(
                lambda: SLLMRecommender(cfg.rec_base_model, cfg.rec_adapter_dir,
                                        max_new_tokens=cfg.rec_max_new_tokens, top_path=top,
                                        serving_guidance=SERVING_GUIDANCE).load())
        elif cfg.rec_backend == "mock":
            state["rec"] = MockRecommender(top)
            if state["extractor"] is None:
                # 데이터 없는 새 clone에서도 규칙 기반 개발 응답을 구조화한다.
                names = (name for names in state["rec"].rule.top.values() for name in names)
                state["extractor"] = IngredientExtractor({"".join(name.split()).upper(): name for name in names})
        else:
            raise ValueError(f"알 수 없는 REC_BACKEND: {cfg.rec_backend}")
        if state["extractor"] is None:
            raise ValueError("ingredient_dict.json이 없어 성분 추천을 구성할 수 없습니다. PROCESSED_DIR을 확인하세요.")
        state["skin"] = load_skin_regressor(cfg.skin_backend)
        state["filter"] = OutputFilter.from_yaml()
        state["mapping"] = load_mapping()
        state["rec_gate"] = Gate(cfg.rec_concurrency, cfg.queue_limit)
        state["skin_gate"] = Gate(cfg.skin_concurrency, cfg.queue_limit)
        state["disease_mapping"] = load_disease_mapping()
        state["medical_knowledge"] = load_medical_knowledge(cfg.disease_knowledge_path)
        if cfg.rec_evidence_mode not in ('off', 'context', 'enforce'):
            raise ValueError('REC_EVIDENCE_MODE must be off, context, or enforce')
        state['ingredient_evidence'] = load_evidence(cfg.rec_evidence_path) if cfg.rec_evidence_mode != 'off' else None
        if state['ingredient_evidence'] is not None:
            state['extractor'] = extend_extractor(state['extractor'], state['ingredient_evidence'])
        if not 0 <= cfg.disease_threshold <= 1:
            raise ValueError("DISEASE_THRESHOLD는 0~1이어야 합니다")
        if disease_classifier_factory is not None:
            state['disease'] = await run_in_threadpool(disease_classifier_factory)
        elif cfg.disease_backend == "densenet":
            state["disease"] = await run_in_threadpool(
                lambda: DenseNetDiseaseClassifier(cfg.disease_weights, cfg.disease_device,
                                                   cfg.disease_threshold).load())
        elif cfg.disease_backend != "disabled":
            raise ValueError(f"알 수 없는 DISEASE_BACKEND: {cfg.disease_backend}")
        # GPU를 명시한 경우 Qwen과 같은 슬롯 사용. 기본 CPU는 독립 직렬 처리.
        state["disease_gate"] = (state["rec_gate"] if cfg.disease_device.startswith("cuda")
                                  else Gate(1, cfg.queue_limit))
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
        # 비정상 확률(NaN/Inf)이나 업로드 원문을 오류 응답에 재삽입하지 않는다.
        details = [{k: v for k, v in error.items() if k not in {"input", "ctx"}} for error in e.errors()]
        return _err("E4002", 422, "요청 형식이 올바르지 않습니다.", json.loads(json.dumps(details, default=str)))

    def auth(x_internal_token: str | None = Header(default=None)):
        # 토큰 미설정 시에도 닫힌 상태로 동작 (fail-closed)
        if not cfg.internal_token or not x_internal_token or not hmac.compare_digest(x_internal_token, cfg.internal_token):
            raise AppError("E1001", 401, "내부 인증 토큰이 없거나 올바르지 않습니다.")

    async def _read_image(image: UploadFile) -> bytes:
        limit = int(cfg.max_image_mb * 1024 * 1024)
        data = await image.read(limit + 1)
        if len(data) > limit:
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
        bundle = retrieve_evidence(state['ingredient_evidence'], x, avoid=req.survey.avoidIngredients) if state['ingredient_evidence'] else None
        return await _generate_recommendation(x, notes, evidence_bundle=bundle)

    async def _generate_recommendation(x, notes: list[str], disease: DiseasePrediction | None = None, evidence_bundle=None) -> RecommendResponse:
        if evidence_bundle is not None and not evidence_bundle['claims']:
            return RecommendResponse(
                skin_type=x.skin_type, concerns=x.concerns, ingredients=[], routine=[], reasoning=[],
                answer='입력·회피 조건에 맞는 성분 근거가 없어 추천을 보류했습니다.',
                disclaimer='본 결과는 참고용 정보이며 의학적 진단이나 치료를 대신하지 않습니다.',
                filtered=False, used_fallback=False, input_notes=list(notes),
                model_version='not_generated', inference_ms=0, generation_origin='not_generated',
                quality_status='withheld', quality_issues=['evidence_no_eligible_claims'],
                evidence_status='withheld', evidence_version=evidence_bundle['version'])
        if evidence_bundle is not None and hasattr(state['rec'], 'recommend_with_context'):
            run = lambda value: generate_grounded(state['rec'],value,notes,state['extractor'],state['filter'],disease,
                evidence_bundle,disable_adapter=not cfg.rec_evidence_use_adapter)
            res,response,findings,review,_ = await state['rec_gate'].run(run,x,timeout=cfg.rec_timeout_s)
        else:
            res = await state['rec_gate'].run(state['rec'].recommend,x,timeout=cfg.rec_timeout_s)
            response,findings=prepare_recommendation(res,x,notes,state['extractor'],state['filter'],disease)
            review=evidence_review(res.parsed,response,evidence_bundle) if evidence_bundle else None
        if evidence_bundle is not None:
            if cfg.rec_evidence_mode == 'enforce':
                response = enforce_evidence(response, review)
            else:
                response.evidence_version = review['version']
                response.ingredient_evidence = review['bindings']
        if findings:
            # Do not log user notes or generated health text in normal server logs.
            log.info("추천 품질 검사: %s", response.quality_issues)
        return response

    async def _disease(data: bytes) -> DiseasePrediction:
        if "disease" not in state:
            raise AppError("E3001", 503, "질환 분류 모델이 비활성화돼 있습니다. DISEASE_BACKEND 설정을 확인하세요.")
        t0 = time.perf_counter()
        try:
            result = await state["disease_gate"].run(state["disease"].predict, data, timeout=30)
        except InvalidDiseaseImage as exc:
            raise AppError("E2002", 422, "이미지를 해석할 수 없습니다. 다시 촬영해 주세요.", str(exc)) from exc
        pred = DiseasePrediction.model_validate(result)
        return pred.model_copy(update={"inference_ms": int((time.perf_counter() - t0) * 1000)})

    async def _disease_recommend(req: DiseaseRecommendRequest) -> DiseaseAnalyzeResponse:
        pred = req.disease.model_copy(update={
            "uncertain": req.disease.uncertain or req.disease.confidence < cfg.disease_threshold})
        try:
            x, notes, status = build_disease_rec_input(
                req.survey, req.metrics, pred, state["mapping"], state["disease_mapping"],
                cfg.disease_threshold)
        except ValueError as exc:
            raise AppError("E4002", 422, "분류/설문/측정값을 해석할 수 없습니다.", str(exc)) from exc
        reference = medical_reference(pred,state['medical_knowledge'],cfg.disease_threshold)
        if x is not None and reference is not None:
            x.user_note = (x.user_note or '') + reference_prompt(reference)
        bundle = retrieve_evidence(state['ingredient_evidence'], x, pred, req.survey.avoidIngredients, cfg.disease_threshold) if x is not None and state['ingredient_evidence'] else None
        if bundle is not None and not bundle['claims']:
            return DiseaseAnalyzeResponse(disease=pred, recommendation=None, recommendation_status='quality_rejected',
                guidance=disease_guidance(pred) + ['회피 조건에 맞는 성분 근거가 없어 추천을 보류했습니다.'],
                quality_issues=['evidence_no_eligible_claims'], generation_origin='not_generated', medical_reference=reference)
        rec = await _generate_recommendation(x, notes, pred, bundle) if x is not None else None
        guidance = disease_guidance(pred) + (notes if rec is None else [])
        quality_issues = rec.quality_issues if rec is not None else []
        generation_origin = rec.generation_origin if rec is not None else 'not_generated'
        if rec is not None and rec.quality_status == "withheld":
            status, rec = "quality_rejected", None
            guidance.append("생성 결과가 추천 품질 기준을 충족하지 못해 성분 추천을 보류했습니다.")
        return DiseaseAnalyzeResponse(disease=pred, recommendation=rec,
                                       recommendation_status=status, guidance=guidance,
                                       quality_issues=quality_issues,generation_origin=generation_origin,medical_reference=reference)

    def _parse_survey(survey: str) -> Survey:
        try:
            return Survey.model_validate_json(survey)
        except ValidationError as exc:
            details = [{k: v for k, v in error.items() if k not in {'input', 'ctx'}} for error in exc.errors()]
            raise AppError("E4002", 422, "survey JSON 형식이 올바르지 않습니다.",
                           json.loads(json.dumps(details, default=str))) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health():
        rec_ok, skin_ok, disease_ok = "rec" in state, "skin" in state, "disease" in state
        ok = rec_ok and skin_ok and (cfg.disease_backend == "disabled" or disease_ok)
        return HealthResponse(status="ok" if ok else "degraded", rec_loaded=rec_ok,
                              skin_loaded=skin_ok, rec_backend=cfg.rec_backend, skin_backend=cfg.skin_backend,
                              disease_loaded=disease_ok, disease_backend=cfg.disease_backend)

    @app.get("/model-info", response_model=ModelInfo, dependencies=[Depends(auth)])
    async def model_info():
        return ModelInfo(rec_model_version=state["rec"].version, skin_model_version=state["skin"].version,
                         rec_base_model=cfg.rec_base_model if cfg.rec_backend == "sllm" else None,
                         rec_adapter_dir=cfg.rec_adapter_dir if cfg.rec_backend == "sllm" else None,
                         disease_model_version=state["disease"].version if "disease" in state else None)

    @app.post("/infer/skin-metrics", response_model=SkinMetricsResponse, dependencies=[Depends(auth)])
    async def skin_metrics(image: UploadFile = File(...), device_type: str = Form("smartphone")):
        return await _skin(await _read_image(image), device_type)

    @app.post("/infer/recommend", response_model=RecommendResponse, dependencies=[Depends(auth)])
    async def recommend(req: RecommendRequest):
        return await _recommend(req)

    @app.post("/infer/disease", response_model=DiseasePrediction, dependencies=[Depends(auth)])
    async def disease(image: UploadFile = File(...)):
        return await _disease(await _read_image(image))

    @app.post("/infer/disease-recommend", response_model=DiseaseAnalyzeResponse, dependencies=[Depends(auth)])
    async def disease_recommend(req: DiseaseRecommendRequest):
        """별도 분류 서버의 원본 JSON을 받는 2단계 연동 경로."""
        return await _disease_recommend(req)

    @app.post("/infer/disease-analyze", response_model=DiseaseAnalyzeResponse, dependencies=[Depends(auth)])
    async def disease_analyze(image: UploadFile = File(...), survey: str = Form(...)):
        sv = _parse_survey(survey)
        pred = await _disease(await _read_image(image))
        return await _disease_recommend(DiseaseRecommendRequest(disease=pred, survey=sv))

    @app.post("/infer/analyze", response_model=AnalyzeResponse, dependencies=[Depends(auth)])
    async def analyze(image: UploadFile = File(...), survey: str = Form(...),
                      device_type: str = Form("smartphone")):
        sv = _parse_survey(survey)
        data = await _read_image(image)
        # 실제 이미지 디코딩/분류를 먼저 수행해 손상 이미지를 Qwen에 넘기지 않는다.
        pred = await _disease(data) if "disease" in state else None
        skin = await _skin(data, device_type)
        metrics = None if skin.is_stub else skin.metrics
        if pred is not None:
            result = await _disease_recommend(DiseaseRecommendRequest(disease=pred, survey=sv, metrics=metrics))
            return AnalyzeResponse(skin=skin, **result.model_dump())
        rec = await _recommend(RecommendRequest(metrics=metrics, survey=sv))
        if rec.quality_status == "withheld":
            return AnalyzeResponse(skin=skin, recommendation=None, recommendation_status="quality_rejected",
                                   quality_issues=rec.quality_issues,
                                   guidance=["생성 결과가 추천 품질 기준을 충족하지 못해 성분 추천을 보류했습니다."])
        return AnalyzeResponse(skin=skin, recommendation=rec, quality_issues=rec.quality_issues)

    return app


app = create_app()
