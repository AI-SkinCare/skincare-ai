"""분류 결과를 실제 추천 입력까지 전달하는 계약/오류/회귀 테스트. GPU 불필요."""
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server import main
from server.adapters import load_mapping
from server.config import Settings
from server.disease_classifier import InvalidDiseaseImage
from server.disease_pipeline import build_disease_rec_input, filter_disease_output, load_disease_mapping
from server.schemas import DISEASE_LABELS, DiseaseAnalyzeResponse, DiseasePrediction, DiseaseRecommendRequest, Survey
from src.llm.inference import MockRecommender

TOKEN = "test-disease-token"
HEADERS = {"X-Internal-Token": TOKEN}
SURVEY = {"gender": "female", "age": 30, "skinType": "dry", "note": "수분 관리가 궁금해요"}
IMAGE = b"\x89PNG\r\n\x1a\n" + b"test image bytes"


def prediction(label_en="atopic", confidence=0.9, uncertain=False):
    label = DISEASE_LABELS[label_en]
    return DiseasePrediction(label=label, label_en=label_en, confidence=confidence, uncertain=uncertain,
                             probabilities={v: confidence if v == label else (1-confidence)/5
                                            for v in DISEASE_LABELS.values()})


@pytest.mark.parametrize("label_en", [k for k in DISEASE_LABELS if k != "normal"])
def test_each_label_reaches_qwen_input_with_prediction_context(label_en):
    x, notes, status = build_disease_rec_input(Survey(**SURVEY), None, prediction(label_en),
                                              load_mapping(), load_disease_mapping(), 0.6)
    assert status == "generated"
    assert x.concerns[0] == load_disease_mapping()[label_en]["concerns"][0]
    assert DISEASE_LABELS[label_en] in x.user_note and "진단 아님" in x.user_note
    assert "수분 관리가 궁금해요" in x.user_note
    assert notes and x.condition_codes is None


@pytest.mark.parametrize("label_en", ["psoriasis", "atopic", "seborrheic"])
def test_sensitive_classification_has_priority_over_unrelated_survey_concerns(label_en):
    survey = Survey(**SURVEY, selfReportedConcerns=["모공", "미백"])
    x, _, _ = build_disease_rec_input(survey, None, prediction(label_en),
                                      load_mapping(), load_disease_mapping(), 0.6)
    assert x.concerns[0] == "민감성(트러블/자극감)"


@pytest.mark.parametrize("conf,uncertain", [(0.55, False), (0.9, True)])
def test_uncertainty_cannot_be_bypassed_by_an_external_flag(conf, uncertain):
    x, _, status = build_disease_rec_input(Survey(**SURVEY), None, prediction(confidence=conf, uncertain=uncertain),
                                          load_mapping(), load_disease_mapping(), 0.6)
    assert x is None and status == "uncertain"


def test_normal_does_not_invent_a_pore_concern():
    x, _, status = build_disease_rec_input(Survey(**SURVEY), None, prediction("normal"),
                                          load_mapping(), load_disease_mapping(), 0.6)
    assert x is None and status == "no_concern"


def test_normal_uses_explicit_survey_concern():
    sv = Survey(**SURVEY, selfReportedConcerns=["여드름"])
    x, _, status = build_disease_rec_input(sv, None, prediction("normal"),
                                          load_mapping(), load_disease_mapping(), 0.6)
    assert status == "generated" and x.concerns == ["여드름/뾰루지"]


def test_normal_does_not_invent_a_concern_from_unknown_free_text():
    sv = Survey(**SURVEY, selfReportedConcerns=["미지원 고민"])
    x, _, status = build_disease_rec_input(sv, None, prediction("normal"),
                                          load_mapping(), load_disease_mapping(), 0.6)
    assert x is None and status == "no_concern"


def test_disease_output_does_not_guarantee_safety_for_sensitive_skin():
    # 실제 Qwen2.5 아토피 문맥 시험 출력에서 관찰한 문장.
    claim = "이들 성분은 민감성 피부에도 안전하게 사용될 수 있어 피부 자극을 최소화합니다."
    parsed = {"reasoning": [{"step": 1, "title": "성분 선택", "content": claim + " 사용 전 반응을 확인하세요."}],
              "answer": claim + " 충분한 보습을 권합니다.", "disclaimer": "전문의와 상담하세요."}
    cleaned, changed = filter_disease_output(parsed)
    assert changed and "안전하게" not in str(cleaned)
    assert cleaned["reasoning"][0]["content"] == "사용 전 반응을 확인하세요."
    assert cleaned["answer"] == "충분한 보습을 권합니다."
    assert cleaned["disclaimer"] == parsed["disclaimer"]


@pytest.mark.parametrize("mutation", [
    {"label_en": "acne"}, {"confidence": 1.1}, {"confidence": float("nan")},
    {"probabilities": {"아토피": 0.9}}, {"label": "unknown"},
    {"probabilities": {v: 0.5 for v in DISEASE_LABELS.values()}},
])
def test_invalid_classifier_json_is_rejected(mutation):
    data = prediction().model_dump()
    data.update(mutation)
    with pytest.raises(ValidationError):
        DiseasePrediction.model_validate(data)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    class FakeClassifier:
        version = "test-classifier"
        result = prediction()
        error = None
        calls = []

        def __init__(self, *args):
            pass

        def load(self):
            return self

        def predict(self, data):
            self.calls.append(data)
            if self.error:
                raise self.error
            return self.result

    class RecordingRecommender(MockRecommender):
        calls = []

        def recommend(self, x):
            self.calls.append(x)
            return super().recommend(x)

    monkeypatch.setattr(main, "DenseNetDiseaseClassifier", FakeClassifier)
    monkeypatch.setattr(main, "MockRecommender", RecordingRecommender)
    cfg = Settings(internal_token=TOKEN, rec_backend="mock", skin_backend="stub", disease_backend="densenet",
                   processed_dir=tmp_path / "no-processed-data")
    with TestClient(main.create_app(cfg)) as client:
        yield client, FakeClassifier, RecordingRecommender


def analyze(client, path="/infer/disease-analyze", headers=HEADERS):
    return client.post(path, headers=headers, data={"survey": json.dumps(SURVEY)},
                       files={"image": ("test.png", io.BytesIO(IMAGE), "image/png")})


def test_image_to_classifier_to_recommender_end_to_end(pipeline):
    client, classifier, rec = pipeline
    response = analyze(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["disease"]["label_en"] == "atopic"
    assert body["recommendation_status"] == "generated" and body["recommendation"]["ingredients"]
    assert classifier.calls == [IMAGE] and len(rec.calls) == 1
    assert rec.calls[0].concerns == ["민감성(트러블/자극감)"]
    assert "아토피" in rec.calls[0].user_note and body["guidance"]
    health = client.get("/health").json()
    assert health["disease_loaded"] and health["disease_backend"] == "densenet"


def test_existing_analyze_also_connects_classifier_and_ignores_stub_metrics(pipeline):
    client, _, rec = pipeline
    response = analyze(client, "/infer/analyze")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["skin"]["is_stub"] and body["disease"]["label_en"] == "atopic"
    assert rec.calls[0].concerns == ["민감성(트러블/자극감)"]


def test_external_classification_json_uses_the_same_pipeline(pipeline):
    client, classifier, rec = pipeline
    response = client.post("/infer/disease-recommend", headers=HEADERS,
                           json={"survey": SURVEY, "disease": prediction("acne").model_dump()})
    assert response.status_code == 200, response.text
    assert not classifier.calls and rec.calls[0].concerns[0] == "여드름/뾰루지"


def test_low_confidence_never_invokes_qwen(pipeline):
    client, classifier, rec = pipeline
    classifier.result = prediction(confidence=0.55, uncertain=False)
    body = analyze(client).json()
    assert body["recommendation_status"] == "uncertain" and body["recommendation"] is None
    assert body["disease"]["uncertain"] is True
    assert not rec.calls


def test_normal_without_concerns_never_invokes_qwen(pipeline):
    client, classifier, rec = pipeline
    classifier.result = prediction("normal")
    body = analyze(client).json()
    assert body["recommendation_status"] == "no_concern" and body["recommendation"] is None
    assert not rec.calls


def test_bad_image_has_no_recommendation(pipeline):
    client, classifier, rec = pipeline
    classifier.error = InvalidDiseaseImage("broken")
    response = analyze(client)
    assert response.status_code == 422 and response.json()["error"]["code"] == "E2002"
    assert not rec.calls


def test_auth_still_required_before_model_execution(pipeline):
    client, classifier, rec = pipeline
    assert analyze(client, headers={}).status_code == 401
    assert not classifier.calls and not rec.calls


def test_classifier_endpoint_and_invalid_external_output(pipeline):
    client, _, rec = pipeline
    response = client.post("/infer/disease", headers=HEADERS,
                           files={"image": ("x.png", IMAGE, "image/png")})
    assert response.status_code == 200 and response.json()["label_en"] == "atopic"
    data = prediction().model_dump()
    data["label_en"] = "acne"
    response = client.post("/infer/disease-recommend", headers=HEADERS, json={"survey": SURVEY, "disease": data})
    assert response.status_code == 422 and not rec.calls


def test_disabled_classifier_is_not_faked():
    with TestClient(main.create_app(Settings(internal_token=TOKEN, rec_backend="mock", skin_backend="stub",
                                               disease_backend="disabled"))) as client:
        assert analyze(client).status_code == 503


@pytest.mark.parametrize("status", ["generated", "uncertain", "no_concern"])
def test_published_contract_fixtures_work_without_models_or_processed_data(tmp_path, status):
    fixtures = Path(__file__).resolve().parents[1] / "docs/fixtures/disease-pipeline"
    payload = json.loads((fixtures / f"{status}.request.json").read_text(encoding="utf-8"))
    DiseaseRecommendRequest.model_validate(payload)
    recorded = DiseaseAnalyzeResponse.model_validate_json((fixtures / f"{status}.response.json").read_text(encoding="utf-8"))
    assert recorded.recommendation_status == status
    cfg = Settings(internal_token=TOKEN, rec_backend="mock", disease_backend="disabled",
                   processed_dir=tmp_path / "no-data")
    with TestClient(main.create_app(cfg)) as client:
        response = client.post("/infer/disease-recommend", headers=HEADERS, json=payload)
    assert response.status_code == 200, response.text
    body = DiseaseAnalyzeResponse.model_validate(response.json())
    assert body.recommendation_status == status
    if status == "generated":
        assert body.recommendation and body.recommendation.ingredients
    else:
        assert body.recommendation is None


@pytest.mark.parametrize("bad_number", ["NaN", "Infinity", "-Infinity"])
def test_invalid_probability_over_http_is_422_not_serialization_error(pipeline, bad_number):
    client, _, rec = pipeline
    body = json.dumps({"survey": SURVEY, "disease": prediction().model_dump()})
    body = body.replace('"confidence": 0.9', '"confidence": ' + bad_number)
    response = client.post("/infer/disease-recommend", headers={**HEADERS, "Content-Type": "application/json"},
                           content=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "E4002"
    assert not rec.calls


def test_stub_metrics_are_not_passed_to_legacy_recommend(monkeypatch):
    calls = []
    original = main.build_rec_input

    def capture(survey, metrics, mapping):
        calls.append(metrics)
        return original(survey, metrics, mapping)

    monkeypatch.setattr(main, "build_rec_input", capture)
    with TestClient(main.create_app(Settings(internal_token=TOKEN, rec_backend="mock", skin_backend="stub",
                                               disease_backend="disabled"))) as client:
        assert analyze(client, "/infer/analyze").status_code == 200
    assert calls == [None]
