import io
import json

import pytest
from fastapi.testclient import TestClient

from server.adapters import build_rec_input, load_mapping
from server.config import Settings
from server.main import create_app
from server.schemas import Survey

TOKEN = "test-token"
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 2048
SURVEY = {"gender": "female", "ageBand": "20s", "sleepHours": 5, "sunExposureLevel": "high",
          "selfReportedConcerns": ["건조함", "모공"], "skinType": "combination"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(Settings(internal_token=TOKEN, rec_backend="mock", skin_backend="stub",
                              processed_dir=tmp_path_factory.mktemp("no-processed-data")))
    with TestClient(app) as c:
        yield c


H = {"X-Internal-Token": TOKEN}


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("hdr", [{}, {"X-Internal-Token": "wrong"}])
def test_auth_required(client, hdr):
    r = client.post("/infer/recommend", json={"survey": SURVEY}, headers=hdr)
    assert r.status_code == 401 and r.json()["error"]["code"] == "E1001"


def test_recommend_mock(client):
    r = client.post("/infer/recommend", headers=H, json={
        "metrics": {"full_face": {"moisture": 0.2, "elasticity": 0.7, "pigmentation": 0.4, "pore": 0.7}},
        "survey": SURVEY})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["skin_type"] == "복합성"
    assert b["concerns"] == ["과각질/악건성", "모공"]   # 자가응답 순서 유지, 최대 2개
    assert len(b["ingredients"]) >= 1 and b["ingredients"][0]["priority"] == 1
    assert b["routine"] and b["disclaimer"] and len(b["reasoning"]) == 3
    assert any("연령대" in n for n in b["input_notes"])


def test_recommend_bad_survey(client):
    r = client.post("/infer/recommend", headers=H, json={"survey": {"gender": "x"}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "E4002"


def test_skin_metrics_stub_and_type_check(client):
    ok = client.post("/infer/skin-metrics", headers=H, files={"image": ("a.jpg", io.BytesIO(JPEG), "image/jpeg")})
    assert ok.status_code == 200 and ok.json()["is_stub"] is True
    bad = client.post("/infer/skin-metrics", headers=H, files={"image": ("a.gif", io.BytesIO(b"GIF89a..."), "image/gif")})
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "E2004"


def test_analyze_end_to_end(client):
    r = client.post("/infer/analyze", headers=H,
                    files={"image": ("a.jpg", io.BytesIO(JPEG), "image/jpeg")},
                    data={"survey": json.dumps(SURVEY), "device_type": "smartphone"})
    assert r.status_code == 200, r.text
    assert r.json()["skin"]["is_stub"] and r.json()["recommendation"]["concerns"]


def test_adapter_metric_concerns_without_survey_concerns():
    x, notes = build_rec_input(Survey(gender="male", age=48),
                               {"full_face": {"moisture": 0.5, "elasticity": 0.3, "pigmentation": 0.8, "pore": 0.5}},
                               load_mapping())
    assert x.gender == "남성" and x.age == 48
    assert x.concerns == ["미백(색소침착/기미/칙칙함)", "주름"]  # 심각도 순 (0.2 > 0.1)
    assert x.condition_codes is None
