from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server import main
from server.config import Settings


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
        ("C:\\private-location\\checkpoint", None),
        ("/private-location/checkpoint", None),
        ("models/local-checkpoint", None),
    ],
)
def test_model_info_returns_versions_without_local_paths(tmp_path, monkeypatch, base, expected):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models/local-checkpoint").mkdir(parents=True)
    (tmp_path / "ingredient_dict.json").write_text('{"lexicon": {}}', encoding="utf-8")
    adapter = r"D:\private-location\adapter"
    loaded_paths = []

    class FakeRecommender:
        version = "rec_sllm_v2"

        def __init__(self, base_model, adapter_dir, **_):
            loaded_paths.append((base_model, adapter_dir))

        def load(self):
            return self

    monkeypatch.setattr(main, "SLLMRecommender", FakeRecommender)
    cfg = Settings(
        internal_token="fixture-token", rec_backend="sllm", rec_base_model=base,
        rec_adapter_dir=adapter, processed_dir=tmp_path, rec_evidence_mode="off",
        disease_backend="disabled",
    )
    app = main.create_app(cfg, disease_classifier_factory=lambda: SimpleNamespace(version="densenet"))
    with TestClient(app) as client:
        assert client.get("/model-info").status_code == 401
        result = client.get("/model-info", headers={"X-Internal-Token": cfg.internal_token})
        assert result.status_code == 200
        assert result.json() == {
            "rec_model_version": "rec_sllm_v2",
            "skin_model_version": "skin_reg_stub",
            "rec_base_model": expected,
            "disease_model_version": "densenet",
        }
    assert loaded_paths == [(base, adapter)]
