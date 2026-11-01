"""Prevent evaluation leakage and incorrect accuracy denominators."""
import csv
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_disease_images import COMMON_SURVEY, classification_metrics, load_samples, request_recommendation


def test_raw_accuracy_includes_rejected_and_failed_predictions():
    records = [
        {"http_status": 200, "true_label": "건선", "view": "정면", "correct": True,
         "response": {"label": "건선", "uncertain": False}},
        {"http_status": 200, "true_label": "아토피", "view": "정면", "correct": True,
         "response": {"label": "아토피", "uncertain": True}},
        {"http_status": 200, "true_label": "지루", "view": "정면", "correct": False,
         "response": {"label": "건선", "uncertain": False}},
        {"http_status": 500, "true_label": "주사", "view": "정면", "correct": False, "response": {}},
    ]
    metrics = classification_metrics(records, total=4)
    assert metrics["accuracy"] == 0.5
    assert metrics["coverage"] == 0.5
    assert metrics["accuracy_among_accepted"] == 0.5
    assert metrics["uncertain"] == 1 and metrics["http_errors"] == 1
    assert metrics["per_class"][4]["recall"] == 0


def test_recommendation_payload_contains_only_model_prediction_and_test_survey():
    prediction = {"label": "여드름", "label_en": "acne", "confidence": 0.9, "uncertain": False,
                  "probabilities": {"건선": .02, "아토피": .02, "여드름": .9, "정상": .02, "주사": .02, "지루": .02}}
    payload = request_recommendation(prediction, COMMON_SURVEY)
    assert set(payload) == {"disease", "survey", "metrics"}
    assert payload["disease"]["label"] == "여드름" and payload["metrics"] is None
    assert payload["survey"]["selfReportedConcerns"] == []
    assert "true_label" not in str(payload) and "rel_path" not in str(payload)


@pytest.mark.parametrize("path", ["../outside.png", "missing.png"])
def test_csv_paths_cannot_escape_dataset_or_refer_to_missing_images(tmp_path, path):
    with (tmp_path/"labels.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_path","identifier","class_name","view","label"])
        writer.writerow([path,"test","건선","정면",0])
    with pytest.raises(ValueError, match="Missing or unsafe"):
        load_samples(tmp_path)


def test_csv_labels_must_match_fixed_class_order(tmp_path):
    folder = tmp_path/"건선"/"정면"
    folder.mkdir(parents=True)
    Image.new("RGB",(10,10)).save(folder/"test.png")
    with (tmp_path/"labels.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_path","identifier","class_name","view","label"])
        writer.writerow(["건선/정면/test.png","test","건선","정면",1])
    with pytest.raises(ValueError, match="Invalid class"):
        load_samples(tmp_path)
