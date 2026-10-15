"""이미지 분류 어댑터의 오류 처리 및 실제 배포 가중치 선택 테스트."""
import io
import os
from pathlib import Path

import pytest

from server.disease_classifier import CLASS_NAMES, DenseNetDiseaseClassifier, InvalidDiseaseImage

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        DenseNetDiseaseClassifier(Path("missing.pt"), threshold=threshold)


def test_predict_requires_loaded_model():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    with pytest.raises(RuntimeError, match="load"):
        DenseNetDiseaseClassifier(Path("missing.pt")).predict(b"image")


@pytest.mark.parametrize("data", [b"not an image", b"\xff\xd8\xff" + b"bad jpeg"])
def test_corrupt_image_is_not_classified(data):
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    classifier = DenseNetDiseaseClassifier(Path("unused.pt"))
    classifier.model = object()
    classifier.transform = object()
    with pytest.raises(InvalidDiseaseImage):
        classifier.predict(data)


@pytest.mark.skipif(os.getenv("RUN_DISEASE_MODEL_TESTS") != "1", reason="실가중치 테스트는 명시적으로 실행")
def test_real_weights_and_preprocessing_match_upstream():
    import torch
    from PIL import Image
    from torchvision import models, transforms

    # 같은 구조와 eval transform을 독립적으로 구성해 결과를 비교한다.
    weights = ROOT / "model/weights/densenet121_s2.pt"
    classifier = DenseNetDiseaseClassifier(weights).load()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("config", {})
    reference = models.densenet121(weights=None)
    reference.classifier = torch.nn.Sequential(torch.nn.Dropout(cfg.get("dropout", 0.5)), torch.nn.Linear(1024, 6))
    reference.load_state_dict(checkpoint["model"])
    reference.eval()
    size = cfg.get("image_size", 256)
    transform = transforms.Compose([
        transforms.Resize((size, size), antialias=True), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.new("RGB", (300, 200), (160, 120, 100))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    actual = classifier.predict(buf.getvalue())
    with torch.inference_mode():
        expected = torch.softmax(reference(transform(image).unsqueeze(0)), dim=1)[0].tolist()
    assert actual.probabilities == pytest.approx(dict(zip(CLASS_NAMES, expected)))
