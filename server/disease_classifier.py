"""배포된 DenseNet121 분류기의 서버 어댑터. Qwen의 VRAM을 위해 기본은 CPU.

원본: AI-SkinCare/skincare-ai, feat/3-skin-disease-classifier,
3ea9506bf72f096b3e2bd545f241a8c57fcbfe46/model/src/{config,model,dataset,inference}.py.
전처리·분류기 구조·클래스 순서를 그대로 사용한다. 원본의 sys.path 변경과
전역 config/model import는 추천 서버와 충돌하므로 사용하지 않는다.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

from .schemas import DiseasePrediction

CLASS_NAMES = ("건선", "아토피", "여드름", "정상", "주사", "지루")
CLASS_NAMES_EN = ("psoriasis", "atopic", "acne", "normal", "rosacea", "seborrheic")
SOURCE_REVISION = "3ea9506bf72f096b3e2bd545f241a8c57fcbfe46"


class InvalidDiseaseImage(ValueError):
    pass


class DenseNetDiseaseClassifier:
    version = "densenet121_s2_" + SOURCE_REVISION[:7]

    def __init__(self, weights: Path, device: str = "cpu", threshold: float = 0.6):
        if not 0 <= threshold <= 1:
            raise ValueError("DISEASE_THRESHOLD는 0~1이어야 합니다")
        self.weights, self.device, self.threshold = Path(weights), device, threshold
        self.model = self.transform = None

    def load(self) -> "DenseNetDiseaseClassifier":
        import torch
        from torchvision import models, transforms

        if not self.weights.is_file():
            raise FileNotFoundError(f"질환 분류 가중치가 없습니다: {self.weights}")
        # 배포 파일의 config에 torch.__version__(TorchVersion str 하위 클래스)이 있다.
        # 이 알려진 메타데이터 형식만 허용하고 일반 pickle 실행은 허용하지 않는다.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
        cfg = checkpoint.get("config", {})
        if "class_names" in cfg and tuple(cfg["class_names"]) != CLASS_NAMES:
            raise ValueError("가중치의 클래스 순서가 배포 분류기와 다릅니다")
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.get("dropout", 0.5)), torch.nn.Linear(1024, len(CLASS_NAMES)))
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model = model.to(self.device).eval()
        size = int(cfg.get("image_size", 256))
        self.transform = transforms.Compose([
            transforms.Resize((size, size), antialias=True), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return self

    def predict(self, image_bytes: bytes) -> DiseasePrediction:
        import torch
        from PIL import Image, UnidentifiedImageError

        if self.model is None or self.transform is None:
            raise RuntimeError("load()를 먼저 호출하세요")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.format not in {"JPEG", "PNG"}:
                        raise InvalidDiseaseImage("JPEG 또는 PNG 이미지만 지원합니다")
                    # 원본 eval transform과 동일: EXIF 회전 등 추가 전처리 없음.
                    x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise InvalidDiseaseImage("이미지를 읽을 수 없습니다. 정상 JPEG/PNG로 다시 보내세요.") from exc
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(x), 1)[0].cpu().tolist()
        k = max(range(len(probabilities)), key=probabilities.__getitem__)
        return DiseasePrediction(
            label=CLASS_NAMES[k], label_en=CLASS_NAMES_EN[k], confidence=probabilities[k],
            uncertain=probabilities[k] < self.threshold,
            probabilities=dict(zip(CLASS_NAMES, probabilities)), model_version=self.version)
