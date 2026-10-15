"""단일 이미지 추론: 이미지 1장 → 진단 결과 dict(JSON 직렬화 가능).

전처리는 학습/평가와 동일 (dataset.build_transforms(train=False): Resize 256² + ImageNet Normalize).

출력 예:
  {"label": "아토피", "label_en": "atopic", "confidence": 0.9412, "uncertain": false,
   "probabilities": {"건선": 0.0213, "아토피": 0.9412, ...}}
  uncertain: confidence < threshold (6클래스 밖의 사진이어도 반드시 6개 중 하나로 답하므로 참고용)

사용 예 (파이썬):
  from inference import SkinClassifier
  clf = SkinClassifier()                                     # 서버 시작 시 1회 로드 (기본: weights/densenet121_s2.pt)
  result = clf.predict("face.png")                          # 경로 / bytes / PIL.Image 모두 가능

사용 예 (CLI):
  python src/inference.py --image face.png
"""
import argparse
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 다른 디렉토리에서 import 해도 config 등을 찾도록
import config as C
from dataset import build_transforms
from model import build_model

DEFAULT_CKPT = C.PROJECT_ROOT / "weights" / "densenet121_s2.pt"  # baseline_s2/best.pt 의 가중치만 (test ACC 0.9491)
DEFAULT_THRESHOLD = 0.6


class SkinClassifier:
    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None, threshold=DEFAULT_THRESHOLD):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(dropout=cfg.get("dropout", C.DROPOUT), pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.transform = build_transforms(train=False, image_size=cfg.get("image_size", C.IMAGE_SIZE))
        self.threshold = threshold

    @staticmethod
    def _load_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            image = io.BytesIO(image)
        with Image.open(image) as im:  # 경로 또는 file-like
            return im.convert("RGB")

    @torch.no_grad()
    def predict(self, image):
        x = self.transform(self._load_image(image)).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(x), 1)[0].cpu()
        k = int(prob.argmax())
        conf = float(prob[k])
        return {
            "label": C.CLASS_NAMES[k],
            "label_en": C.CLASS_NAMES_EN[k],
            "confidence": round(conf, 4),
            "uncertain": conf < self.threshold,
            "probabilities": {name: round(float(p), 4) for name, p in zip(C.CLASS_NAMES, prob)},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    clf = SkinClassifier(args.ckpt, device="cpu" if args.cpu else None, threshold=args.threshold)
    print(json.dumps(clf.predict(args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
