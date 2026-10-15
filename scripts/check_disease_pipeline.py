"""실제 가중치를 쓰는 로컬 통합 점검. 사진은 저장/전송하지 않는다.

python scripts/check_disease_pipeline.py --synthetic --rec-backend sllm
python scripts/check_disease_pipeline.py --image face.png --survey survey.json --rec-backend sllm
--synthetic은 단색 생성 이미지로 배선만 확인하며 질환 정확도 시험이 아니다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    image_arg = parser.add_mutually_exclusive_group(required=True)
    image_arg.add_argument("--image", type=Path)
    image_arg.add_argument("--synthetic", action="store_true")
    image_arg.add_argument("--classification", type=Path, help="분류 결과 JSON으로 2단계 API 점검")
    parser.add_argument("--survey", type=Path)
    parser.add_argument("--rec-backend", choices=["mock", "sllm"], default="mock")
    parser.add_argument("--adapter", default=str(ROOT / "outputs/rec_sllm_v2/adapter"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/disease_pipeline_smoke.json")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from fastapi.testclient import TestClient
    from server.config import Settings
    from server.main import create_app

    if args.synthetic:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), (160, 120, 100)).save(buf, format="PNG")
        image = buf.getvalue()
    elif args.image:
        image = args.image.read_bytes()
    survey = (json.loads(args.survey.read_text(encoding="utf-8")) if args.survey else
              {"gender": "female", "age": 30, "skinType": "dry", "selfReportedConcerns": ["건조함"]})
    # TestClient는 프로세스 내에서만 요청하며 외부 서버/포트를 열지 않는다.
    token = "local-in-process-probe"
    cfg = Settings(internal_token=token, disease_backend="densenet", disease_device="cpu",
                   rec_backend=args.rec_backend, rec_base_model="Qwen/Qwen2.5-7B-Instruct",
                   rec_adapter_dir=args.adapter, skin_backend="stub")
    with TestClient(create_app(cfg)) as client:
        health = client.get("/health").json()
        if args.classification:
            prediction = json.loads(args.classification.read_text(encoding="utf-8"))
            response = client.post("/infer/disease-recommend", headers={"X-Internal-Token": token},
                                   json={"survey": survey, "disease": prediction})
        else:
            response = client.post("/infer/disease-analyze", headers={"X-Internal-Token": token},
                                   files={"image": ("probe.png", image, "image/png")},
                                   data={"survey": json.dumps(survey, ensure_ascii=False)})
        result = {"synthetic_input": args.synthetic, "classification_input": bool(args.classification), "clinical_validation": False,
                  "health": health, "http_status": response.status_code, "result": response.json()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if response.status_code != 200:
        raise SystemExit(1)
    rec = result["result"].get("recommendation")
    if args.rec_backend == "sllm" and rec and rec["used_fallback"]:
        raise SystemExit("Qwen 직접 출력이 아닌 규칙 폴백이 반환됐습니다.")


if __name__ == "__main__":
    main()
