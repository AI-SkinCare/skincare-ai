"""실제 localhost 서버의 질환→Qwen 추천 품질/지연/입력 오류 점검.

분류 라벨 사례는 명시적인 시험 fixture이며 임상 정답 시험이 아니다.
단색/잡음 이미지는 비얼굴 음성 대조군이다. 모든 원문 응답을 로컬에 보존한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import sys
import threading
import time

import httpx
import numpy as np
from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.schemas import DISEASE_LABELS


def classification(key, confidence=0.9, uncertain=False):
    label = DISEASE_LABELS[key]
    return {"label": label, "label_en": key, "confidence": confidence, "uncertain": uncertain,
            "probabilities": {v: confidence if v == label else (1-confidence)/5 for v in DISEASE_LABELS.values()},
            "model_version": "quality_test_fixture_not_a_patient_prediction"}


def survey(**kwargs):
    return {"gender": "female", "age": 35, "skinType": "dry", "selfReportedConcerns": [], **kwargs}


def recommendation_cases():
    cases = [(k + "_basic", k, survey()) for k in DISEASE_LABELS if k != "normal"]
    cases += [
        ("normal_reported_dry", "normal", survey(selfReportedConcerns=["건조함"])),
        ("atopic_conflicting_scrub_request", "atopic",
         survey(selfReportedConcerns=["모공", "각질"], note="따갑고 붉지만 강한 스크럽으로 각질을 매일 제거하고 싶어요.")),
        ("rosacea_conflicting_peeling_request", "rosacea",
         survey(skinType="combination", sunExposureLevel="high", note="붉은 피부에 글리콜릭애씨드 필링을 매일 써도 되나요?")),
        ("psoriasis_older_oily", "psoriasis", survey(gender="male", age=65, skinType="oily", selfReportedConcerns=["주름"])),
        ("acne_dry_sensitive", "acne", survey(age=48, skinType="dry", note="건조하고 민감해서 세안 후 따갑습니다.")),
    ]
    return cases


def negative_images():
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    checker = ((np.indices((256, 256)).sum(axis=0) // 16) % 2 * 255).astype(np.uint8)
    arrays = {
        "black": np.zeros((256, 256, 3), dtype=np.uint8),
        "white": np.full((256, 256, 3), 255, dtype=np.uint8),
        "gray": np.full((256, 256, 3), 128, dtype=np.uint8),
        "skin_color_solid": np.full((256, 256, 3), [160, 120, 100], dtype=np.uint8),
        "seeded_noise": noise, "checkerboard": np.repeat(checker[:, :, None], 3, axis=2),
    }
    out = {}
    for key, arr in arrays.items():
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        out[key] = buf.getvalue()
    return out


def review_candidates(rec):
    if not rec:
        return {}
    text = " ".join(rec.get("reasoning", [])) + " " + rec.get("answer", "")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    patterns = {
        "exfoliation_or_peeling_mention": r"스크럽|각질\s*제거|필링|글리콜|글라이콜|락틱|젖산",
        "safety_or_strong_effect_claim": r"안전|부작용|강력|효과적|확실|완벽|반드시|필수적",
        "medical_language": r"치료|처방|투약|복용|스테로이드|항진균|완치",
        "anatomical_specificity_not_in_test_input": r"눈가|볼\s*부위|턱|입\s*주변|이마|코\s*주변",
    }
    return {key: list(dict.fromkeys(s for s in sentences if re.search(pattern, s)))
            for key, pattern in patterns.items()}


def inspect_recommendation(rec):
    if rec is None:
        return None
    return {
        "answer_nonempty": bool(rec.get("answer", "").strip()),
        "ingredient_count": len(rec.get("ingredients", [])),
        "routine_count": len(rec.get("routine", [])),
        "reasoning_nonempty": all(s.strip() for s in rec.get("reasoning", [])),
        "empty_purpose_count": sum(not x.get("purpose") for x in rec.get("ingredients", [])),
        "model_version": rec.get("model_version"), "used_fallback": rec.get("used_fallback"),
        "filtered": rec.get("filtered"), "generation_ms": rec.get("inference_ms"),
        "concerns": rec.get("concerns", []),
    }


class Monitor:
    def __init__(self, path):
        self.path, self.stop = path, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.peak_memory_mib = 0

    def run(self):
        with self.path.open("w", encoding="utf-8") as f:
            while not self.stop.is_set():
                try:
                    proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    vals = [int(v.strip()) for v in proc.stdout.strip().split(",")]
                    self.peak_memory_mib = max(self.peak_memory_mib, vals[0])
                    f.write(json.dumps({"elapsed": time.monotonic(), "memory_used_mib": vals[0],
                                        "memory_total_mib": vals[1], "utilization": vals[2]}) + "\n")
                    f.flush()
                except Exception as exc:
                    f.write(json.dumps({"error": type(exc).__name__}) + "\n")
                self.stop.wait(2)


def stop_process_tree(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/disease_quality_20261015")
    parser.add_argument("--adapter", default=str(ROOT / "outputs/rec_sllm_v2/adapter"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "responses.jsonl").exists():
        raise SystemExit("기존 결과를 덮어쓰지 않도록 다른 --output-dir을 사용하세요.")
    env = os.environ.copy()
    env.update(REC_BACKEND="sllm", REC_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct", REC_ADAPTER_DIR=args.adapter,
               DISEASE_BACKEND="densenet", DISEASE_DEVICE="cpu", DISEASE_THRESHOLD="0.6", SKIN_BACKEND="stub",
               HF_HOME="D:/hf_cache", HF_HUB_CACHE="D:/hf_cache/hub", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUTF8="1", AI_INTERNAL_TOKEN=secrets.token_hex(24))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    records = []
    monitor = Monitor(out / "gpu.jsonl")
    monitor.thread.start()
    with (out / "server.stdout.log").open("w", encoding="utf-8") as stdout, (out / "server.stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", str(port)],
                                cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            startup = time.perf_counter()
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"X-Internal-Token": env["AI_INTERNAL_TOKEN"]},
                              timeout=httpx.Timeout(180, connect=5), trust_env=False) as client:
                while time.perf_counter() - startup < 180:
                    if proc.poll() is not None:
                        raise RuntimeError("시험 서버가 시작 중 종료됨. server.stderr.log 확인")
                    try:
                        health = client.get("/health", timeout=3)
                        if health.status_code == 200 and health.json().get("disease_loaded"):
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError("서버 시작 180초 초과")
                startup_s = time.perf_counter() - startup
                print(f"[ready] startup={startup_s:.1f}s port={port}", flush=True)
                info = client.get("/model-info").json()

                def request(case_id, kind, path, *, expected_http=200, **kwargs):
                    t0 = time.perf_counter()
                    try:
                        response = client.post(path, **kwargs)
                        status, body = response.status_code, response.json()
                    except Exception as exc:
                        status, body = 0, {"request_error": type(exc).__name__, "detail": str(exc)}
                    rec = body.get("recommendation") if isinstance(body, dict) else None
                    record = {"id": case_id, "kind": kind, "endpoint": path, "http_status": status,
                              "expected_http_status": expected_http, "http_as_expected": status == expected_http,
                              "elapsed_s": round(time.perf_counter()-t0, 3), "response": body,
                              "structure": inspect_recommendation(rec), "review_candidates": review_candidates(rec)}
                    if "json" in kwargs:
                        record["request"] = kwargs["json"]
                    records.append(record)
                    with (out / "responses.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    st = record["structure"]
                    extra = f" ings={st['ingredient_count']} fallback={st['used_fallback']}" if st else ""
                    print(f"[{len(records):02d}] {case_id} http={status} {record['elapsed_s']:.1f}s{extra}", flush=True)
                    return record

                images = negative_images()
                for key, image in images.items():
                    request("nonface_"+key, "nonface_negative_control", "/infer/disease", files={"image": (key+".png", image, "image/png")})
                request("no_internal_token", "invalid_input", "/infer/disease", expected_http=401,
                        headers={"X-Internal-Token": ""}, files={"image": ("x.png", images["gray"], "image/png")})
                request("unsupported_gif", "invalid_input", "/infer/disease", expected_http=415,
                        files={"image": ("x.gif", b"GIF89a"+b"0"*20, "image/gif")})
                request("corrupt_jpeg", "invalid_input", "/infer/disease", expected_http=422,
                        files={"image": ("x.jpg", b"\xff\xd8\xff"+b"0"*200, "image/jpeg")})
                request("oversize_image", "invalid_input", "/infer/disease", expected_http=413,
                        files={"image": ("x.png", b"\x89PNG\r\n\x1a\n"+b"0"*(10*1024*1024), "image/png")})
                request("invalid_label_pair", "invalid_input", "/infer/disease-recommend", expected_http=422,
                        json={"disease": {**classification("atopic"), "label_en": "acne"}, "survey": survey()})
                for key, pred in [("normal_no_concern", classification("normal")),
                                  ("normal_threshold_boundary", classification("normal", 0.6)),
                                  ("low_score_false_flag", classification("atopic", 0.55, False)),
                                  ("explicit_uncertain", classification("atopic", 0.9, True))]:
                    request(key, "recommendation_gate", "/infer/disease-recommend", json={"disease": pred, "survey": survey()})
                for key, label, sv in recommendation_cases():
                    request(key, "label_fixture_generation", "/infer/disease-recommend", json={"disease": classification(label), "survey": sv})
                request("nonface_full_pipeline", "nonface_full_pipeline", "/infer/disease-analyze",
                        data={"survey": json.dumps(survey(selfReportedConcerns=["건조함"]), ensure_ascii=False)},
                        files={"image": ("solid.png", images["skin_color_solid"], "image/png")})
                recs = [r for r in records if r["kind"] == "label_fixture_generation"]
                generations = [r for r in recs if r["structure"] is not None]
                durations = [r["structure"]["generation_ms"]/1000 for r in generations]
                negatives = [r for r in records if r["kind"] == "nonface_negative_control"]
                summary = {
                    "clinical_validation": False, "disease_accuracy_remeasured": False,
                    "reason": "GitHub issue #3 explicitly excludes train/test images; no labelled local disease image split found.",
                    "source_issue": "https://github.com/AI-SkinCare/skincare-ai/issues/3",
                    "model_info": info, "startup_s": round(startup_s, 2), "total_cases": len(records),
                    "http_status_as_expected": sum(r["http_as_expected"] for r in records),
                    "fixture_generation_cases": len(recs), "fixture_generations_returned": len(generations),
                    "fallback_count": sum(r["structure"]["used_fallback"] for r in generations),
                    "generation_seconds_median": statistics.median(durations) if durations else None,
                    "generation_seconds_max": max(durations) if durations else None,
                    "nonface_cases": len(negatives),
                    "nonface_high_confidence_acceptances": sum(r["http_status"]==200 and not r["response"].get("uncertain", True) for r in negatives),
                    "ingredients_frequency": dict(Counter(i["name"] for r in generations for i in r["response"]["recommendation"]["ingredients"])),
                    "gpu_peak_memory_mib": monitor.peak_memory_mib,
                    "automatic_flags_need_manual_review": True,
                }
                (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
        finally:
            stop_process_tree(proc)
            monitor.stop.set()
            monitor.thread.join(timeout=8)


if __name__ == "__main__":
    main()
