"""Evaluate provided labelled images through the real image and recommendation APIs.

Labels are used only for scoring. They are never sent to the recommendation API.
The default survey is a declared hypothetical common test profile, not image metadata.
Run classify first, then all to resume with the real Qwen model. No source images are copied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import socket
import statistics
import subprocess
import sys
import time

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.schemas import DISEASE_LABELS, DiseaseAnalyzeResponse, DiseasePrediction, Survey
from evaluate_disease_pipeline import Monitor, inspect_recommendation, review_candidates, stop_process_tree

LABELS = list(DISEASE_LABELS.values())
COMMON_SURVEY = {"gender": "female", "age": 30, "skinType": "normal", "selfReportedConcerns": []}


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_record(path, data):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_samples(input_root):
    root = input_root.resolve()
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"rel_path", "identifier", "class_name", "view", "label"} <= rows[0].keys():
        raise ValueError("labels.csv needs rel_path, identifier, class_name, view, label")
    samples, seen = [], set()
    for row in rows:
        rel = Path(row["rel_path"].replace("\\", "/"))
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image path: {rel}")
        if rel.as_posix() in seen:
            raise ValueError(f"Duplicate CSV path: {rel}")
        seen.add(rel.as_posix())
        if row["class_name"] not in LABELS or int(row["label"]) != LABELS.index(row["class_name"]):
            raise ValueError(f"Invalid class name/index: {rel}")
        if row["class_name"] not in rel.parts or row["view"] not in rel.parts:
            raise ValueError(f"CSV and directory labels disagree: {rel}")
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError(f"Unsupported image: {rel}")
            width, height, fmt = image.width, image.height, image.format
            image.verify()
        samples.append({"rel_path": rel.as_posix(), "identifier": row["identifier"],
                        "true_label": row["class_name"], "view": row["view"],
                        "sha256": digest(path), "bytes": path.stat().st_size,
                        "width": width, "height": height, "format": fmt})
    images_on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    if images_on_disk != seen:
        raise ValueError("CSV and image inventory do not match")
    return samples


def classification_metrics(records, total):
    valid = [r for r in records if r["http_status"] == 200]
    matrix = [[0] * len(LABELS) for _ in LABELS]
    support = Counter(r["true_label"] for r in records)
    for record in valid:
        matrix[LABELS.index(record["true_label"])][LABELS.index(record["response"]["label"])] += 1
    per_class = []
    for k, label in enumerate(LABELS):
        tp, actual = matrix[k][k], support[label]
        predicted = sum(row[k] for row in matrix)
        precision, recall = (tp / predicted if predicted else 0), (tp / actual if actual else 0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class.append({"label": label, "support": actual, "correct": tp,
                          "precision": precision, "recall": recall, "f1": f1})
    certain = [r for r in valid if not r["response"]["uncertain"]]
    views = sorted({r["view"] for r in records})
    return {"total_images": total, "http_success": len(valid), "http_errors": len(records)-len(valid),
            "correct": sum(r["correct"] for r in valid),
            "accuracy": sum(r["correct"] for r in valid) / total if total else None,
            "macro_f1": statistics.mean(c["f1"] for c in per_class),
            "uncertain": len(valid)-len(certain), "coverage": len(certain) / total if total else None,
            "accuracy_among_accepted": sum(r["correct"] for r in certain)/len(certain) if certain else None,
            "by_view": {view: {"n": sum(r["view"]==view for r in records),
                               "correct": sum(r["view"]==view and r.get("correct", False) for r in valid),
                               "accuracy": sum(r["view"]==view and r.get("correct", False) for r in valid)
                                           /sum(r["view"]==view for r in records)} for view in views},
            "class_order": LABELS, "confusion_matrix": matrix, "per_class": per_class}


def request_recommendation(prediction, survey):
    # Truth labels, paths and filenames must not leak into the AI input.
    return {"disease": DiseasePrediction.model_validate(prediction).model_dump(), "survey": survey, "metrics": None}


def write_reports(out, samples, classifications, recommendations, phase, started):
    valid_recs = [r for r in recommendations if r["http_status"] == 200]
    generated = [r for r in valid_recs if r["response"]["recommendation"] is not None]
    latencies = [r["response"]["recommendation"]["inference_ms"]/1000 for r in generated]
    summary = {"updated_at": now(), "phase": phase, "started_at": started,
               "actual_image_input": True, "recommendation_labels_from_model_not_ground_truth": True,
               "clinical_validation": False, "train_test_overlap_independently_verified": False,
               "survey_is_common_hypothetical_profile": True,
               "classification": classification_metrics(classifications, len(samples)),
               "recommendation": {"processed": len(recommendations), "total": len(samples),
                                  "http_success": len(valid_recs), "http_errors": len(recommendations)-len(valid_recs),
                                  "status_counts": dict(Counter(r["response"]["recommendation_status"] for r in valid_recs)),
                                  "generated": len(generated),
                                  "fallback": sum(bool(r["structure"]["used_fallback"]) for r in generated),
                                  "generation_seconds_median": statistics.median(latencies) if latencies else None,
                                  "generation_seconds_max": max(latencies) if latencies else None,
                                  "empty_ingredients": sum(not r["response"]["recommendation"]["ingredients"] for r in generated),
                                  "empty_routine": sum(not r["response"]["recommendation"]["routine"] for r in generated),
                                  "ingredient_items": sum(r["structure"]["ingredient_count"] for r in generated),
                                  "purpose_missing": sum(r["structure"]["empty_purpose_count"] for r in generated),
                                  "generated_from_wrong_classification": sum(not r["classification_correct"] for r in generated)}}
    write_json(out / "summary.json", summary)
    columns = ["rel_path", "true_label", "view", "prediction", "confidence", "uncertain", "correct", "inference_ms", "http_status"]
    with (out / "classification.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for r in classifications:
            p = r["response"] if r["http_status"]==200 else {}
            writer.writerow({**{k:r[k] for k in ("rel_path","true_label","view","http_status")},
                             "prediction":p.get("label"), "confidence":p.get("confidence"),
                             "uncertain":p.get("uncertain"), "correct":r.get("correct"), "inference_ms":p.get("inference_ms")})
    c, rec = summary["classification"], summary["recommendation"]
    lines = ["# 제공 이미지 96장 실제 파이프라인 실행", "",
             "사진은 원래 폴더에서 읽고 복사·외부 전송하지 않았다. 분류는 실제 이미지 API를 호출했다.",
             "추천은 예측된 분류 JSON을 입력으로 사용하며 정답 라벨을 입력하지 않는다.",
             "설문은 실제 사용자 정보가 아닌 여성·30세·중성·자가 고민 없음의 공통 가정이다.",
             "labels.csv 대비 표본 평가이며 학습 중복·데이터 출처의 독립성이나 임상 유효성을 입증한 결과는 아니다.", "",
             f"- 상태: {phase}", f"- 분류: {c['correct']}/{c['total_images']}, 정확도 {c['accuracy']:.2%}, Macro F1 {c['macro_f1']:.4f}",
             f"- 불확실 추천 보류 대상: {c['uncertain']}건", f"- 추천 API: {rec['processed']}/{rec['total']}, 상태 {rec['status_counts']}, HTTP 오류 {rec['http_errors']}건",
             f"- 규칙 폴백: {rec['fallback']}건", "", "## 클래스별 분류", "",
             "| 정답 클래스 | 건수 | 정답 | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for row in c["per_class"]:
        lines.append(f"| {row['label']} | {row['support']} | {row['correct']} | {row['recall']:.2%} | {row['f1']:.4f} |")
    lines += ["", "## 오분류", "", "| 이미지 | 정답 | 예측 | 점수 | 보류 |", "|---|---|---|---:|---|"]
    for r in classifications:
        if r["http_status"]==200 and not r["correct"]:
            p=r["response"]; lines.append(f"| {r['rel_path']} | {r['true_label']} | {p['label']} | {p['confidence']:.4f} | {p['uncertain']} |")
    lines += ["", "개별 원문은 classifications.jsonl / recommendations.jsonl, 입력·버전·해시는 manifest.json / provenance.json에 보존한다."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,required=True)
    parser.add_argument("--processed-dir",type=Path,required=True)
    parser.add_argument("--phase",choices=["classify","all"],default="all")
    parser.add_argument("--code-revision",default="unknown")
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    samples=load_samples(args.input_root)
    manifest={"input_root":str(args.input_root.resolve()),"labels_sha256":digest(args.input_root/"labels.csv"),
              "samples":samples,"common_hypothetical_survey":COMMON_SURVEY,"code_revision":args.code_revision,
              "classifier_sha256":digest(ROOT/"model/weights/densenet121_s2.pt"),
              "adapter_sha256":digest(args.adapter/"adapter_model.safetensors"),
              "ingredient_dict_sha256":digest(args.processed_dir/"ingredient_dict.json")}
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text(encoding="utf-8"))!=manifest:
            raise ValueError("Input, model or code changed; use a new output directory")
    else:
        write_json(out/"manifest.json",manifest)
    classes=read_records(out/"classifications.jsonl")
    recs=read_records(out/"recommendations.jsonl")
    started=now()
    env=os.environ.copy()
    token=secrets.token_hex(24)
    env.update(REC_BACKEND="mock" if args.phase=="classify" else "sllm", REC_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct",
               REC_ADAPTER_DIR=str(args.adapter.resolve()), PROCESSED_DIR=str(args.processed_dir.resolve()),
               DISEASE_BACKEND="densenet",DISEASE_DEVICE="cpu",DISEASE_THRESHOLD="0.6",SKIN_BACKEND="stub",
               HF_HOME="D:/hf_cache",HF_HUB_CACHE="D:/hf_cache/hub",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               PYTHONUTF8="1",AI_INTERNAL_TOKEN=token)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    monitor=Monitor(out/f"gpu-{args.phase}.jsonl");monitor.thread.start()
    with (out/f"server-{args.phase}.stdout.log").open("a",encoding="utf-8") as stdout, (out/f"server-{args.phase}.stderr.log").open("a",encoding="utf-8") as stderr:
        proc=subprocess.Popen([sys.executable,"-m","uvicorn","server.main:app","--host","127.0.0.1","--port",str(port)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        write_json(out/f"process-{args.phase}.json",{"runner_pid":os.getpid(),"server_pid":proc.pid,"port":port,"started_at":started})
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}",headers={"X-Internal-Token":token},timeout=httpx.Timeout(180,connect=5),trust_env=False) as client:
                deadline=time.monotonic()+180
                while time.monotonic()<deadline:
                    if proc.poll() is not None: raise RuntimeError("Server exited during startup")
                    try:
                        health=client.get("/health",timeout=3)
                        if health.status_code==200 and health.json().get("disease_loaded"):break
                    except httpx.RequestError:pass
                    time.sleep(1)
                else:raise RuntimeError("Server did not become ready within 180 seconds")
                provenance={"updated_at":now(),"code_root":str(ROOT),"code_revision":args.code_revision,
                            "code_source":"clean export of committed pipeline, excluding unrelated working-tree experiments",
                            "model_info":client.get("/model-info").json(),"survey":COMMON_SURVEY,
                            "survey_is_hypothetical":True,"truth_used_only_for_scoring":True,
                            "clinical_validation":False,"train_test_overlap_independently_verified":False}
                write_json(out/"provenance.json",provenance)
                known={r["rel_path"] for r in classes}
                for sample in samples:
                    if sample["rel_path"] in known:continue
                    path=args.input_root/sample["rel_path"]
                    t0=time.perf_counter()
                    response=client.post("/infer/disease",files={"image":(path.name,path.read_bytes(),"image/png" if sample["format"]=="PNG" else "image/jpeg")})
                    body=response.json()
                    if response.status_code==200:DiseasePrediction.model_validate(body)
                    record={**{k:sample[k] for k in ("rel_path","identifier","true_label","view")},
                            "http_status":response.status_code,"elapsed_s":round(time.perf_counter()-t0,4),
                            "response":body,"correct":response.status_code==200 and body["label"]==sample["true_label"]}
                    append_record(out/"classifications.jsonl",record);classes.append(record)
                    print(f"[classify {len(classes)}/{len(samples)}] {sample['true_label']} -> {body.get('label')} correct={record['correct']}",flush=True)
                summary=write_reports(out,samples,classes,recs,"classification_complete",started)
                print("[classification] "+json.dumps(summary["classification"],ensure_ascii=False),flush=True)
                if args.phase=="classify":return
                if len(classes)!=len(samples) or any(r["http_status"]!=200 for r in classes):
                    raise RuntimeError("Classification incomplete; refusing recommendation generation")
                known={r["rel_path"] for r in recs}
                # Interleave classes and viewpoints so early progress covers all conditions.
                groups={}
                for r in classes:groups.setdefault((r["true_label"],r["view"]),[]).append(r)
                rng=random.Random(42)
                for group in groups.values():rng.shuffle(group)
                ordered=[]
                while any(groups.values()):
                    for key in sorted(groups):
                        if groups[key]:ordered.append(groups[key].pop())
                for c in ordered:
                    if c["rel_path"] in known:continue
                    payload=request_recommendation(c["response"],COMMON_SURVEY)
                    t0=time.perf_counter()
                    response=client.post("/infer/disease-recommend",json=payload)
                    body=response.json()
                    if response.status_code==200:DiseaseAnalyzeResponse.model_validate(body)
                    rec=body.get("recommendation")
                    record={**{k:c[k] for k in ("rel_path","identifier","true_label","view")},
                            "classification_correct":c["correct"],"http_status":response.status_code,
                            "elapsed_s":round(time.perf_counter()-t0,3),"request":payload,"response":body,
                            "structure":inspect_recommendation(rec),"review_candidates":review_candidates(rec)}
                    append_record(out/"recommendations.jsonl",record);recs.append(record)
                    summary=write_reports(out,samples,classes,recs,"recommending",started)
                    print(f"[recommend {len(recs)}/{len(samples)}] {c['true_label']}/{c['view']} pred={c['response']['label']} status={body.get('recommendation_status')} http={response.status_code} {record['elapsed_s']:.1f}s",flush=True)
                summary=write_reports(out,samples,classes,recs,"complete",started)
                print("[complete] "+json.dumps(summary["recommendation"],ensure_ascii=False),flush=True)
        except BaseException as exc:
            write_json(out/"failure.json",{"at":now(),"phase":args.phase,"error":type(exc).__name__,"message":str(exc)})
            raise
        finally:
            stop_process_tree(proc);monitor.stop.set();monitor.thread.join(timeout=8)


if __name__=="__main__":
    main()
