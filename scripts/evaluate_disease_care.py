"""Local care-condition regression plus fresh FastAPI/DenseNet/Qwen inference.

Images are only read at their original paths. New cases change fictional surveys
on existing photos; they are not a new patient generalization benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def code_hashes():
    paths = list((ROOT / "server").glob("*.py")) + [ROOT / "configs/ingredient_evidence.json", Path(__file__)]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def evaluate(args):
    os.environ.update(HF_HOME="D:/hf_cache", HF_HUB_CACHE="D:/hf_cache/hub",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from fastapi.testclient import TestClient
    from server import main as api
    from server.adapters import load_mapping
    from server.config import Settings
    from server.disease_care import review_care
    from server.disease_pipeline import build_disease_rec_input, load_disease_mapping
    from server.ingredient_evidence import load_evidence, retrieve_evidence
    from server.schemas import DiseasePrediction, Survey

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    previous = rows(args.baseline / "recommendations.jsonl")
    prior_cases = {row["case"]: row for row in previous}
    prior_plan = {row["case"]: row for row in json.loads((args.baseline / "recommendation_plan.json").read_text(encoding="utf-8"))}
    provenance = json.loads((args.baseline / "plan.json").read_text(encoding="utf-8"))
    store = load_evidence()
    mapping, disease_mapping = load_mapping(), load_disease_mapping()
    for row in previous:
        body = row["response"]
        rec = body.get("recommendation")
        record = dict(case=row["case"], status=body["recommendation_status"], origin=body["generation_origin"])
        if rec:
            disease = DiseasePrediction(**body["disease"])
            survey = Survey(**prior_plan[row["case"]]["survey"])
            x, _, _ = build_disease_rec_input(survey, None, disease, mapping, disease_mapping, .6)
            bundle = retrieve_evidence(store, x, disease, survey.avoidIngredients)
            parsed = dict(reasoning=[dict(content=text) for text in rec["reasoning"]], answer=rec["answer"])
            response = SimpleNamespace(ingredients=[SimpleNamespace(**item) for item in rec["ingredients"]])
            record["review"] = review_care(parsed, response, bundle)
        append(args.output / "replay.jsonl", record)

    cases = [dict(case=key, group="existing", source_case=key, survey=prior_plan[key]["survey"],
                  photo=key in ("P004", "P023", "P080", "P092"), expected="generated")
             for key in ("P004", "P023", "P080", "P092", "P034", "avoid_partial")]
    novel = [
        ("new_atopic_sensitive", "P023", dict(gender="female", age=22, skinType="dry",
            selfReportedConcerns=["과각질/악건성", "붉어짐(홍조)"], avoidIngredients=["세라마이드"],
            note="세라마이드는 제외해주세요. 보습제와 선크림의 성분·효능·사용 조건을 각각 알려주세요.")),
        ("new_psoriasis_acne", "P004", dict(gender="male", age=48, skinType="combination",
            selfReportedConcerns=["여드름/뾰루지"], avoidIngredients=["페트롤라툼"],
            note="보습제와 선크림을 알려주세요. 페트롤라툼은 제외하고 세정 후 사용 조건도 써주세요.")),
        ("new_rosacea_avoid_nia", "P080", dict(gender="female", age=54, skinType="dry",
            selfReportedConcerns=["붉어짐(홍조)"], avoidIngredients=["나이아신아마이드"],
            note="나이아신아마이드를 제외한 보습제와 선크림을 알려주세요. 분류 점수로 장벽 손상을 단정해도 되나요?")),
        ("new_seborrheic_sunscreen_only", "P092", dict(gender="male", age=35, skinType="oily",
            selfReportedConcerns=[], avoidIngredients=["글리세린"],
            note="보습제는 추천하지 마세요. 선크림 성분과 완제품 사용 조건만 알려주세요.")),
    ]
    cases += [dict(case=key, group="new_fictional_survey", source_case=source,
                   survey=survey, photo=True, expected="generated") for key, source, survey in novel]
    normal = next(row["case"] for row in previous if row["response"]["recommendation_status"] == "no_concern")
    for key, expected in ((normal, "no_concern"), ("P068", "uncertain"), ("avoid_all", "quality_rejected")):
        cases.append(dict(case="gate_" + key, group="gate", source_case=key,
                          survey=prior_plan[key]["survey"], photo=False, expected=expected))
    cases.append(dict(case="invalid_photo", group="invalid_image", source_case="P004",
                      survey=prior_plan["P004"]["survey"], photo=True, expected="http_422"))
    hashes = code_hashes()
    assets = args.assets_dir
    weights = assets / "model/weights/densenet121_s2.pt"
    adapter = assets / "outputs/rec_sllm_v2/adapter"
    knowledge = assets / "outputs/disease_precision_20261110/requirements_audit/medical_knowledge_train_only.json"
    cfg = Settings(
        internal_token=secrets.token_urlsafe(24), rec_backend="sllm",
        rec_base_model="Qwen/Qwen2.5-7B-Instruct", rec_adapter_dir=str(adapter),
        rec_max_new_tokens=2400, rec_timeout_s=120, disease_backend="densenet", disease_device="cpu",
        disease_threshold=.6, disease_weights=weights, disease_knowledge_path=knowledge,
        processed_dir=assets / "data/processed/sllm", rec_evidence_mode="enforce", rec_evidence_use_adapter=False,
    )
    assert digest(weights) == provenance["weights_sha256"]
    assert digest(adapter / "adapter_model.safetensors") == provenance["adapter_sha256"]
    photos = {item["case"]: item for item in provenance["photos"]}
    for case in cases:
        index = prior_plan[case["source_case"]]["index"]
        photo = photos[f"P{index+1:03}"]
        assert digest(args.photos / photo["rel_path"]) == photo["sha256"]
    dump(args.output / "plan.json", dict(cases=cases, code_hashes=hashes,
        source_sha256=store["sha256"], source_version=store["version"], gpu=torch.cuda.get_device_name(),
        weights_sha256=digest(weights), adapter_sha256=digest(adapter / "adapter_model.safetensors"),
        adapter_disabled_for_source_generation=True, classifier_threshold=.6,
        baseline=str(args.baseline), photos=str(args.photos), surveys_fictional=True,
        new_patient_validation=False, clinical_efficacy_measured=False,
        transport="in-process FastAPI TestClient", original_photos_copied=False))
    traces = []
    original = api.generate_grounded

    def observed(*a, **kw):
        value = original(*a, **kw)
        traces.append(dict(result=asdict(value[0]), review=value[3], trace=value[4]))
        return value

    api.generate_grounded = observed
    headers = {"X-Internal-Token": cfg.internal_token}
    with TestClient(api.create_app(cfg)) as client:
        health = client.get("/health").json()
        assert health["rec_loaded"] and health["disease_loaded"], health
        dump(args.output / "health.json", health)
        for case in cases:
            before = len(traces)
            start = time.perf_counter()
            prior = prior_cases[case["source_case"]]["response"]
            if case["photo"]:
                index = prior_plan[case["source_case"]]["index"]
                photo = photos[f"P{index+1:03}"]
                content = b"\x89PNG\r\n\x1a\ninvalid payload" if case["case"] == "invalid_photo" else (args.photos / photo["rel_path"]).read_bytes()
                response = client.post("/infer/disease-analyze", headers=headers,
                    data={"survey": json.dumps(case["survey"], ensure_ascii=False)},
                    files={"image": ("photo.png", content, "image/png")})
            else:
                response = client.post("/infer/disease-recommend", headers=headers,
                    json=dict(disease=prior["disease"], survey=case["survey"]))
            body = response.json()
            record = dict(**case, http=response.status_code, seconds=time.perf_counter()-start, response=body,
                          generation=traces[-1] if len(traces)>before else None)
            append(args.output / "responses.jsonl", record)
            actual = "http_422" if response.status_code == 422 else body.get("recommendation_status")
            print(case["case"], response.status_code, actual, body.get("generation_origin"), round(record["seconds"], 2), flush=True)
            assert actual == case["expected"], (case["case"], body)
            if actual == "generated":
                rec = body["recommendation"]
                assert rec["quality_status"] == "ok" and rec["evidence_status"] == "source_linked"
                assert rec["ingredients"] and rec["routine"] and all(i["purpose"] for i in rec["ingredients"])
                assert not record["generation"]["review"]["issues"]
                assert not ({i["name"] for i in rec["ingredients"]} & set(case["survey"].get("avoidIngredients", [])))
                if case["photo"]:
                    pred = body["disease"]
                    assert pred["label_en"] == prior["disease"]["label_en"]
                    assert pred["probabilities"] == prior["disease"]["probabilities"]
            else:
                assert record["generation"] is None
    assert code_hashes() == hashes, "Serving code changed during evaluation"
    dump(args.output / "done.json", dict(cases=len(cases), status="complete"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("plan.json", "replay.jsonl", "responses.jsonl")):
        raise RuntimeError("Use a fresh output directory; never overwrite previous measurements")
    try:
        evaluate(args)
    except BaseException as exc:
        dump(args.output / "failure.json", dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
