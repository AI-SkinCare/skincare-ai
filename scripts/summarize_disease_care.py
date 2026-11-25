"""Audit completed local care trials, including the actual displayed response."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.ingredient_evidence import load_evidence


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact(text):
    return re.sub(r"\s+", "", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    done = json.loads((out / "done.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    trials = rows(out / "responses.jsonl")
    initial_requests = len(trials)
    followup = rows(out / "followup.responses.jsonl") if (out / "followup.done.json").exists() else []
    replacements = {row["case"]: row for row in followup}
    trials = [replacements.get(row["case"], row) for row in trials]
    replay = rows(out / "replay.jsonl")
    correction_path = out / "protocol_correction.json"
    correction = json.loads(correction_path.read_text(encoding="utf-8")) if correction_path.exists() else {}
    assert len(trials) == done["cases"] == len(plan["cases"])
    assert len({row["case"] for row in trials}) == len(trials)
    store = load_evidence()
    assert store["sha256"] == plan["source_sha256"]
    catalog = {card["id"]: card for card in store["claims"]}
    audits = []
    for row in trials:
        body = row["response"]
        rec = body.get("recommendation")
        gen = row["generation"]
        audit = dict(case=row["case"], group=row["group"], http=row["http"], seconds=row["seconds"],
                     status=body.get("recommendation_status"), origin=body.get("generation_origin"))
        actual = "http_" + str(row["http"]) if row["http"] != 200 else body.get("recommendation_status")
        expected = correction.get("expected_overrides", {}).get(row["case"], row["expected"])
        audit["expected_status"] = actual == expected
        if rec:
            care = gen["review"]["product_conditions"]["disease_care"]
            routine = " ".join(step["action"] for step in rec["routine"])
            reasoning = rec["reasoning"][2]
            audit["care"] = [dict(id=rule["id"], source_id=rule["source_id"],
                text=rule["text"], routine_present=compact(rule["text"]) in compact(routine),
                reasoning_present=compact(rule["text"]) in compact(reasoning)) for rule in care["bindings"]]
            assert all(rule["routine_present"] and rule["reasoning_present"] for rule in audit["care"]), audit
            audit["ingredients"] = []
            binding_by_name = {b["ingredient"]: b for b in rec["ingredient_evidence"]}
            for ingredient in rec["ingredients"]:
                binding = binding_by_name[ingredient["name"]]
                card = catalog[binding["claim_id"]]
                item = dict(name=ingredient["name"], purpose=bool(ingredient["purpose"]),
                    exact_card_effect=compact(ingredient["name"] + " " + ingredient["purpose"]) == compact(card["effect"]),
                    verbatim_card_conditions=all(compact(text) in compact(routine) for text in card["conditions"]),
                    source_urls={(s["source_id"], s["url"]) for s in binding["sources"]} ==
                                {(sid, store["sources"][sid]["url"]) for sid in card["sources"]})
                assert all(item[key] for key in ("purpose", "exact_card_effect", "source_urls")), item
                audit["ingredients"].append(item)
            audit["avoidance"] = not ({i["name"] for i in rec["ingredients"]} & set(row["survey"].get("avoidIngredients", [])))
            audit["product_usage_checks_pass"] = not gen["review"]["product_conditions"]["issues"]
            audit["no_blank_output"] = bool(rec["ingredients"] and rec["routine"] and rec["answer"] and len(rec["reasoning"]) == 3)
            audit["neutral_first_step"] = rec["reasoning"][0] == "사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다."
            audit["provenance"] = rec["generation_origin"] == body["generation_origin"] and rec["used_fallback"] == (body["generation_origin"] == "source_cards")
            audit["raw_model_preserved"] = all(gen["result"][key] == gen["trace"][-1]["result"][key] for key in ("raw_text", "parsed"))
            assert all(audit[key] for key in ("avoidance", "no_blank_output", "neutral_first_step", "provenance", "raw_model_preserved", "product_usage_checks_pass")), audit
        else:
            audit["no_model_call"] = gen is None
            assert audit["no_model_call"]
        assert audit["expected_status"]
        audits.append(audit)
    supplied = [row for row in audits if "ingredients" in row]
    care_cases = [row for row in supplied if row["care"]]
    before = [row for row in replay if row.get("review", {}).get("required")]
    seconds = [row["seconds"] for row in supplied]
    summary = dict(
        requests=len(trials), status_passes=sum(row["expected_status"] for row in audits),
        actual_http_requests=initial_requests+len(followup), final_code_fresh_followup=len(followup),
        statuses=dict(Counter(row["status"] or "invalid_image_http_" + str(row["http"]) for row in audits)),
        origins=dict(Counter(row["origin"] or "no_response_origin" for row in audits)),
        provided=len(supplied), care_cases=len(care_cases),
        care_sentences=sum(len(row["care"]) for row in supplied),
        class_care_retained=sum(all(rule["routine_present"] and rule["reasoning_present"] for rule in row["care"]) for row in care_cases),
        ingredient_items=sum(len(row["ingredients"]) for row in supplied),
        product_usage_check_passes=sum(row["product_usage_checks_pass"] for row in supplied),
        verbatim_card_conditions=sum(item["verbatim_card_conditions"] for row in supplied for item in row["ingredients"]),
        previous_responses=len(replay), previous_care_cases=len(before),
        previous_care_omissions=sum(bool(row["review"]["issues"]) for row in before),
        generation_seconds=dict(median=statistics.median(seconds), max=max(seconds), min=min(seconds)),
        image_requests=sum(row["photo"] for row in trials),
        actual_valid_image_classifications=sum(row["photo"] and row["case"] != "invalid_photo" for row in trials),
        novel_fictional_surveys=sum(row["group"] == "new_fictional_survey" for row in audits),
        protocol_correction=correction,
        new_patient_validation=False, clinical_efficacy_accuracy=None,
        audits=audits,
    )
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "audits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
