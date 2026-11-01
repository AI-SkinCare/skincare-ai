import copy

import pytest

from src.llm.review_evidence import inventory, reference_ids, resolve_reference, row_status
from src.llm.prepare_followup_review import diagnostic_inputs, dictionary_matches, validate_external_cases
from src.llm.prepare_quality_data import serving_key
from src.llm.schema import CONCERNS, RecInput


def test_reference_normalization_does_not_guess_plain_numbers():
    assert reference_ids("PMID: 16029679; https://doi.org/10.3390/COSMETICS4020016") == ["doi:10.3390/cosmetics4020016", "pmid:16029679"]
    assert reference_ids("시험 인원 16029679") == []
    assert any(k.startswith("placeholder:") for k in reference_ids("DOI:10.xxxx/xxxxx"))
    assert reference_ids("https://pubmed.ncbi.nlm.nih.gov/12345678/") == ["pmid:12345678"]


def test_real_identifier_is_not_claim_approval(monkeypatch):
    from src.llm import review_evidence as module
    monkeypatch.setattr(module, "fetch_json", lambda _: {"message": {"DOI": "10.1111/jocd.13276", "title": ["JCD Editorial January, 2020"], "type": "journal-article"}})
    result = resolve_reference("doi:10.1111/jocd.13276")
    assert result["status"] == "metadata_resolved"
    assert "non_efficacy_study_title_review" in result["review_flags"]
    status = row_status([result["reference"]], {result["reference"]: result})
    assert status["all_identifiers_resolved"] and not status["efficacy_approved"]


def test_lookup_failure_does_not_mean_invalid_efficacy(monkeypatch):
    from src.llm import review_evidence as module
    def failure(_):
        raise TimeoutError()
    monkeypatch.setattr(module, "fetch_json", failure)
    result = resolve_reference("pmid:1")
    assert result["status"] == "lookup_error"
    assert result["claim_status"] == "not_expert_verified"


def test_inventory_deduplicates_references_within_record():
    refs, rows = inventory([{"id": "a", "concern": "모공", "evidence_sources": ["PMID:1", "PMID: 1"]},
                            {"id": "b", "concern": "주름", "evidence_sources": "PMID:1"}])
    assert refs["pmid:1"]["row_count"] == 2
    assert rows["a"]["reference_ids"] == ["pmid:1"]


def test_exact_match_keeps_parts_distinct_and_blank_flags_unknown():
    matches = dictionary_matches({"PLANTLEAFEXTRACT": "잎", "PLANTROOTEXTRACT": "뿌리"},
                                 [{"inci_name": "plant leaf extract", "comedogenic": ""}])
    assert next(r for r in matches if r["ingredient"] == "잎")["exact_matches"][0]["comedogenic"] == ""
    assert not next(r for r in matches if r["ingredient"] == "뿌리")["exact_matches"]


def test_synthetic_design_covers_all_concerns_but_is_not_new_human_gold():
    blocked = {serving_key({"input": RecInput("여성", 25, "건성", [CONCERNS[0]]).to_dict()})}
    rows = diagnostic_inputs(set(blocked))
    keys = {serving_key(r) for r in rows}
    assert len(keys) == len(rows) == 120 and not keys & blocked
    assert {c for r in rows for c in r["input"]["concerns"]} == set(CONCERNS)
    assert all(r["gold_by_concern"] is None and not r["is_independent_human_test"] for r in rows)
    assert not validate_external_cases(rows, blocked, min_per_concern=0, min_multi=0)["ready"]


def approved_case():
    return {"id": "new_1", "input": RecInput("여성", 30, "건성", [CONCERNS[0], CONCERNS[3]]).to_dict(),
            "source_origin": "new_human_collection", "collection_batch_id": "team_batch_1",
            "gold_by_concern": {c: {"acceptable_ingredients": ["판테놀"], "evidence_urls": ["https://example.org/paper"]} for c in (CONCERNS[0], CONCERNS[3])},
            "review": {"status": "approved", "reviewer_id": "expert_1", "qualification": "dermatologist", "reviewed_at": "2026-10-06", "claim_evidence_verified": True}}


def test_independent_gate_blocks_seen_profiles_missing_concern_gold_and_no_review():
    row = approved_case()
    assert validate_external_cases([row], set(), 0, 0)["ready"]
    assert not validate_external_cases([row], {serving_key(row)}, 0, 0)["ready"]
    missing = copy.deepcopy(row)
    del missing["gold_by_concern"][CONCERNS[3]]
    assert not validate_external_cases([missing], set(), 0, 0)["ready"]
    pending = copy.deepcopy(row)
    pending["review"]["status"] = "pending"
    assert not validate_external_cases([pending], set(), 0, 0)["ready"]
    assert not validate_external_cases([row, copy.deepcopy(row)], set(), 0, 0)["ready"]


def test_independent_gate_requires_rare_and_multi_coverage():
    result = validate_external_cases([approved_case()], set())
    assert not result["ready"]
    assert any(e.get("concern") == "붉어짐(홍조)" for e in result["errors"])
    assert any("multi-concern" in e["reason"] for e in result["errors"])


def test_external_scores_require_each_concern_and_keep_fallback_separate():
    from src.llm.evaluate_external_cases import external_metrics
    gold = {"모공": {"acceptable_ingredients": ["A"], "avoid_ingredients": ["X"]},
            "주름": {"acceptable_ingredients": ["B"]}}
    result = external_metrics(["A", "X"], gold, False)
    assert not result["all_concerns_have_ingredient_match"]
    assert result["avoid_ingredient_matches"] == ["X"]
    assert result["recognized_name_precision_at_3"] == pytest.approx(1/3)
    assert external_metrics(["A", "B"], gold, True)["recognized_name_precision_at_3"] is None


def test_sealed_external_test_rejects_changed_model_or_cases(tmp_path):
    from src.llm.evaluate_external_cases import seal_manifest
    p = tmp_path / "seal.json"
    seal_manifest(p, {"model": "a", "cases": "v1"})
    seal_manifest(p, {"model": "a", "cases": "v1"})
    with pytest.raises(ValueError, match="sealed"):
        seal_manifest(p, {"model": "b", "cases": "v1"})


def test_pipeline_rejects_different_eval_ids_and_dataset(tmp_path, monkeypatch):
    import json
    from scripts import run_quality_followup as runner
    from src.llm.prepare_quality_data import digest
    data = tmp_path / "data/processed/sllm_v3"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({"dataset_id": "expected"}), encoding="utf8")
    (data / "val_eval.jsonl").write_text(json.dumps({"id": "a"}) + "\n", encoding="utf8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "experiment_fingerprint", lambda: {"fixed": "hash"})
    p = tmp_path / "result.json"
    summary = {"dataset_id": "expected", "split": "val_eval", "evaluated_ids_hash": digest(["a"])}
    p.write_text(json.dumps({"summary": summary, "items": [{"id": "a"}]}), encoding="utf8")
    runner.eval_artifact([p], {"fixed": "hash"})
    summary["evaluated_ids_hash"] = digest(["b"])
    p.write_text(json.dumps({"summary": summary, "items": [{"id": "a"}]}), encoding="utf8")
    with pytest.raises(RuntimeError, match="unexpected evaluation"):
        runner.eval_artifact([p], {"fixed": "hash"})
