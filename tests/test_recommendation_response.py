import json

import pytest
from fastapi.testclient import TestClient

from server import main
from server.config import Settings
from server.recommendation_response import prepare_recommendation, reports_sensitivity
from src.llm.fallback import RuleBasedRecommender
from src.llm.inference import MockRecommender, RecResult, SLLMRecommender
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter
from src.llm.schema import RecInput, build_messages


def sample(answer="판테놀을 보습 관리 후보로 제안합니다."):
    return {"reasoning": [
        {"step": 1, "title": "문제 분석", "content": "입력한 고민은 민감성입니다."},
        {"step": 2, "title": "성분 선택", "content": "판테놀은 보습에 도움을 줄 수 있습니다."},
        {"step": 3, "title": "사용법", "content": "저녁에는 보습제를 사용하세요."}], "answer": answer}


def inputs():
    return RecInput("여성", 30, "중성", ["민감성(트러블/자극감)"])


def test_no_efficacy_is_invented_to_fill_a_missing_purpose():
    parsed = sample()
    parsed["reasoning"][1]["content"] = "판테놀을 후보로 제안합니다."
    result = RecResult(parsed, "test", False, 1, "", 10, 20)
    rec, _ = prepare_recommendation(result, inputs(), [], IngredientExtractor({"판테놀": "판테놀"}), OutputFilter({}))
    assert rec.ingredients[0].purpose is None
    assert rec.ingredients[0].purpose_status == "unavailable"
    assert rec.quality_status == "limited"
    assert "ingredient_purpose_unavailable" in rec.quality_issues
    assert rec.used_fallback is False


def test_removed_content_cannot_appear_in_another_response_field():
    parsed = sample("피부 해독에 탁월한 효과를 보장합니다.")
    result = RecResult(parsed, "test", False, 1, "", 1, 1)
    rec, findings = prepare_recommendation(result, inputs(), [], IngredientExtractor({"판테놀": "판테놀"}), OutputFilter({}))
    assert rec.quality_status == "withheld" and not rec.ingredients and not rec.routine
    assert "피부 해독" not in rec.model_dump_json()
    assert findings and not rec.used_fallback


def test_disease_http_quality_rejection_is_not_a_generated_success(monkeypatch, tmp_path):
    class InvalidContent(MockRecommender):
        def recommend(self, x):
            return RecResult(sample("부작용이 없고 안전하게 사용할 수 있습니다."), "broken", False, 1, "", 1, 1)

    monkeypatch.setattr(main, "MockRecommender", InvalidContent)
    labels = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
    payload = {"survey": {"gender": "female", "age": 30, "skinType": "normal"},
               "disease": {"label": "아토피", "label_en": "atopic", "confidence": .9, "uncertain": False,
                           "probabilities": {k: .9 if k == "아토피" else .02 for k in labels}}}
    with TestClient(main.create_app(Settings(internal_token="test", rec_backend="mock", processed_dir=tmp_path))) as client:
        response = client.post("/infer/disease-recommend", json=payload, headers={"X-Internal-Token": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["recommendation"] is None and data["recommendation_status"] == "quality_rejected"
    assert data["quality_issues"] and data["guidance"]


def test_fallback_covers_both_concerns_without_filling_duplicate_slots(tmp_path):
    path = tmp_path / "top.json"
    path.write_text(json.dumps({"모공": ["A", "B", "C"], "주름": ["A", "D", "E"]}), encoding="utf-8")
    rule = RuleBasedRecommender(path)
    x = RecInput("여성", 30, "중성", ["모공", "주름"])
    assert rule.ingredients_for(x) == ["A", "B", "D"]
    assert rule.ingredients_for(x, 1) == ["A"]


def test_serving_policy_is_opt_in_and_does_not_change_training_messages():
    before = build_messages(inputs())
    model = SLLMRecommender("unused", None, serving_guidance="serving-only-policy")
    assert model.serving_guidance == "serving-only-policy"
    assert build_messages(inputs()) == before
    assert "serving-only-policy" not in json.dumps(before)


def test_legacy_analyze_propagates_quality_rejection_without_classifier(monkeypatch, tmp_path):
    class InvalidContent(MockRecommender):
        def recommend(self, x):
            return RecResult(sample("부작용이 없고 안전하게 사용할 수 있습니다."), "broken", False, 1, "", 1, 1)
    monkeypatch.setattr(main, "MockRecommender", InvalidContent)
    with TestClient(main.create_app(Settings(internal_token="test", rec_backend="mock", processed_dir=tmp_path,
                                             disease_backend="disabled", skin_backend="stub"))) as client:
        response = client.post("/infer/analyze", headers={"X-Internal-Token": "test"},
                               files={"image": ("x.jpg", b"\xff\xd8\xff" + b"0" * 20, "image/jpeg")},
                               data={"survey": json.dumps({"gender": "female", "skinType": "normal"})})
    body = response.json()
    assert response.status_code == 200
    assert body["recommendation_status"] == "quality_rejected"
    assert body["recommendation"] is None and body["quality_issues"]


def test_quality_benchmark_includes_rejected_generation_time():
    from scripts.evaluate_recommendation_quality import summarize
    def row(status, ms):
        return {"http_status": 200, "response": {"recommendation_status": status, "recommendation": None,
                "quality_issues": ["no_supported_ingredients"] if status == "quality_rejected" else []},
                "generation": {"result": {"latency_ms": ms, "used_fallback": False}}}
    summary = summarize([row("quality_rejected", 7000), row("generated", 1000)])
    assert summary["generation_max_s"] == 7 and summary["generation_p50_s"] == 4
    assert summary["quality_rejected"] == 1 and summary["rejected_no_ingredients"] == 1


def test_incomplete_model_reasoning_does_not_pass_as_complete_recommendation():
    parsed = sample()
    parsed["reasoning"] = parsed["reasoning"][1:]
    result = RecResult(parsed, "test", False, 1, "", 10, 20)
    rec, _ = prepare_recommendation(result, inputs(), [], IngredientExtractor({"판테놀": "판테놀"}), OutputFilter({}))
    assert rec.quality_status == "withheld"
    assert "invalid_reasoning_structure" in rec.quality_issues


def test_removed_claim_does_not_leave_a_routine_ingredient_out_of_the_list():
    parsed = sample()
    parsed["reasoning"][1]["content"] += " 쑥잎추출물은 강력한 항염 효과가 있습니다."
    parsed["reasoning"][2]["content"] += " 저녁에는 쑥잎추출물이 포함된 제품을 사용하세요."
    result = RecResult(parsed, "test", False, 1, "", 10, 20)
    ext = IngredientExtractor({"판테놀": "판테놀", "쑥잎추출물": "쑥잎추출물"})
    rec, _ = prepare_recommendation(result, inputs(), [], ext, OutputFilter({}))
    names = {i.name: i for i in rec.ingredients}
    assert "쑥잎추출물" in names and names["쑥잎추출물"].purpose is None
    assert "강력한 항염" not in rec.model_dump_json()
    assert rec.quality_status == "limited"


def test_contradictory_use_and_avoid_is_withheld():
    parsed = sample()
    parsed["reasoning"][2]["content"] += " 판테놀은 사용하지 마세요."
    result = RecResult(parsed, "test", False, 1, "", 10, 20)
    rec, _ = prepare_recommendation(result, inputs(), [], IngredientExtractor({"판테놀": "판테놀"}), OutputFilter({}))
    assert rec.quality_status == "withheld"
    assert "ingredient_reference_inconsistent" in rec.quality_issues


@pytest.mark.parametrize("note,expected", [
    ("건조하고 민감해서 세안 후 따갑습니다.", True),
    ("민감하지 않습니다.", False),
    ("민감성은 없어요. 하지만 세안 후 따갑습니다.", True),
    ("보습을 원합니다.", False),
])
def test_explicit_sensitivity_in_survey_note(note, expected):
    assert reports_sensitivity(note) is expected


def test_sensitive_note_applies_even_when_primary_concern_is_acne():
    x = RecInput("여성", 48, "건성", ["여드름/뾰루지"], user_note="건조하고 민감해서 세안 후 따갑습니다.")
    parsed = sample()
    parsed["reasoning"][2]["content"] += " 주 1~2회 순한 각질 제거제를 사용하세요."
    parsed["answer"] += " 주 1~2회 순한 각질 제거를 병행하세요."
    result = RecResult(parsed, "test", False, 1, "", 1, 1)
    rec, _ = prepare_recommendation(result, x, [], IngredientExtractor({"판테놀": "판테놀"}), OutputFilter({}))
    assert "각질 제거" not in rec.model_dump_json()
    assert "sensitive_exfoliation" in rec.quality_issues


def test_sensitive_ingredient_selected_for_exfoliating_benefit_is_withheld():
    x = RecInput("여성", 48, "건성", ["여드름/뾰루지"], user_note="건조하고 민감해서 세안 후 따갑습니다.")
    parsed = sample()
    parsed["reasoning"][1]["content"] = "미네랄 솔트는 피부에 미네랄을 공급하고 각질 제거를 도와 건강한 피부 턴오버를 돕습니다."
    parsed["reasoning"][2]["content"] = "미네랄 솔트가 함유된 보습제를 사용하세요."
    parsed["answer"] = "미네랄 솔트는 각질 관리와 미네랄 공급을 돕습니다."
    rec, _ = prepare_recommendation(RecResult(parsed, "test", False, 1, "", 1, 1), x, [],
                                    IngredientExtractor({"미네랄솔트": "미네랄 솔트"}), OutputFilter({}))
    assert rec.quality_status == "withheld" and not rec.ingredients
    assert "restricted_ingredient_purpose" in rec.quality_issues


def test_bamboo_parenthesis_with_mugwort_and_generic_parts_is_withheld():
    parsed = sample("다양한 대나무 추출물(쑥잎, 새싹, 잎)을 함유한 제품을 사용하는 것이 도움이 될 수 있습니다.")
    rec, _ = prepare_recommendation(RecResult(parsed, "test", False, 1, "", 1, 1), inputs(), [],
                                    IngredientExtractor({"판테놀": "판테놀"}), OutputFilter({}))
    assert rec.quality_status == "withheld" and "ingredient_identity_conflict" in rec.quality_issues
