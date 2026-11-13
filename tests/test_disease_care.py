import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from server.disease_care import applicable_care, class_care, review_care
from server.grounded_contract import recovery_prompt, source_payload
from server.grounded_generation import assess, generate_grounded
from server.ingredient_evidence import load_evidence, retrieve_evidence, extend_extractor, enforce_evidence
from server.product_conditions import routine_lines, prompt_usage
from server.schemas import DISEASE_LABELS, DiseasePrediction
from src.llm.inference import RecResult
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter
from src.llm.schema import RecInput


def fixture(label="atopic", sunscreen_only=False, avoid=()):
    store = load_evidence()
    x = RecInput(gender="여성", age=30, skin_type="중성", concerns=[],
                 user_note="보습제와 선크림의 사용 조건을 알려주세요.")
    disease = DiseasePrediction(
        label=DISEASE_LABELS[label], label_en=label, confidence=.8, uncertain=False,
        probabilities={name: .8 if key == label else .04 for key, name in DISEASE_LABELS.items()},
    )
    bundle = retrieve_evidence(store, x, disease, avoid)
    cards = [card for card in bundle["claims"] if card["id"] in ("E_GLY_MOIST", "E_ZN_SUN")
             and (not sunscreen_only or card["id"].endswith("_SUN"))]
    response = SimpleNamespace(ingredients=[SimpleNamespace(name=card["ingredient"]) for card in cards])
    extractor = extend_extractor(IngredientExtractor({}), store)
    return x, disease, bundle, cards, response, extractor


def result(payload):
    return RecResult(payload, "fixture", False, 1, json.dumps(payload, ensure_ascii=False), 10, 1)


@pytest.mark.parametrize("label,source", [
    ("atopic", "AAD_AD"), ("psoriasis", "AAD_PS"),
    ("rosacea", "AAD_ROS"), ("seborrheic", "AAD_SD"),
])
def test_missing_class_care_is_blocked_and_recovery_carries_its_actual_source(label, source):
    x, disease, bundle, cards, response, extractor = fixture(label)
    missing = source_payload(cards)
    original = deepcopy(missing)
    review = review_care(missing, response, bundle)
    assert review["issues"] == ["care_condition_missing_" + rule["id"] for rule in class_care(label)]
    assert missing == original
    assert review["bindings"][0]["source_id"] == source
    assert review["bindings"][0]["source"] == bundle["sources"][source]
    served, _, full_review = assess(result(missing), x, [], extractor, OutputFilter.from_yaml(), disease, bundle)
    assert enforce_evidence(served, full_review).evidence_status == "withheld"

    complete = source_payload(cards, bundle)
    served, _, full_review = assess(result(complete), x, [], extractor, OutputFilter.from_yaml(), disease, bundle)
    assert not full_review["issues"] and served.quality_status == "ok"
    assert enforce_evidence(served, full_review).evidence_status == "source_linked"
    condition = class_care(label)[0]["text"]
    assert condition in served.reasoning[2]
    assert condition in " ".join(step.action for step in served.routine)
    for text in (routine_lines(bundle, [c["ingredient"] for c in cards]),
                 prompt_usage(bundle), recovery_prompt(cards, bundle)):
        assert condition in text
    assert all(ingredient.purpose for ingredient in served.ingredients)
    assert not full_review["product_conditions"]["disease_care"]["clinical_entailment_verified"]


@pytest.mark.parametrize("label", ["atopic", "psoriasis", "rosacea", "seborrheic"])
def test_sunscreen_only_does_not_receive_moisturizer_timing(label):
    _, _, bundle, cards, response, _ = fixture(label, sunscreen_only=True)
    payload = source_payload(cards, bundle)
    expected = [rule for rule in class_care(label) if rule["product"] == "all"]
    assert applicable_care(bundle, cards) == expected
    assert not review_care(payload, response, bundle)["issues"]
    assert class_care(label)[0]["text"] not in payload["reasoning"][2]["content"]
    for rule in expected:
        assert rule["text"] in payload["reasoning"][2]["content"]
        assert rule["text"] in routine_lines(bundle, [card["ingredient"] for card in cards])


@pytest.mark.parametrize("label", ["acne", "normal"])
def test_unsupported_class_care_is_never_added(label):
    _, _, bundle, cards, response, _ = fixture(label)
    assert not class_care(label)
    assert source_payload(cards, bundle) == source_payload(cards)
    assert not review_care(source_payload(cards), response, bundle)["issues"]


@pytest.mark.parametrize("label,opposite", [
    ("atopic", "보습제는 피부가 마른 다음 사용하세요."),
    ("psoriasis", "보습제는 물기를 완전히 말린 후 사용하세요."),
    ("rosacea", "보습제는 수분이 남아 있을 때 사용하세요."),
    ("rosacea", "보습제는 피부가 마르기 전에 사용하세요."),
    ("seborrheic", "보습제는 알코올이 함유된 제품을 선택하세요."),
    ("seborrheic", "보습제는 알코올 없는 제품을 쓸 필요 없습니다."),
])
@pytest.mark.parametrize("field", ["routine", "answer"])
def test_copying_correct_text_does_not_excuse_explicit_opposite_advice(label, opposite, field):
    _, _, bundle, cards, response, _ = fixture(label)
    payload = source_payload(cards, bundle)
    if field == "routine":
        payload["reasoning"][2]["content"] += "\n" + opposite
    else:
        payload["answer"] += "\n" + opposite
    review = review_care(payload, response, bundle)
    assert review["covered"] == review["required"]
    assert any(issue.startswith("care_condition_contradicted_") for issue in review["issues"])


def test_no_new_claim_is_created_without_the_source_text_and_source_entry():
    _, _, bundle, cards, _, _ = fixture()
    assert applicable_care({**bundle, "care_conditions": []}, cards) == []
    assert applicable_care({**bundle, "sources": {}}, cards) == []


def test_care_only_failure_causes_recorded_model_repair_without_patching_the_raw_answer():
    x, disease, bundle, cards, _, extractor = fixture()
    bad = result(source_payload(cards))
    good = result(source_payload(cards, bundle))
    calls = []

    class Model:
        def recommend_with_context(self, x, context, **kwargs):
            calls.append(context)
            return bad if len(calls) == 1 else good

    generated, served, _, review, trace = generate_grounded(
        Model(), x, [], extractor, OutputFilter.from_yaml(), disease, bundle,
    )
    assert [step["phase"] for step in trace] == ["source", "product_repair"]
    assert trace[0]["result"]["parsed"] == bad.parsed
    assert generated.raw_text == good.raw_text and generated.attempts == 2
    assert served.generation_origin == "model" and not served.used_fallback
    assert not review["issues"]
    assert class_care("atopic")[0]["text"] in calls[1]


def test_repeated_model_failure_has_complete_source_care_and_explicit_fallback_provenance():
    x, disease, bundle, cards, _, extractor = fixture("seborrheic")
    bad = result(source_payload(cards))

    class Model:
        def recommend_with_context(self, *args, **kwargs):
            return bad

    generated, served, _, review, trace = generate_grounded(
        Model(), x, [], extractor, OutputFilter.from_yaml(), disease, bundle,
    )
    assert generated.raw_text == bad.raw_text and generated.parsed == bad.parsed
    assert served.generation_origin == "source_cards" and served.used_fallback
    assert all(ingredient.purpose_source == "source_card" for ingredient in served.ingredients)
    assert not review["issues"] and trace[-1]["review"]["issues"]
    assert class_care("seborrheic")[0]["text"] in served.reasoning[2]
    assert enforce_evidence(served, review).evidence_status == "source_linked"
    assert set(served.source_recovery["source_ids"]) <= {card["id"] for card in bundle["claims"]}


def test_avoidance_is_preserved_by_class_care_recovery():
    _, _, bundle, _, _, _ = fixture("atopic", avoid=("글리세린", "세라마이드"))
    cards = bundle["claims"]
    assert all(card["ingredient"] not in ("글리세린", "세라마이드") for card in cards)
    payload = source_payload(cards, bundle)
    assert "글리세린" not in json.dumps(payload, ensure_ascii=False)
    assert "세라마이드" not in json.dumps(payload, ensure_ascii=False)
    assert class_care("atopic")[0]["text"] in payload["reasoning"][2]["content"]


def test_seborrheic_sunscreen_alcohol_cannot_be_excused_by_correct_moisturizer_care():
    _, _, bundle, cards, response, _ = fixture("seborrheic")
    payload = source_payload(cards, bundle)
    payload["reasoning"][2]["content"] += "\n선크림은 알코올이 함유된 완제품을 선택하세요."
    assert "care_condition_contradicted_all_products_alcohol_free" in review_care(payload, response, bundle)["issues"]


def test_care_must_survive_the_final_routine_length_limit():
    x, disease, bundle, cards, _, extractor = fixture()
    payload = source_payload(cards, bundle)
    condition = class_care("atopic")[0]["text"]
    payload["reasoning"][2]["content"] = "무향의 보습제를 제품 표시대로 사용한다. " * 7 + condition
    served, _, review = assess(result(payload), x, [], extractor, OutputFilter.from_yaml(), disease, bundle)
    assert condition in payload["reasoning"][2]["content"]
    assert condition not in " ".join(step.action for step in served.routine)
    assert "care_condition_not_delivered_moisturizer_damp_after_wash" in review["issues"]
    assert enforce_evidence(served, review).evidence_status == "withheld"


def test_care_must_survive_display_filtering():
    x, disease, bundle, cards, _, extractor = fixture()
    payload = source_payload(cards, bundle)
    condition = class_care("atopic")[0]["text"]
    output_filter = OutputFilter({"drop_sentence_terms": [condition]})
    served, _, review = assess(result(payload), x, [], extractor, output_filter, disease, bundle)
    assert condition not in " ".join(served.reasoning)
    assert "care_condition_not_delivered_moisturizer_damp_after_wash" in review["issues"]


def test_internal_source_marker_is_hidden_while_care_and_provenance_are_preserved():
    x, disease, bundle, cards, _, extractor = fixture("psoriasis")
    payload = source_payload(cards, bundle)
    payload["reasoning"][2]["content"] += " (AAD_PS)"
    raw = result(payload)
    served, _, review = assess(raw, x, [], extractor, OutputFilter.from_yaml(), disease, bundle)
    assert "(AAD_PS)" not in " ".join(served.reasoning)
    assert "(AAD_PS)" not in " ".join(step.action for step in served.routine)
    assert "(AAD_PS)" in raw.raw_text and "(AAD_PS)" in raw.parsed["reasoning"][2]["content"]
    assert class_care("psoriasis")[0]["text"] in served.reasoning[2]
    assert review["product_conditions"]["disease_care"]["bindings"][0]["source_id"] == "AAD_PS"
    assert not review["issues"]


def test_valid_purposes_cannot_excuse_a_mixed_sunscreen_benefit_in_the_summary():
    x, disease, bundle, cards, _, extractor = fixture("acne")
    payload = source_payload(cards, bundle)
    payload["answer"] = "글리세린 함유 보습제와 징크옥사이드 함유 완제품 선크림을 사용하여 피부 수분 유지와 건조함 관리를 돕습니다."
    served, _, review = assess(result(payload), x, [], extractor, OutputFilter.from_yaml(), disease, bundle)
    assert all(ingredient.purpose for ingredient in served.ingredients)
    assert "evidence_sunscreen_moisturizer_claim_mixed" in review["issues"]
    assert enforce_evidence(served, review).evidence_status == "withheld"


@pytest.mark.parametrize("answer", [
    "글리세린 함유 보습제와 징크옥사이드 함유 선크림을 선택합니다.",
    "글리세린 함유 보습제는 수분 유지에, 징크옥사이드 함유 선크림은 자외선 차단에 사용합니다.",
    "징크옥사이드 함유 선크림을 선택하고, 글리세린 함유 보습제는 수분 유지에 사용합니다.",
])
def test_separate_moisturizer_and_sunscreen_aims_are_preserved(answer):
    x, disease, bundle, cards, _, extractor = fixture("acne")
    payload = source_payload(cards, bundle)
    payload["answer"] = answer
    _, _, review = assess(result(payload), x, [], extractor, OutputFilter.from_yaml(), disease, bundle)
    assert "evidence_sunscreen_moisturizer_claim_mixed" not in review["issues"]


def test_mixed_summary_triggers_a_real_repair_call_instead_of_a_server_rewrite():
    x, disease, bundle, cards, _, extractor = fixture("acne")
    good = result(source_payload(cards, bundle))
    payload = deepcopy(good.parsed)
    payload["answer"] = "글리세린 함유 보습제와 징크옥사이드 함유 선크림으로 피부 수분 유지와 건조함 관리를 돕습니다."
    bad = result(payload)
    calls = []

    class Model:
        def recommend_with_context(self, actual_input, context, **kwargs):
            calls.append(context)
            return bad if len(calls) == 1 else good

    generated, served, _, review, trace = generate_grounded(
        Model(), x, [], extractor, OutputFilter.from_yaml(), disease, bundle,
    )
    assert len(calls) == 2 and generated.raw_text == good.raw_text
    assert trace[0]["result"]["parsed"]["answer"] == bad.parsed["answer"]
    assert "evidence_sunscreen_moisturizer_claim_mixed" in trace[0]["review"]["issues"]
    assert not review["issues"] and served.answer == good.parsed["answer"]
    assert served.generation_origin == "model" and not served.used_fallback
