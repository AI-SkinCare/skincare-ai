"""모델 자유 서술 → ingredients / routine 구조화. 입력 문장은 v2 모델의 실제 출력(2026-10-02)에서 가져왔다."""
from server.postprocess import _purpose, structure_ingredients, structure_routine
from src.llm.ingredients import IngredientExtractor

STEP2 = ("모공 관리를 위해 피지 조절, 항염, 피부 탄력 개선, 그리고 피부 진정 효과를 가진 성분들을 복합적으로 활용하는 것이 중요합니다. "
         "추천 성분 중 '우엉뿌리추출물'은 강력한 피지 조절 및 모공 수렴 효과로 과도한 피지 분비를 억제하고 모공을 타이트하게 관리하는 데 도움을 줍니다. "
         "'서양톱풀오일'은 항염 및 수렴 작용으로 피부 진정과 함께 모공 관리에 기여하며, '카나다발삼추출물'은 항산화 및 항염 작용으로 피부를 보호하고 전반적인 피부결 개선에 도움을 줍니다. "
         "또한, '알부민추출물'은 일시적인 피부 탄력 증진 및 모공 수축 효과를 제공하여 모공이 덜 도드라져 보이도록 돕습니다.")
STEP3 = ("효과적인 모공 관리를 위해서는 꾸준하고 체계적인 스킨케어 루틴이 필요합니다. "
         "첫째, 부드러운 클렌징으로 모공 속 노폐물과 과도한 피지를 깨끗하게 제거하되, 피부 장벽을 손상시키지 않도록 약산성 클렌저를 사용하는 것이 좋습니다. "
         "둘째, 주 1~2회 정도 각질 제거를 통해 모공 막힘을 예방하고 피부 턴오버를 촉진합니다. "
         "넷째, 자외선 차단제는 필수이며, 모공 확장을 유발할 수 있는 자외선으로부터 피부를 보호해야 합니다.")
PARSED = {"reasoning": [{"title": "문제 분석 및 원인 파악", "content": "원인."},
                        {"title": "성분 선택 및 근거 제시", "content": STEP2},
                        {"title": "사용법 및 관리방안", "content": STEP3}],
          "answer": "'우엉뿌리추출물', '서양톱풀오일', '카나다발삼추출물', '알부민추출물'을 권합니다."}
EXT = IngredientExtractor({k: k for k in ["우엉뿌리추출물", "서양톱풀오일", "카나다발삼추출물", "알부민추출물"]})


def test_purpose_is_own_clause_without_subject():
    ing = {i["name"]: i["purpose"] for i in structure_ingredients(PARSED, EXT)}
    assert ing["우엉뿌리추출물"] == "강력한 피지 조절 및 모공 수렴 효과로 과도한 피지 분비를 억제하고 모공을 타이트하게 관리하는 데 도움을 줍니다."
    # 한 문장에 두 성분: 각자 자기 절만, 연결어미는 문장으로 마무리
    assert ing["서양톱풀오일"] == "항염 및 수렴 작용으로 피부 진정과 함께 모공 관리에 기여합니다."
    assert ing["카나다발삼추출물"] == "항산화 및 항염 작용으로 피부를 보호하고 전반적인 피부결 개선에 도움을 줍니다."
    assert ing["알부민추출물"].startswith("일시적인 피부 탄력 증진")


def test_purpose_none_when_only_listed():
    assert _purpose("추천 성분은 우엉뿌리추출물, 서양톱풀오일입니다.", "서양톱풀오일") is None


def test_routine_skips_generic_intro():
    r = structure_routine(PARSED)
    assert [s["step"] for s in r] == [1, 2, 3]
    assert r[0]["action"].startswith("부드러운 클렌징")          # 도입 문장·서수 제거
    assert r[1]["time"] == "weekly"
    assert r[2]["action"].startswith("자외선 차단제")


# Saved v2 failures from the 96-image baseline; these are model claims, not
# an approved efficacy reference.
import pytest


def _parsed(step2, answer="", routine=""):
    return {"reasoning": [{"title": "성분 선택 및 근거 제시", "content": step2},
                          {"title": "사용법 및 관리방안", "content": routine}],
            "answer": answer}


def _ingredients(step2, answer="", ext=None, routine=""):
    return {row["name"]: row for row in structure_ingredients(
        _parsed(step2, answer, routine), ext or EXT, limit=20)}


BAMBOO_EXT = IngredientExtractor({
    "쑥잎추출물": "쑥잎 추출물",
    "ARTEMISIA ARGYI LEAF EXTRACT": "쑥잎 추출물",
    "용담": "용담",
    "BAMBUSA VULGARIS EXTRACT": "대나무추출물",
    "BAMBUSA VULGARIS LEAF EXTRACT": "대나무 잎 추출물",
    "BAMBUSA VULGARIS SHOOT EXTRACT": "밤부사 불가리스 싹 추출물",
})


def test_baseline_listing_does_not_mask_later_explanation_or_create_fragment():
    text = ("붉어짐과 민감성 피부 진정을 위해 '쑥잎 추출물'과 다양한 'BAMBUSA VULGARIS' "
            "추출물(쑥, 대나무, 용담) 성분들을 추천합니다. "
            "쑥잎 추출물은 강력한 항염 및 항산화 효과로 붉어진 피부를 신속하게 진정시키고 "
            "외부 자극으로부터 피부를 보호하는 데 탁월합니다. "
            "대나무 추출물들은 뛰어난 보습 및 진정 효과를 제공하며, 피부에 수분을 공급하고 "
            "탄력을 개선하여 민감성 피부 장벽 강화에 기여합니다.")
    got = _ingredients(text, ext=BAMBOO_EXT)
    assert got["쑥잎 추출물"]["purpose"].startswith("강력한 항염")
    assert got["쑥잎 추출물"]["purpose_source"] == "model"
    assert got["용담"]["purpose"] is None
    assert got["용담"]["purpose_status"] == "unavailable"
    assert got["용담"]["purpose_source"] == "unavailable"
    assert got["대나무추출물"]["purpose"].startswith("뛰어난 보습")


@pytest.mark.parametrize("group", [
    "'대나무추출물', '대나무 잎 추출물', '대나무추출물', '밤부사 불가리스 싹 추출물' 등 대나무 유래 성분들은",
    "'대나무추출물' 및 '대나무 잎 추출물', '대나무추출물', '밤부사 불가리스 싹 추출물'과 같은 대나무 유래 성분들은",
])
def test_baseline_explicit_group_gets_shared_purpose_without_list_fragments(group):
    purpose = "뛰어난 보습 및 진정 효과를 제공하여 피부 장벽을 강화하고 수분 공급을 돕습니다."
    got = _ingredients("쑥잎 추출물을 권합니다. 둘째, " + group + " " + purpose, ext=BAMBOO_EXT)
    assert got["쑥잎 추출물"]["purpose"] is None
    for name in ("대나무추출물", "대나무 잎 추출물", "밤부사 불가리스 싹 추출물"):
        assert got[name]["purpose"] == purpose
        assert got[name]["purpose_source"] == "model_shared"
        assert got[name]["purpose_status"] == "provided"


@pytest.mark.parametrize("name", [
    "'bambusa vulgaris leaf extract'",
    "“BAMBUSA  VULGARIS\tLEAF EXTRACT”",
    "**대나무 잎 추출물**",
    "대나무잎추출물(BAMBUSA VULGARIS LEAF EXTRACT)",
    "BAMBUSA VULGARIS LEAF EXTRACT(대나무 잎 추출물)",
])
def test_aliases_and_bilingual_spans_retain_original_purpose(name):
    got = _ingredients(name + "은 피부 진정에 도움을 줍니다.", ext=BAMBOO_EXT)
    assert list(got) == ["대나무 잎 추출물"]
    assert got["대나무 잎 추출물"]["purpose"] == "피부 진정에 도움을 줍니다."
    assert got["대나무 잎 추출물"]["purpose_source"] == "model"


def test_english_explanation_and_alias():
    got = _ingredients("BAMBUSA VULGARIS EXTRACT helps maintain moisture.", ext=BAMBOO_EXT)
    assert got["대나무추출물"]["purpose"] == "helps maintain moisture."


def test_distinct_parts_and_longest_alias_never_inherit_each_others_claims():
    got = _ingredients(
        "BAMBUSA VULGARIS EXTRACT는 보습에 도움을 줍니다. "
        "대나무잎추출물은 피부 진정에 도움을 줍니다. "
        "BAMBUSA VULGARIS SHOOT EXTRACT를 권합니다.", ext=BAMBOO_EXT)
    assert got["대나무추출물"]["purpose"] == "보습에 도움을 줍니다."
    assert got["대나무 잎 추출물"]["purpose"] == "피부 진정에 도움을 줍니다."
    assert got["밤부사 불가리스 싹 추출물"]["purpose"] is None


def test_overlapping_alias_does_not_match_prefix_or_suffix():
    ext = IngredientExtractor({"ALPHA": "알파오일", "ALPHA LEAF EXTRACT": "알파잎추출물"})
    got = _ingredients("ALPHA LEAF EXTRACT는 피부 보습에 도움을 줍니다.", ext=ext)
    assert list(got) == ["알파잎추출물"]


@pytest.mark.parametrize("text", [
    "'우엉뿌리추출물', '서양톱풀오일'을 추천합니다.",
    "추천 성분은 우엉뿌리추출물, 서양톱풀오일입니다.",
    "우엉뿌리추출물 및 서양톱풀오일을 권합니다.",
    "우엉뿌리추출물은 추천 성분입니다.",
])
def test_list_only_has_no_invented_purpose(text):
    assert all(r["purpose"] is None and r["purpose_source"] == "unavailable"
               for r in _ingredients(text).values())


def test_unbound_group_pronoun_is_not_assigned_to_preceding_list():
    got = _ingredients("우엉뿌리추출물, 서양톱풀오일을 권합니다. "
                       "이 성분들은 피부 보습에 도움을 줍니다.")
    assert all(r["purpose"] is None for r in got.values())


def test_later_individual_explanation_overrides_shared_explanation():
    got = _ingredients("우엉뿌리추출물과 서양톱풀오일은 피부 보습에 도움을 줍니다. "
                       "우엉뿌리추출물은 피부 진정에 도움을 줍니다.")
    assert got["우엉뿌리추출물"]["purpose"] == "피부 진정에 도움을 줍니다."
    assert got["우엉뿌리추출물"]["purpose_source"] == "model"
    assert got["서양톱풀오일"]["purpose_source"] == "model_shared"


def test_unquoted_separate_clauses_are_not_a_shared_group():
    got = _ingredients("우엉뿌리추출물은 피부 보습에 기여하며, "
                       "서양톱풀오일은 피부 진정에 도움을 줍니다.")
    assert got["우엉뿌리추출물"]["purpose"] == "피부 보습에 기여합니다."
    assert got["서양톱풀오일"]["purpose"] == "피부 진정에 도움을 줍니다."
    assert all(r["purpose_source"] == "model" for r in got.values())


def test_answer_only_explanation_and_no_mid_sentence_clipping():
    purpose = ("피부에 수분을 공급하고 건조한 환경에서 피부를 보호하는 데 도움을 주며, "
               "세안 후 수분을 유지하는 데 기여하고 외부 환경으로 인해 건조해진 피부의 보습을 돕고, "
               "일상적인 관리 과정에서 피부의 수분 손실을 줄이고 피부를 부드럽게 유지하는 데 도움을 줍니다.")
    assert len(purpose) > 120
    got = _ingredients("우엉뿌리추출물을 권합니다.", "우엉뿌리추출물은 " + purpose)
    assert got["우엉뿌리추출물"]["purpose"] == purpose


@pytest.mark.parametrize("negative", [
    "우엉뿌리추출물은 추천하지 않습니다.",
    "우엉뿌리추출물을 사용하지 마세요.",
    "우엉뿌리추출물은 피하세요.",
    "우엉뿌리추출물의 사용을 피하는 것이 좋습니다.",
    "피해야 할 성분은 우엉뿌리추출물입니다.",
    "추천하지 않는 성분: 우엉뿌리추출물.",
])
def test_explicit_negative_wins_over_positive_claim(negative):
    got = _ingredients("우엉뿌리추출물은 피부 보습에 도움을 줍니다. " + negative
                       + " 서양톱풀오일은 피부 진정에 도움을 줍니다.")
    assert "우엉뿌리추출물" not in got
    assert got["서양톱풀오일"]["priority"] == 1


@pytest.mark.parametrize("text", [
    "BAMBUSA VULGARIS EXTRACT is not recommended.",
    "Avoid BAMBUSA VULGARIS EXTRACT.",
    "Do not use BAMBUSA VULGARIS EXTRACT.",
])
def test_english_negated_alias_is_excluded(text):
    assert _ingredients(text, ext=BAMBOO_EXT) == {}


def test_negated_group_only_excludes_explicit_members():
    got = _ingredients("우엉뿌리추출물과 서양톱풀오일은 사용하지 마세요. "
                       "카나다발삼추출물은 피부 보습에 도움을 줍니다.")
    assert list(got) == ["카나다발삼추출물"]


def test_negative_instruction_in_routine_excludes_ingredient():
    assert _ingredients("우엉뿌리추출물은 피부 보습에 도움을 줍니다.",
                        routine="우엉뿌리추출물을 사용하지 마세요.") == {}


def test_negative_effect_word_does_not_mean_negative_recommendation():
    got = _ingredients("우엉뿌리추출물은 피지 분비를 억제하고 피부 자극을 줄이는 데 도움을 줍니다.")
    assert got["우엉뿌리추출물"]["purpose_status"] == "provided"


def test_no_extractor_empty_steps_and_limit():
    assert structure_ingredients(_parsed("우엉뿌리추출물을 권합니다."), None) == []
    assert structure_ingredients({"reasoning": [], "answer": ""}, EXT) == []
    assert structure_ingredients(PARSED, EXT, limit=0) == []
    assert len(structure_ingredients(PARSED, EXT, limit=1)) == 1


def test_routine_drops_descriptive_filler_and_preserves_actions_and_negatives():
    text = ("피부 장벽은 외부 환경으로부터 피부를 보호하는 중요한 역할을 합니다. "
            "이 성분의 보습 효과는 건강한 피부 관리에 도움이 될 수 있습니다. 피부 장벽은 피부의 수분을 유지합니다. "
            "꾸준한 관리에는 관심과 시간이 필요합니다. "
            "아침에는 보습 크림을 충분히 바르세요. "
            "스크럽이나 자극이 강한 각질 제거는 피하세요. "
            "피부를 강하게 문지르지 마세요.")
    got = structure_routine(_parsed("", routine=text))
    assert [r["action"] for r in got] == [
        "아침에는 보습 크림을 충분히 바르세요.",
        "스크럽이나 자극이 강한 각질 제거는 피하세요.",
        "피부를 강하게 문지르지 마세요.",
    ]
    assert [r["step"] for r in got] == [1, 2, 3]


def test_routine_can_be_empty_without_real_instruction():
    assert structure_routine(_parsed("", routine="충분한 보습은 피부 관리에 매우 중요한 역할을 합니다.")) == []



@pytest.mark.parametrize("recap", [
    "과 같은 대나무 추출물들은",
    "과 같은 '대나무추출물'들은",
    "등 BAMBUSA VULGARIS EXTRACT 성분들은",
])
def test_explicit_group_recap_repeated_canonical_name_stays_shared(recap):
    # Real v2_final seborrheic_basic shape, including distinct plant parts.
    names = ("'대나무추출물', '대나무 잎 추출물', '대나무 잎/줄기 추출물', "
             "'밤부사 불가리스 싹 추출물'")
    purpose = "풍부한 수분 공급과 함께 피부 진정 및 보호막 강화에 기여합니다."
    got = _ingredients(names + recap + " " + purpose, ext=BAMBOO_EXT)
    assert list(got) == ["대나무추출물", "대나무 잎 추출물",
                         "대나무잎/줄기추출물", "밤부사 불가리스 싹 추출물"]
    assert all(row["purpose"] == purpose for row in got.values())
    assert all(row["purpose_source"] == "model_shared" for row in got.values())
    assert all(row["purpose_status"] == "provided" for row in got.values())


def test_unlisted_recap_name_does_not_inherit_group_explanation():
    got = _ingredients("'대나무추출물', '대나무 잎 추출물'과 같은 "
                       "'밤부사 불가리스 싹 추출물'은 피부 보습에 도움을 줍니다.",
                       ext=BAMBOO_EXT)
    assert got["대나무추출물"]["purpose"] is None
    assert got["대나무 잎 추출물"]["purpose"] is None
    assert got["밤부사 불가리스 싹 추출물"]["purpose_source"] == "model"


def test_single_canonical_name_recap_is_not_shared():
    got = _ingredients("'대나무추출물', 'BAMBUSA VULGARIS EXTRACT'과 같은 "
                       "대나무추출물들은 피부 보습에 도움을 줍니다.", ext=BAMBOO_EXT)
    assert list(got) == ["대나무추출물"]
    assert got["대나무추출물"]["purpose_source"] == "model"


def test_negated_recap_excludes_all_explicit_members():
    got = _ingredients("'대나무추출물', '대나무 잎 추출물'과 같은 "
                       "대나무추출물들은 사용하지 마세요.", ext=BAMBOO_EXT)
    assert got == {}


@pytest.mark.parametrize("body", [
    "도움이 될 수 있는 선택입니다.",
    "피부 관리에 도움이 될 수 있는 선택입니다.",
    "피부 관리에 기여하는 효과적인 선택입니다.",
    "피부에 도움이 됩니다.",
    "is a helpful and effective choice.",
])
def test_generic_endorsement_without_benefit_is_unavailable(body):
    # Real v2 acne_dry_sensitive failure: presence of "도움" is insufficient.
    got = _ingredients("우엉뿌리추출물은 " + body)["우엉뿌리추출물"]
    assert got["purpose"] is None
    assert got["purpose_source"] == "unavailable"
    assert got["purpose_status"] == "unavailable"


def test_generic_first_endorsement_does_not_mask_later_specific_benefit():
    got = _ingredients("우엉뿌리추출물은 도움이 될 수 있는 선택입니다. "
                       "우엉뿌리추출물은 피부 보습에 도움을 줍니다.")
    assert got["우엉뿌리추출물"]["purpose"] == "피부 보습에 도움을 줍니다."
    assert got["우엉뿌리추출물"]["purpose_source"] == "model"


def test_explicit_benefit_in_choice_sentence_is_preserved():
    got = _ingredients("우엉뿌리추출물은 피부 보습에 도움이 될 수 있는 선택입니다.")
    assert got["우엉뿌리추출물"]["purpose"] == "피부 보습에 도움이 될 수 있는 선택입니다."



def test_real_seborrheic_or_routine_does_not_override_shared_purpose():
    group = ("'대나무추출물', '대나무 잎 추출물', '대나무 잎/줄기 추출물', "
             "'밤부사 불가리스 싹 추출물'과 같은 대나무 추출물들은 ")
    purpose = ("풍부한 수분 공급과 함께 피부 진정 및 항산화 작용을 통해 건조하고 민감한 피부에 "
               "깊은 보습과 영양을 제공하며, 피부 장벽 강화에 기여하여 추가적인 자극을 예방합니다.")
    routine = ("특히, 쑥잎 추출물이나 대나무 추출물이 포함된 제품을 사용하여 "
               "피부를 진정시키고 수분을 공급해야 합니다.")
    parsed = _parsed(group + purpose, routine=routine)
    rows = structure_ingredients(parsed, BAMBOO_EXT, include_routine=True)
    got = {row["name"]: row for row in rows}
    assert len(got) == 5
    for name in ("대나무추출물", "대나무 잎 추출물",
                 "대나무잎/줄기추출물", "밤부사 불가리스 싹 추출물"):
        assert got[name]["purpose"] == purpose
        assert got[name]["purpose_source"] == "model_shared"
    assert got["쑥잎 추출물"]["purpose"] is None
    assert got["쑥잎 추출물"]["purpose_source"] == "unavailable"
    assert structure_routine(parsed)[0]["action"] == routine


@pytest.mark.parametrize("modifier,product", [
    ("포함된", "제품"), ("함유된", "크림"), ("배합된", "화장품"),
])
def test_product_relative_clause_does_not_assign_or_ingredients_a_purpose(modifier, product):
    parsed = _parsed("", routine=(
        f"쑥잎 추출물이나 대나무 추출물이 {modifier} {product}을 사용하여 "
        "피부를 진정시키고 수분을 공급해야 합니다."))
    got = structure_ingredients(parsed, BAMBOO_EXT, include_routine=True)
    assert {row["name"] for row in got} == {"쑥잎 추출물", "대나무추출물"}
    assert all(row["purpose"] is None and row["purpose_source"] == "unavailable"
               and row["purpose_status"] == "unavailable" for row in got)


def test_product_clause_in_answer_does_not_mask_individual_explanation():
    got = _ingredients(
        "대나무추출물은 피부 보습에 도움을 줍니다.",
        "대나무추출물이 포함된 제품을 사용하여 피부를 진정시키세요.",
        ext=BAMBOO_EXT)
    assert got["대나무추출물"]["purpose"] == "피부 보습에 도움을 줍니다."
    assert got["대나무추출물"]["purpose_source"] == "model"
