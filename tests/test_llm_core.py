import json

import pytest

from src.llm.ingredients import IngredientExtractor, harvest
from src.llm.output_filter import OutputFilter
from src.llm.schema import ExternalFactor, RecInput, build_messages, build_target, parse_output


def _x(**kw):
    d = dict(gender="여성", age=53, skin_type="복합성", concerns=["모공"],
             external_factors=[ExternalFactor("호르몬변화", "생리 영향")])
    d.update(kw)
    return RecInput(**d)


# ── schema ──
def test_prompt_is_deterministic_and_contains_fields():
    a, b = build_messages(_x()), build_messages(_x())
    assert a == b
    u = a[1]["content"]
    assert "53세" in u and "복합성" in u and "모공" in u and "호르몬변화(생리 영향)" in u
    assert "피부 상태 코드" not in u  # 코드 없으면 줄 자체가 없어야 함


def test_prompt_includes_codes_and_note_when_given():
    u = build_messages(_x(condition_codes="NS/P1/W2/NA/VP/SA/ND/NR", user_note="코 모공"))[1]["content"]
    assert "NS/P1/W2" in u and "추가 요청: 코 모공" in u


def test_validate_rejects_unknown_labels():
    with pytest.raises(ValueError):
        _x(concerns=["여드름"]).validate()   # 라벨은 '여드름/뾰루지'
    with pytest.raises(ValueError):
        _x(skin_type="민감성").validate()


def test_target_roundtrip_and_no_evidence():
    t = build_target([{"title": "a", "content": "x"}, {"title": "b", "content": "y"}], "ans")
    assert "evidence" not in t and "PMID" not in t
    assert parse_output(t) == {"reasoning": [{"step": 1, "title": "a", "content": "x"},
                                             {"step": 2, "title": "b", "content": "y"}], "answer": "ans"}


@pytest.mark.parametrize("raw", [
    '```json\n{"reasoning":[{"title":"t","content":"c"}],"answer":"a"}\n```',
    '네, 답변입니다.\n{"reasoning":[{"title":"t","content":"c"}],"answer":"a"} 끝',
])
def test_parse_tolerates_wrapping(raw):
    assert parse_output(raw)["answer"] == "a"


@pytest.mark.parametrize("raw", ["", "그냥 텍스트", '{"reasoning": [], "answer": "a"}',
                                 '{"reasoning":[{"title":"t"}],"answer":"a"}', '{"reasoning":[{"content":"c"}]'])
def test_parse_rejects_bad(raw):
    assert parse_output(raw) is None


# ── ingredients ──
CORPUS = [
    "SULFUR(유황)는 피지를 조절합니다. LITCHI CHINENSIS SEED POWDER(리치씨 파우더)는 각질 제거.",
    "'알로에신(Aloesin)'과 '헥사펩타이드-2(Hexapeptide-2)'는 톤 개선. 자외선 차단을 위해(SPF 30+, PA+++) 바르세요.",
    "리치씨앗가루(LITCHI CHINENSIS SEED POWDER)와 알로에신(ALOESIN) 사용.",
]


def test_harvest_merges_synonyms_and_skips_noise():
    lex, freq = harvest(CORPUS)
    ext = IngredientExtractor(lex)
    assert "위해" not in freq  # 'SPF 30+' 괄호에서 딸려온 일반어 배제
    # 리치씨 파우더 == 리치씨앗가루 (같은 INCI명)
    assert ext.extract("리치씨앗가루") == ext.extract("리치씨 파우더")
    assert ext.extract("LITCHI CHINENSIS SEED POWDER 사용") == ext.extract("리치씨 파우더")


def test_extract_order_and_longest_match():
    ext = IngredientExtractor({"알로에": "알로에", "알로에신": "알로에신", "유황": "유황"})
    assert ext.extract("유황 다음 알로에신") == ["유황", "알로에신"]  # '알로에'가 따로 잡히면 안 됨


def test_extract_drops_non_ingredients_and_merges_aliases():
    ext = IngredientExtractor({"항산화제": "항산화제", "항산화": "항산화", "유황": "유황",
                               "트라넥삼산": "트라넥삼산", "트라넥사믹애씨드": "트라넥사믹애씨드",
                               "큐티박테리움아크네스": "큐티박테리움 아크네스"})
    text = "항산화제와 유황, 트라넥삼산 그리고 트라넥사믹애씨드. 큐티박테리움 아크네스 억제"
    # '항산화제'는 결과에서 빠지되 그 안의 '항산화'도 따로 잡히지 않아야 한다
    assert ext.extract(text) == ["유황", "트라넥사믹애씨드"]


def test_rule_fallback_skips_non_ingredients(tmp_path):
    from src.llm.fallback import RuleBasedRecommender
    from src.llm.schema import RecInput
    p = tmp_path / "top.json"
    p.write_text(json.dumps({"주름": ["항산화제", "자두추출물", "아지렐린", "레티노이드", "리치 씨앗 가루"]},
                            ensure_ascii=False), encoding="utf-8")
    x = RecInput(gender="여성", age=40, skin_type="건성", concerns=["주름"])
    assert RuleBasedRecommender(p).ingredients_for(x) == ["자두추출물", "아세틸헥사펩타이드-8", "리치 씨앗 가루"]


# ── output filter ──
@pytest.fixture(scope="module")
def filt():
    return OutputFilter.from_yaml()


def test_filter_drops_medical_sentences_but_keeps_referral(filt):
    parsed = {"reasoning": [{"step": 1, "title": "t",
                             "content": "아토피 치료에 효과적입니다. 증상이 지속되면 피부과 전문의 상담을 권합니다. 보습이 중요합니다."}],
              "answer": "레티놀은 주름을 개선합니다. PMID:12345678"}
    out, rep = filt.apply(parsed)
    body = out["reasoning"][0]["content"]
    assert "아토피" not in body and "전문의" in body and "보습이 중요합니다" in body
    assert "PMID" not in out["answer"]
    assert "개선하는 데 도움을 줄 수 있습니다" in out["answer"]
    assert rep.changed and out["disclaimer"]


def test_filter_leaves_clean_text_untouched(filt):
    parsed = {"reasoning": [{"step": 1, "title": "t", "content": "아침에는 자외선 차단제를 바르세요."}], "answer": "보습을 꼼꼼히 하세요."}
    out, rep = filt.apply(parsed)
    assert not rep.changed and out["answer"] == "보습을 꼼꼼히 하세요."
