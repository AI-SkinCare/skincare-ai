import collections
import json

import pytest

from src.llm.clean_targets import TargetCleaner
from src.llm.output_filter import OutputFilter

LEX = {"SULFUR": "유황", "ALOESIN": "알로에신", "MOMORDICACHARANTIAFRUITEXTRACT": "여주 열매 추출물",
       "LITCHICHINENSISSEEDPOWDER": "리치 씨앗 가루", "HEXAPEPTIDE-2": "헥사펩타이드-2"}


@pytest.fixture(scope="module")
def cleaner():
    return TargetCleaner(OutputFilter.from_yaml(), LEX)


def _c(cleaner, text):
    return cleaner.clean_text(text, collections.Counter())


@pytest.mark.parametrize("src,want", [
    ("'SULFUR(유황)'는 피지를 줄여 줍니다.", "'유황'은 피지를 줄여 줍니다."),          # 영문(한글) + 조사 교정
    ("'알로에신(Aloesin)'은 톤을 밝혀 줍니다.", "'알로에신'은 톤을 밝혀 줍니다."),       # 한글(영문) → 병기 삭제
    ("'SULFUR'와 'MOMORDICA CHARANTIA FRUIT EXTRACT'가 좋아요.", "'유황'과 '여주 열매 추출물'이 좋아요."),
    ("SULFUR, LITCHI CHINENSIS SEED POWDER를 권해요.", "유황, 리치 씨앗 가루를 권해요."),  # 쉼표로 이어진 영문
    ("'헥사펩타이드-2(Hexapeptide-2)'는 좋아요.", "'헥사펩타이드-2'는 좋아요."),         # 숫자로 끝나면 조사 유지
    ("SPF 30 이상, Tyrosinase 억제", "SPF 30 이상, Tyrosinase 억제"),                   # 사전에 없는 영문은 그대로
])
def test_korean_names_and_josa(cleaner, src, want):
    assert _c(cleaner, src) == want


def test_medical_terms_rewritten_and_claims_softened(cleaner):
    out = _c(cleaner, "피부 질환 치료에 쓰이며 주름을 개선합니다.")
    assert "치료" not in out and "질환" not in out
    assert out == "피부 고민 관리에 쓰이며 주름을 개선하는 데 도움을 줄 수 있습니다."


def test_keeps_line_breaks(cleaner):
    assert _c(cleaner, "추천 성분입니다.\n1. **유황**: 피지 조절") == "추천 성분입니다.\n1. **유황**: 피지 조절"


def test_clean_keeps_json_shape(cleaner):
    t = json.dumps({"reasoning": [{"step": 1, "title": "t", "content": "'SULFUR'를 씁니다."}], "answer": "치료합니다."},
                   ensure_ascii=False)
    out = json.loads(cleaner.clean(t, collections.Counter()))
    assert out["reasoning"][0]["content"] == "'유황'을 씁니다."
    assert out["answer"] == "관리합니다."   # '치료'→'관리'. '관리'는 효능 단정이 아니라 완화 대상 아님
