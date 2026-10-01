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
