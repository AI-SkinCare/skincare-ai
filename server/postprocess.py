"""모델의 자유 서술 결과 → API 응답 구조(ingredients / routine / reasoning)."""
from __future__ import annotations

import re

from src.llm.ingredients import IngredientExtractor

_SENT = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s*")
_ORDINAL = re.compile(r"^(첫째|둘째|셋째|넷째|다섯째|여섯째|마지막으로|또한)\s*,?\s*")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


# 한 문장에 성분 여러 개가 이어질 때의 경계: "…기여하며, '카나다발삼추출물'은 …"
_CLAUSE = re.compile(r",\s*(?=['‘\"“*]|또한|그리고)")
# 주어 부분: "추천 성분 중 '우엉뿌리추출물'은 " → 제거 (이름 뒤 따옴표·굵게·괄호 병기·조사까지)
_LEAD = r"^.*?{name}['’\"”*]*\s*(?:\([^)]*\))?\s*(?:은|는|이|가|와|과)?\s*,?\s*"
# 절이 연결어미로 끝나면 문장으로 마무리: "기여하며" → "기여합니다."
_ENDINGS = [("있으며", "있습니다."), ("하며", "합니다."), ("주며", "줍니다."), ("되며", "됩니다."), ("이며", "입니다."),
            ("하고", "합니다."), ("주고", "줍니다."), ("되고", "됩니다.")]


def _purpose(sentence: str, name: str) -> str | None:
    """성분이 나온 문장에서 그 성분을 설명하는 절만 골라 주어를 뗀다."""
    clause = next((c for c in _CLAUSE.split(sentence) if name.replace(" ", "") in c.replace(" ", "")), sentence)
    lead = re.compile(_LEAD.format(name=re.escape(name)))
    body = lead.sub("", clause, count=1) if name in clause else clause
    body = body.strip(" ,")
    if body == clause.strip(" ,") and name in body:   # 주어로 쓰이지 않은 문장(나열 등)은 효능 설명이 아님
        return None
    for a, b in _ENDINGS:
        if body.endswith(a):
            body = body[: -len(a)] + b
            break
    if not body.endswith((".", "다")):
        body += "."
    if len(body) < 8:
        return None
    return body if len(body) <= 120 else body[:117] + "…"


def structure_ingredients(parsed: dict, ext: IngredientExtractor | None, limit: int = 5) -> list[dict]:
    step2 = next((s["content"] for s in parsed["reasoning"] if "성분" in s.get("title", "")),
                 parsed["reasoning"][min(1, len(parsed["reasoning"]) - 1)]["content"])
    names = ext.extract(step2 + "\n" + parsed["answer"]) if ext else []
    sents = _sentences(step2)
    out = []
    for i, name in enumerate(names[:limit]):
        key = name.replace(" ", "")
        sent = next((s for s in sents if key in s.replace(" ", "")), None)
        out.append({"name": name, "purpose": _purpose(sent, name) if sent else None, "priority": i + 1})
    return out


def _time_of(s: str) -> str:
    if "아침" in s and "저녁" in s:
        return "daily"
    if "아침" in s or "외출" in s:
        return "morning"
    if "저녁" in s or "밤" in s or "자기 전" in s:
        return "evening"
    if re.search(r"주\s*\d", s):
        return "weekly"
    if "매일" in s:
        return "daily"
    return "anytime"


# "효과적인 모공 관리를 위해서는 꾸준하고 체계적인 루틴이 필요합니다." 같은 도입·총평 문장은 행동 단계가 아니다
_GENERIC_END = re.compile(r"(필요|중요)합니다\.?$")
_ACTION = re.compile(r"바르|발라|사용|세안|클렌|차단|도포|마사지|각질|팩|섭취|수면|보습|토너|세럼|크림|앰플|에센스|흡수|두드|헹구|피하")


def structure_routine(parsed: dict, limit: int = 6) -> list[dict]:
    step3 = next((s["content"] for s in parsed["reasoning"] if "사용법" in s.get("title", "") or "루틴" in s.get("title", "")),
                 parsed["reasoning"][-1]["content"])
    steps = []
    for s in _sentences(step3):
        s = _ORDINAL.sub("", s)
        if len(s) < 8 or (_GENERIC_END.search(s) and not _ACTION.search(s)):
            continue
        steps.append({"step": len(steps) + 1, "time": _time_of(s), "action": s})
        if len(steps) >= limit:
            break
    return steps
