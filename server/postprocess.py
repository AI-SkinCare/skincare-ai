"""모델의 자유 서술 결과 → API 응답 구조(ingredients / routine / reasoning)."""
from __future__ import annotations

import re

from src.llm.ingredients import IngredientExtractor

_SENT = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s*")
_ORDINAL = re.compile(r"^(첫째|둘째|셋째|넷째|다섯째|여섯째|마지막으로|또한)\s*,?\s*")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def structure_ingredients(parsed: dict, ext: IngredientExtractor | None, limit: int = 5) -> list[dict]:
    step2 = next((s["content"] for s in parsed["reasoning"] if "성분" in s.get("title", "")),
                 parsed["reasoning"][min(1, len(parsed["reasoning"]) - 1)]["content"])
    names = ext.extract(step2 + "\n" + parsed["answer"]) if ext else []
    sents = _sentences(step2)
    out = []
    for i, name in enumerate(names[:limit]):
        key = name.replace(" ", "")
        purpose = next((s for s in sents if key in s.replace(" ", "")), None)
        if purpose and len(purpose) > 120:
            purpose = purpose[:117] + "…"
        out.append({"name": name, "purpose": purpose, "priority": i + 1})
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


def structure_routine(parsed: dict, limit: int = 6) -> list[dict]:
    step3 = next((s["content"] for s in parsed["reasoning"] if "사용법" in s.get("title", "") or "루틴" in s.get("title", "")),
                 parsed["reasoning"][-1]["content"])
    steps = []
    for s in _sentences(step3):
        s = _ORDINAL.sub("", s)
        if len(s) < 8:
            continue
        steps.append({"step": len(steps) + 1, "time": _time_of(s), "action": s})
        if len(steps) >= limit:
            break
    return steps
