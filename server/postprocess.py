"""모델의 자유 서술 결과 → API 응답 구조(ingredients / routine / reasoning)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.llm.ingredients import IngredientExtractor, canonical

_SENT = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s*")
_ORDINAL = re.compile(r"^(첫째|둘째|셋째|넷째|다섯째|여섯째|마지막으로|또한)\s*,?\s*")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


# These are attribution/grammar rules, not an ingredient efficacy catalogue.
_MARKUP = "'‘’\"“”`*"
_JOIN = re.compile(r"(?:,|및|와|과|그리고|/|·|&|AND|,AND|,및|,그리고)+")
_SUBJECT = re.compile(r"\s*(?:(?:성분|추출물)\s*)?(?:들)?(?:은|는|이|가)\s*")
_GROUP_SUBJECT = re.compile(
    r"\s*(?:등|[과와]\s*같은)\s*(?:[\w/-]+\s+){0,4}"
    r"(?:성분|추출물)(?:들)?(?:은|는|이|가)\s*"
)
_ENDINGS = [
    ("있으며", "있습니다."), ("하며", "합니다."), ("주며", "줍니다."),
    ("되며", "됩니다."), ("이며", "입니다."), ("하고", "합니다."),
    ("주고", "줍니다."), ("되고", "됩니다."), ("돕고", "돕습니다."),
]
# Bare "helpful/effective choice" is an endorsement, not a stated benefit.
# These cues describe the model's assertion; they do not validate its efficacy.
_EXPLANATION = re.compile(
    r"공급|유지|완화|보호|개선|강화|조절|억제|감소|방지|예방|"
    r"진정|보습|항염|항산화|항균|수렴|수축|탄력|세정|차단|유연|촉진|회복|흡수|제거|"
    r"\b(?:reduces?|soothes?|hydrates?|moisturizes?|protects?|improves?|"
    r"maintains?|prevents?|moisture|hydration|antioxidant|anti-inflammatory)\b", re.I
)
_NEGATIVE = re.compile(
    r"(?:추천|권장|권하|사용|도포|선택|포함)\s*(?:하지|하지는)\s*"
    r"(?:않|마|말)|(?:쓰|바르)지\s*(?:않|마|말)|"
    r"(?:사용|도포|선택)(?:을|은)?\s*(?:피하|피해야|피하는|삼가|금지)|"
    r"\b(?:not\s+recommended|should\s+not\s+be\s+used|"
    r"do\s+not\s+use|don't\s+use)\b", re.I
)
_AVOID_AFTER = re.compile(
    r"^\s*(?:들)?(?:은|는|이|가|을|를)?\s*"
    r"(?:(?:가급적|가능하면|반드시)\s*)?(?:피하|피해야|피하는|제외|비추천)"
)
_AVOID_BEFORE = re.compile(
    r"(?:피해야\s*(?:할\s*)?(?:성분|것)(?:은|:)?|"
    r"(?:사용|추천|권장)하지\s*(?:않는|말아야\s*할)\s*성분(?:은|:)?|"
    r"\bavoid|\bdo\s+not\s+use)\s*[:：]?\s*$", re.I
)


def _normalized(text: str) -> tuple[str, list[int]]:
    """Keep offsets into the original text while normalizing aliases."""
    chars, offsets = [], []
    for i, ch in enumerate(text):
        if ch.isspace() or ch in _MARKUP:
            continue
        upper = ch.upper()
        chars.extend(upper)
        offsets.extend([i] * len(upper))
    return "".join(chars), offsets


@dataclass(frozen=True)
class _Mention:
    name: str
    start: int
    end: int


class _AliasSpans:
    def __init__(self, ext: IngredientExtractor):
        self.names = {}
        for alias, name in ext.lexicon.items():
            names = canonical([name])
            key, _ = _normalized(alias)
            if key and names:
                self.names[key] = names[0]
        # Longest alias wins at each position; consumed spans cannot overlap.
        keys = sorted(self.names, key=lambda k: (-len(k), k))
        self.pattern = re.compile("|".join(re.escape(k) for k in keys)) if keys else None

    def find(self, text: str) -> list[_Mention]:
        if self.pattern is None:
            return []
        norm, offsets = _normalized(text)
        found = []
        for match in self.pattern.finditer(norm):
            start, end = offsets[match.start()], offsets[match.end() - 1] + 1
            # Normalization must not turn an alias into a substring of a word.
            if start and re.match(r"[가-힣A-Za-z0-9]", text[start - 1]):
                continue
            if end < len(text) and re.match(r"[A-Za-z0-9]", text[end]):
                continue
            if match[0] == "황" and end < len(text) and text[end] not in " 은는이가을를과와의에로.,;)":
                continue
            found.append(_Mention(self.names[match[0]], start, end))
        # A bilingual parenthesis is one mention, not a second subject.
        merged = []
        for mention in found:
            if merged and merged[-1].name == mention.name:
                gap, _ = _normalized(text[merged[-1].end:mention.start])
                closing = re.match(r"['’\"”*`\s]*\)", text[mention.end:])
                if gap == "(" and closing:
                    old = merged.pop()
                    merged.append(_Mention(old.name, old.start, mention.end + closing.end()))
                    continue
            merged.append(mention)
        return merged


def _groups(text: str, mentions: list[_Mention]) -> list[list[_Mention]]:
    """Only names joined by an explicit enumeration share a subject."""
    groups = []
    for mention in mentions:
        gap = _normalized(text[groups[-1][-1].end:mention.start])[0] if groups else ""
        prior_names = {m.name for m in groups[-1]} if groups else set()
        # "A, B, C과 같은 A들은 ..." recaps the explicit group, not A alone.
        recap = (len(prior_names) >= 2 and mention.name in prior_names
                 and re.fullmatch(r"(?:등|[과와]같은)", gap))
        if groups and (_JOIN.fullmatch(gap) or recap):
            groups[-1].append(mention)
        else:
            groups.append([mention])
    return groups


def _after_name(text: str, end: int) -> str:
    return text[end:].lstrip(_MARKUP + " \t")


def _finish_explanation(body: str) -> str | None:
    body = body.strip(_MARKUP + " \t,;")
    body = re.sub(r"[,;]?\s*(?:또한|그리고|반면에?|하지만)\s*,?$", "", body).rstrip(" ,;")
    if not body or re.match(r"^(?:[)\],;]|및\b|등\b|과\s|와\s)", body):
        return None
    # "A or B이 포함된 제품을 사용하여 ..." describes the product/action,
    # not B's individual benefit. It must not replace an explicit shared claim.
    if re.match(r"^(?:포함|함유|배합)된\s*(?:제품|화장품|토너|세럼|에센스|크림|로션)"
                r"(?=\s|[은는이가을를의에]|[.!?]|$)", body):
        return None
    # An open parenthesis or a list tail is not an attributable explanation.
    if body.count("(") != body.count(")") or not _EXPLANATION.search(body):
        return None
    for ending, replacement in _ENDINGS:
        if body.endswith(ending):
            body = body[:-len(ending)] + replacement
            break
    if not re.search(r"(?:다|요)[.!?]?$", body):
        if not (re.search(r"[A-Za-z]", body) and body.endswith((".", "!", "?"))):
            return None
    if not body.endswith((".", "!", "?")):
        body += "."
    return body


def _sentence_evidence(sentence: str, spans: _AliasSpans) -> tuple[list[str], dict, set[str]]:
    mentions = spans.find(sentence)
    groups = _groups(sentence, mentions)
    candidates, excluded = {}, set()
    for index, group in enumerate(groups):
        names = list(dict.fromkeys(m.name for m in group))
        stop = groups[index + 1][0].start if index + 1 < len(groups) else len(sentence)
        tail = _after_name(sentence[:stop], group[-1].end)
        # Negative recommendations win even if another sentence praises the name.
        prefix_start = groups[index - 1][-1].end if index else 0
        prefix = sentence[prefix_start:group[0].start].rstrip(_MARKUP + " \t")
        if _NEGATIVE.search(tail) or _AVOID_AFTER.search(tail) or _AVOID_BEFORE.search(prefix):
            excluded.update(names)
            continue
        shared = len(names) > 1
        subject = _GROUP_SUBJECT.match(tail) if shared else None
        if subject is None:
            subject = _SUBJECT.match(tail)
        if subject:
            body = tail[subject.end():]
        elif tail.startswith((":", "：")):
            body = tail[1:].lstrip()
        elif re.match(r"(?:helps?|supports?|provides?|reduces?|soothes?|hydrates?|"
                      r"moisturizes?|protects?|improves?|maintains?|prevents?)\b", tail, re.I):
            body = tail
        else:
            continue
        purpose = _finish_explanation(body)
        if purpose:
            for name in names:
                candidates[name] = (purpose, "model_shared" if shared else "model")
    return [m.name for m in mentions], candidates, excluded


def _ingredient_evidence(texts: list[str], spans: _AliasSpans) -> tuple[list[str], dict, set[str]]:
    names, purposes, excluded = [], {}, set()
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+|(?<=다\.)\s*|\n+", text):
            seen, candidates, negative = _sentence_evidence(sentence, spans)
            names.extend(n for n in seen if n not in names)
            excluded.update(negative)
            for name, candidate in candidates.items():
                # A later individual explanation takes precedence over a list/group.
                if name not in purposes or (purposes[name][1] == "model_shared" and candidate[1] == "model"):
                    purposes[name] = candidate
    return names, purposes, excluded


def _purpose(sentence: str, name: str) -> str | None:
    """Compatibility helper for a single literal name; API uses the full lexicon."""
    spans = _AliasSpans(IngredientExtractor({name: name}))
    _, purposes, excluded = _ingredient_evidence([sentence], spans)
    return purposes.get(name, (None, "unavailable"))[0] if name not in excluded else None


def structure_ingredients(parsed: dict, ext: IngredientExtractor | None, limit: int = 5,
                          *, include_routine: bool = False) -> list[dict]:
    if ext is None or limit <= 0:
        return []
    steps = parsed.get("reasoning", [])
    step2 = next((s["content"] for s in steps if "성분" in s.get("title", "")),
                 steps[min(1, len(steps) - 1)]["content"] if steps else "")
    spans = _AliasSpans(ext)
    texts = [step2, parsed.get("answer", "")]
    if include_routine:
        # A removed efficacy sentence does not erase an explicit recommendation
        # still present in a surviving action. Preserve the name, not its claim.
        texts.extend(step["action"] for step in structure_routine(parsed))
    names, purposes, excluded = _ingredient_evidence(texts, spans)
    # Do not recommend a name that is explicitly rejected in the usage guidance.
    for step in steps:
        if step.get("content", "") != step2:
            _, _, negative = _ingredient_evidence([step.get("content", "")], spans)
            excluded.update(negative)
    out = []
    for name in names:
        if name in excluded:
            continue
        purpose, source = purposes.get(name, (None, "unavailable"))
        out.append({"name": name, "purpose": purpose, "priority": len(out) + 1,
                    "purpose_status": "provided" if purpose else "unavailable",
                    "purpose_source": source})
        if len(out) >= limit:
            break
    return out


def contradictory_ingredients(parsed: dict, ext: IngredientExtractor) -> list[str]:
    """Names both positively mentioned as candidates and explicitly excluded.

    Bounded grammar check; no inference of aliases outside the supplied lexicon.
    """
    spans = _AliasSpans(ext)
    steps = parsed.get("reasoning", [])
    step2 = next((s["content"] for s in steps if "성분" in s.get("title", "")), "")
    texts = [step2, parsed.get("answer", "")]
    texts.extend(s["action"] for s in structure_routine(parsed))
    positive, excluded = set(), set()
    for text in texts:
        for sentence in _sentences(text):
            names, _, negative = _sentence_evidence(sentence, spans)
            positive.update(set(names) - negative)
            excluded.update(negative)
    return sorted(positive & excluded)



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
_ACTION = re.compile(
    r"바르|발라|도포하|도포해|도포합|사용하|사용해|사용합|세안하|세안해|세안합|"
    r"클렌징하|클렌징해|선택하|선택해|선택합|헹구|씻|두드|흡수시|마사지하|"
    r"덧발|보습하|보습해|섭취하|섭취해|수면을\s*취|마시|"
    r"각질\s*제거를\s*(?:통해|하)|차단제(?:는|를)[^.!?]*(?:필수|바르|사용)|"
    r"피하|피해야|피하는|삼가|문지르지|벗기지|(?:쓰|바르|사용하|도포하|하지)지?\s*(?:마|말)|"
    r"패치\s*테스트(?:를)?\s*(?:진행|실시|하|해)|"
    r"\b(?:apply|use|wash|rinse|avoid|do not|don't)\b", re.I
)


def structure_routine(parsed: dict, limit: int = 6) -> list[dict]:
    step3 = next((s["content"] for s in parsed["reasoning"] if "사용법" in s.get("title", "") or "루틴" in s.get("title", "")),
                 parsed["reasoning"][-1]["content"])
    steps = []
    for s in _sentences(step3):
        s = _ORDINAL.sub("", s)
        if not _ACTION.search(s):
            continue
        steps.append({"step": len(steps) + 1, "time": _time_of(s), "action": s})
        if len(steps) >= limit:
            break
    return steps
