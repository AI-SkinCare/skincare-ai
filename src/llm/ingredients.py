"""성분명 사전 구축 및 추출.

원본 데이터의 성분 표기가 제각각이다:
  - 'SULFUR(유황)'            영문(한글)
  - '알로에신(Aloesin)'        한글(영문), 따옴표 유무 혼재
  - '**우엉뿌리추출물**'       굵게
  - '첫째, 우엉뿌리추출물은'    서수 + 조사
정규식 하나로는 30% 정도만 잡혀서, 말뭉치 전체에서 사전을 먼저 만들고
정답·예측 양쪽에 '같은 추출기'를 적용한다. 그래야 평가(Top-k 일치율)가 공정하다.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

# 성분이 아닌 일반 제형·행위 어휘 (사전에서 제외)
STOPWORDS = {
    # 제형·행위
    "성분", "보습제", "보습", "자외선차단제", "선크림", "토너", "세럼", "앰플", "크림", "클렌저",
    "에센스", "로션", "마스크팩", "각질제거제", "이중세안", "세안", "제품", "화장품", "성분들", "보호막",
    # 피부 상태·생리 용어 (성분 아님)
    "피부", "수분", "각질", "피지", "멜라닌", "콜라겐", "엘라스틴", "항산화", "항염", "진정", "미백",
    "주름", "모공", "탄력", "자외선", "피부장벽", "장벽", "유분", "천연", "기능성", "색소침착", "염증",
    "여드름균", "활성산소", "수용체", "기질", "길항제", "효소", "티로시나아제", "티로시나제", "티로신",
    "호르몬", "멜라닌세포자극호르몬", "펩타이드", "각질제거", "제거", "조절",
    # 조사·부사·일반어 (따옴표 없는 '단어(영문)' 패턴에서 딸려 들어옴)
    "효과", "위해", "위해서는", "관리", "또한", "특히", "뛰어난", "마지막으로", "첫째", "둘째", "셋째",
    "넷째", "다섯째", "이를", "이는", "사용", "권장", "필수",
    # 어미만 남은 조각
    "추출물", "오일", "파우더", "가루", "솔트", "워터", "엑스", "대나무",
}
# 사전에는 잡히지만 특정 성분이 아닌 이름: 성분 계열·제품 속성·생체 물질·균.
# 사전 키에는 남겨 두고(해당 텍스트를 소비해서 '레티노이드' 안의 부분 매칭을 막음) 추출 결과에서만 뺀다.
# 정답에 '항산화제' 854회, '레티노이드' 281회가 들어 있어, 모델이 '항산화제'라고만 써도 정답으로 채점되던 문제.
NON_INGREDIENTS = {
    "항산화제", "논코메도제닉", "레티노이드", "리모노이드", "신호 펩타이드",
    "도파퀴논", "큐티박테리움 아크네스",
}
# 같은 성분의 다른 표기 (INCI명이 달리 적혀 harvest의 Union-Find로 묶이지 않은 것)
ALIASES = {
    "트라넥삼산": "트라넥사믹애씨드",
    "트라넥사민애씨드": "트라넥사믹애씨드",
    "아젤라익산": "아젤라익애씨드",
    "아지렐린": "아세틸헥사펩타이드-8",
}


def canonical(names: list[str]) -> list[str]:
    """대표명 정규화: 별칭 통합, 성분 아닌 이름 제거, 순서 유지·중복 제거."""
    out: list[str] = []
    for n in names:
        n = ALIASES.get(n, n)
        if n not in NON_INGREDIENTS and n not in out:
            out.append(n)
    return out
# 한글 성분명으로 볼 만한 어미 (서수/굵게 패턴처럼 정밀도가 낮은 패턴에만 적용)
ING_SUFFIX = re.compile(
    r"(추출물|오일|워터|엑스|가루|즙|염|산|애씨드|아마이드|놀|펩타이드|세라마이드|파우더|글루칸|하이드레이트|"
    r"알코올|글리세린|판테놀|레이트|나이드|토코페롤|비타민[A-Z0-9]*|씨|잎|꽃|뿌리|열매|발효물|여과물|"
    r"폴리머|솔트|클레이|에이트|이드|린|틴|신|올|논|닌|젠|카|민)$"
)

_Q = "['‘’\"“”]?"
# 한글(영문): 따옴표 있으면 공백 허용, 없으면 한 단어
P_KO_EN_QUOTED = re.compile(r"['‘’\"“”]([가-힣][가-힣0-9\- /]{0,38}?)\s?\(([A-Za-z][A-Za-z0-9 \-/,\.']{1,70})\)['‘’\"“”]")
P_KO_EN_WORD = re.compile(r"(?<![가-힣])([가-힣][가-힣0-9\-]{1,24})\s?\(([A-Za-z][A-Za-z0-9 \-/,\.']{1,70})\)")
# 영문(한글)
P_EN_KO = re.compile(r"\b([A-Z][A-Z0-9\-/,\.]*(?: [A-Z0-9\-/,\.]+){0,8})\s?\(([가-힣][가-힣0-9 \-/]{0,30})\)")
# 저정밀 패턴
P_BOLD = re.compile(r"\*\*([가-힣][가-힣0-9 \-/]{1,30}?)\*\*")
P_ORDINAL = re.compile(
    r"(?:첫째|둘째|셋째|넷째|다섯째|여섯째|마지막으로|또한)\s?,\s?" + _Q + r"([가-힣][가-힣0-9\-]{1,25})" + _Q + r"(?:은|는|이|가|과|와)\s"
)


_BAD_EN = re.compile(r"\b(SPF|PA|UVA?|UVB|IU|PH)\b|\+|%|\d{2,}", re.I)


def _ok_en(name: str) -> bool:
    n = name.strip()
    return len(re.sub(r"[^A-Za-z]", "", n)) >= 4 and not _BAD_EN.search(n)


def _key(name: str) -> str:
    """비교용 정규화: 공백·따옴표 제거, 영문 대문자화."""
    return re.sub(r"[\s'‘’\"“”]", "", name).upper()


def _ok_ko(name: str) -> bool:
    k = _key(name)
    return 2 <= len(k) <= 30 and k not in {_key(s) for s in STOPWORDS}


def harvest(texts: list[str]) -> tuple[dict[str, str], collections.Counter]:
    """말뭉치에서 (정규화키 → 대표 한글명) 사전과 대표명별 출현 빈도를 만든다.

    같은 성분의 여러 한글 표기(리치씨앗가루 / 리치씨파우더 / 여지씨가루 …)는
    공유하는 영문 INCI명을 매개로 Union-Find로 묶어 하나의 대표명으로 통합한다.
    """
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    display: dict[str, str] = {}
    ko_freq: collections.Counter = collections.Counter()
    weak: collections.Counter = collections.Counter()
    en_keys: set[str] = set()

    for t in texts:
        pairs: list[tuple[str, str]] = []
        for pat in (P_KO_EN_QUOTED, P_KO_EN_WORD):
            for ko, en in pat.findall(t):
                # 따옴표 없는 한 단어 패턴은 정밀도가 낮아 성분 어미까지 요구
                if pat is P_KO_EN_WORD and not ING_SUFFIX.search(_key(ko)):
                    continue
                pairs.append((ko, en))
        pairs += [(ko, en) for en, ko in P_EN_KO.findall(t)]
        for ko, en in pairs:
            if _ok_ko(ko) and _ok_en(en):
                k, e = _key(ko), "EN:" + _key(en)
                display.setdefault(k, ko.strip())
                ko_freq[k] += 1
                en_keys.add(e)
                union(e, k)
        for pat in (P_BOLD, P_ORDINAL):
            for ko in pat.findall(t):
                if _ok_ko(ko) and ING_SUFFIX.search(_key(ko)):
                    weak[_key(ko)] += 1
                    display.setdefault(_key(ko), ko.strip())

    # 저정밀 패턴 후보는 2회 이상 나오거나 고정밀 패턴에서도 확인된 경우만 채택
    for k, c in weak.items():
        if c >= 2 or k in ko_freq:
            ko_freq[k] += c
            find(k)

    # 그룹별 대표명 = 가장 많이 쓰인 한글 표기
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for k in ko_freq:
        groups[find(k)].append(k)
    lexicon: dict[str, str] = {}
    freq: collections.Counter = collections.Counter()
    for root, members in groups.items():
        rep = display[max(members, key=lambda m: ko_freq[m])]
        for m in members:
            lexicon[m] = rep
        freq[rep] += sum(ko_freq[m] for m in members)
    for e in en_keys:  # 영문 표기로 나와도 같은 대표명으로
        root = find(e)
        if root in groups:
            lexicon[e[3:]] = display[max(groups[root], key=lambda m: ko_freq[m])]
    return lexicon, freq


class IngredientExtractor:
    """사전 기반 추출기. 가장 긴 이름부터 매칭해 '알로에'가 '알로에신' 안에서 잡히는 것을 막는다."""

    def __init__(self, lexicon: dict[str, str]):
        self.lexicon = lexicon
        keys = sorted(lexicon, key=len, reverse=True)
        self._pattern = re.compile("|".join(re.escape(k) for k in keys)) if keys else None

    @classmethod
    def load(cls, path: str | Path) -> "IngredientExtractor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["lexicon"])

    def extract(self, text: str) -> list[str]:
        """등장 순서대로 중복 없는 대표 한글 성분명 리스트 (별칭 통합, 성분 아닌 이름 제외)."""
        if not text or self._pattern is None:
            return []
        norm = _key(text)
        # finditer는 겹치지 않게 왼쪽부터, 대안은 긴 것 우선
        return canonical([self.lexicon[m.group(0)] for m in self._pattern.finditer(norm)])


def save_lexicon(path: str | Path, lexicon: dict[str, str], freq: collections.Counter) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"lexicon": lexicon, "freq": dict(freq.most_common())}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
