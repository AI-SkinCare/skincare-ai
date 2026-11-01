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
    "도파퀴논", "큐티박테리움 아크네스", "아세틸헥사펩타이드",
}
# 같은 성분의 다른 표기 (INCI명이 달리 적혀 harvest의 Union-Find로 묶이지 않은 것)
ALIASES = {
    "트라넥삼산": "트라넥사믹애씨드",
    "트라넥사민애씨드": "트라넥사믹애씨드",
    "아젤라익산": "아젤라익애씨드",
    "아지렐린": "아세틸헥사펩타이드-8",
}

INGREDIENT_SCHEMA_VERSION = "ingredient_names_v3"

# 이름의 동일성만 교정한다. 성분의 효능·안전성을 승인하는 목록은 아니다.
# 원문에서 서로 다른 식물/부위가 Union-Find로 합쳐진 사례와 누락된 이름.
NAME_OVERRIDES = {
    "카멜그라스잎/줄기추출물": "카멜그라스잎/줄기추출물",
    "CYMBOPOGON SCHOENANTHUS LEAF/STEM EXTRACT": "카멜그라스잎/줄기추출물",
    "대나무잎/줄기추출물": "대나무잎/줄기추출물",
    "밤부사불가리스잎/줄기추출물": "대나무잎/줄기추출물",
    "BAMBUSA VULGARIS LEAF/STEM EXTRACT": "대나무잎/줄기추출물",
    "BAMBUSA VULGARIS LEAF/STEM/SHOOT EXTRACT": "대나무잎/줄기/싹추출물",
    "BAMBUSA VULGARIS EXTRACT": "대나무추출물",
    "BAMBUSA VULGARIS LEAF EXTRACT": "대나무 잎 추출물",
    "BAMBUSA VULGARIS WATER": "대나무수",
    "BAMBUSA VULGARIS SHOOT EXTRACT": "밤부사 불가리스 싹 추출물",
    "리치열매추출물": "리치열매추출물",
    "LITCHI CHINENSIS FRUIT EXTRACT": "리치열매추출물",
    "LITCHI CHINENSIS SEED POWDER": "리치 씨앗 가루",
    "여지씨가루": "리치 씨앗 가루",
    "황": "유황", "SULFUR": "유황",
    "소듐티오설페이트": "티오황산나트륨",
    "SODIUM THIOSULFATE": "티오황산나트륨",
    "호스래디시퍼옥시다아제": "고추냉이 퍼옥시다아제",
    "HORSERADISH PEROXIDASE": "고추냉이 퍼옥시다아제",
    "호스래디시뿌리추출물": "서양고추냉이뿌리추출물",
    "COCHLEARIA ARMORACIA ROOT EXTRACT": "서양고추냉이뿌리추출물",
    "3-o-에틸아스코빅애씨드": "3-o-에틸아스코빅애씨드",
    "3-o-Ethyl Ascorbic Acid": "3-o-에틸아스코빅애씨드",
    "1-메틸하이단토인-2-이미드": "1-메틸하이단토인-2-이미드",
    "1-Methylhydantoin-2-Imide": "1-메틸하이단토인-2-이미드",
    "헥사펩타이드-11": "헥사펩타이드-11", "HEXAPEPTIDE-11": "헥사펩타이드-11",
    "아줄렌": "아줄렌", "AZULENE": "아줄렌",
    "자세오시딘": "자세오시딘", "JACEOSIDIN": "자세오시딘",
}
_AMBIGUOUS_KEYS = {
    "ROOTEXTRACT", "SEEDEXTRACT", "LEAFEXTRACT", "잎추출물",
    "키위추출물/키위수", "대나무추출물/대나무수",
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


_BAD_EN = re.compile(r"\b(SPF|PA|UVA?|UVB|IU|PH)\b|\+|%|,", re.I)


def _ok_en(name: str) -> bool:
    n = name.strip()
    k = _key(n)
    return (len(re.sub(r"[^A-Za-z]", "", n)) >= 4 and not _BAD_EN.search(n)
            and k not in _AMBIGUOUS_KEYS and "EXTRACT/" not in k)


def _key(name: str) -> str:
    """비교용 정규화: 공백·따옴표 제거, 영문 대문자화."""
    return re.sub(r"[\s'‘’\"“”]", "", name).upper()


def repair_lexicon(lexicon: dict[str, str]) -> dict[str, str]:
    """기존 사전도 교정한 이름 매핑으로 읽는다. 저장 파일은 수정하지 않는다."""
    repaired = {}
    overrides = {_key(k): v for k, v in NAME_OVERRIDES.items()}
    overrides.update({_key(name): name for name in NAME_OVERRIDES.values()})
    for key, value in lexicon.items():
        key = _key(key)
        if key in _AMBIGUOUS_KEYS or "," in key or "EXTRACT/" in key:
            continue
        repaired[key] = overrides.get(_key(value), value)
    repaired.update(overrides)
    # 대표 한글명도 반드시 같은 사전에서 추출할 수 있어야 한다.
    for name in list(repaired.values()):
        repaired.setdefault(_key(name), name)
    return repaired


def _ok_ko(name: str) -> bool:
    k = _key(name)
    return 2 <= len(k) <= 30 and k not in {_key(s) for s in STOPWORDS}


def harvest(texts: list[str]) -> tuple[dict[str, str], collections.Counter]:
    """같은 영문 이름 안에서만 별칭을 묶는다. 다의적인 한글 이름은 연결자로 쓰지 않는다."""
    display: dict[str, str] = {}
    by_en: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ko_ens: dict[str, set[str]] = collections.defaultdict(set)
    weak: collections.Counter = collections.Counter()
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
                k, e = _key(ko), _key(en)
                display.setdefault(k, ko.strip())
                by_en[e][k] += 1
                ko_ens[k].add(e)
        for pat in (P_BOLD, P_ORDINAL):
            for ko in pat.findall(t):
                if _ok_ko(ko) and ING_SUFFIX.search(_key(ko)):
                    weak[_key(ko)] += 1
                    display.setdefault(_key(ko), ko.strip())

    overrides = {_key(k): v for k, v in NAME_OVERRIDES.items()}
    lexicon: dict[str, str] = {}
    for en, counts in sorted(by_en.items()):
        # 명시적인 이름 교정이 없으면, 다른 EN에 연결되지 않은 KO만 대표 후보로 쓴다.
        candidates = [k for k in counts if len(ko_ens[k]) == 1]
        rep = overrides.get(en)
        if rep is None and candidates:
            rep = display[max(candidates, key=lambda k: (counts[k], k))]
        if rep is None:
            continue
        lexicon[en] = rep
        for ko in candidates:
            # 이미 확인된 KO→EN 충돌은 영문 쪽에 맞춰 덮어쓰지 않는다.
            if ko not in overrides or overrides[ko] == rep:
                lexicon[ko] = rep
    for k, c in weak.items():
        if c >= 2 and k not in ko_ens and k not in _AMBIGUOUS_KEYS:
            lexicon.setdefault(k, display[k])
    lexicon = repair_lexicon(lexicon)
    # 빈도는 별칭 출현 횟수가 아닌 원본 레코드별 등장 횟수.
    ext = IngredientExtractor(lexicon)
    freq: collections.Counter = collections.Counter()
    for text in texts:
        freq.update(ext.extract(text))
    return lexicon, freq


class IngredientExtractor:
    """사전 기반 추출기. 가장 긴 이름부터 매칭해 '알로에'가 '알로에신' 안에서 잡히는 것을 막는다."""

    def __init__(self, lexicon: dict[str, str]):
        self.lexicon = repair_lexicon(lexicon)
        keys = sorted((k for k in self.lexicon if k != "황"), key=len, reverse=True)
        # 11을 1로, 미등록 111을 11로 잘못 채점하지 않는다.
        self._pattern = re.compile("|".join(re.escape(k) + (r"(?!\d)" if k[-1:].isdigit() else "")
                                             for k in keys)) if keys else None

    @classmethod
    def load(cls, path: str | Path) -> "IngredientExtractor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["lexicon"])

    def extract(self, text: str) -> list[str]:
        """등장 순서대로 중복 없는 대표 한글 성분명 리스트 (별칭 통합, 성분 아닌 이름 제외)."""
        if not text or self._pattern is None:
            return []
        # 한 글자 별칭은 공백을 제거하기 전에 경계를 검사한다.
        text = re.sub(r"(?<![가-힣A-Za-z0-9])황(?=$|[\s'‘’\"“”.,;:()]|[은는이가을를과와의에로](?![가-힣]))", "유황", text)
        norm = _key(text)
        # finditer는 겹치지 않게 왼쪽부터, 대안은 긴 것 우선
        out = []
        for m in self._pattern.finditer(norm):
            key = m.group(0)
            out.append(self.lexicon[key])
        return canonical(out)


def save_lexicon(path: str | Path, lexicon: dict[str, str], freq: collections.Counter) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"lexicon": lexicon, "freq": dict(freq.most_common())}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
