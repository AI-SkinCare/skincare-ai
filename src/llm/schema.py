"""학습과 서빙이 공유하는 입력/출력 규약.

프롬프트를 만드는 함수는 반드시 이 파일 하나에만 둔다.
학습 때와 서빙 때 프롬프트가 조금이라도 다르면(train/serve skew)
에러 없이 성능만 조용히 떨어지기 때문이다.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# ── 라벨 어휘 (「스킨케어 성분-효능 추천 데이터」 9,000건에서 확인한 실제 값) ──
SKIN_TYPES = ("건성", "중성", "지성", "복합성")
CONCERNS = (
    "과각질/악건성",
    "모공",
    "미백(색소침착/기미/칙칙함)",
    "민감성(트러블/자극감)",
    "붉어짐(홍조)",
    "여드름/뾰루지",
    "주름",
    "피부처짐/탄력저하",
)
EXTERNAL_FACTORS = (
    "자외선/블루라이트",
    "대기오염물질",
    "스트레스",
    "계절적요인",
    "화장품",
    "생활습관",
    "호르몬변화",
)
GENDERS = ("여성", "남성")

SYSTEM_PROMPT = (
    "당신은 사용자의 피부 정보를 바탕으로 스킨케어 성분과 관리법을 단계적으로 추론해 "
    "제안하는 뷰티 상담 도우미입니다. 의학적 진단이나 치료, 처방을 제공하지 않으며, "
    "질환이 의심되는 경우 피부과 전문의 상담을 권합니다. "
    "답변은 반드시 지정된 JSON 형식으로만 작성합니다."
)

OUTPUT_FORMAT_GUIDE = (
    "아래 JSON 형식으로만 답하세요.\n"
    '{"reasoning": ['
    '{"step": 1, "title": "문제 분석 및 원인 파악", "content": "..."}, '
    '{"step": 2, "title": "성분 선택 및 근거 제시", "content": "..."}, '
    '{"step": 3, "title": "사용법 및 관리방안", "content": "..."}'
    '], "answer": "..."}'
)


@dataclass
class ExternalFactor:
    factor: str
    details: str = ""


@dataclass
class RecInput:
    """추천 모델 입력. 학습 데이터(meta/external)와 서빙(설문+측정값) 모두 이 형태로 변환한다."""

    gender: str
    age: int
    skin_type: str
    concerns: list[str]
    condition_codes: str | None = None  # 예: "NS/P1/W2/NA/VP/SA/ND/NR" — 의미 일부 미확인, docs/label_spec.md 참고
    external_factors: list[ExternalFactor] = field(default_factory=list)
    user_note: str | None = None

    def validate(self) -> None:
        if self.gender not in GENDERS:
            raise ValueError(f"gender는 {GENDERS} 중 하나여야 합니다: {self.gender!r}")
        if not (10 <= int(self.age) <= 100):
            raise ValueError(f"age 범위 오류: {self.age}")
        if self.skin_type not in SKIN_TYPES:
            raise ValueError(f"skin_type은 {SKIN_TYPES} 중 하나여야 합니다: {self.skin_type!r}")
        if not self.concerns:
            raise ValueError("concerns가 비어 있습니다")
        bad = [c for c in self.concerns if c not in CONCERNS]
        if bad:
            raise ValueError(f"알 수 없는 concern: {bad}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecInput":
        ext = [ExternalFactor(**e) if isinstance(e, dict) else e for e in d.get("external_factors", [])]
        return cls(
            gender=d["gender"],
            age=int(d["age"]),
            skin_type=d["skin_type"],
            concerns=list(d["concerns"]),
            condition_codes=d.get("condition_codes"),
            external_factors=ext,
            user_note=d.get("user_note"),
        )


def build_user_prompt(x: RecInput) -> str:
    lines = [
        f"- 성별: {x.gender}",
        f"- 나이: {x.age}세",
        f"- 피부 타입: {x.skin_type}",
        f"- 주요 고민: {', '.join(x.concerns)}",
    ]
    if x.condition_codes:
        lines.append(f"- 피부 상태 코드: {x.condition_codes}")
    if x.external_factors:
        parts = [f"{e.factor}({e.details})" if e.details else e.factor for e in x.external_factors]
        lines.append(f"- 외부 요인: {'; '.join(parts)}")
    if x.user_note:
        lines.append(f"- 추가 요청: {x.user_note.strip()}")
    return (
        "다음 사용자에게 맞는 스킨케어 성분과 관리법을 단계적으로 제안해 주세요.\n"
        + "\n".join(lines)
        + "\n\n"
        + OUTPUT_FORMAT_GUIDE
    )


def build_messages(x: RecInput) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(x)},
    ]


def build_target(chain_of_thought: list[dict[str, Any]], answer: str) -> str:
    """학습 정답 문자열. 들여쓰기 없는 JSON으로 만들어 토큰 수를 줄인다.

    evidence_sources는 의도적으로 제외한다. 원본 9,000건 중 최소 3,357건
    (37.3%)에 자리표시자/무관한 PMID가 있다. 나머지도 효능 근거를 확인한
    것은 아니므로, 모델의 근거 인용 정답으로 사용하지 않는다.
    """
    steps = [
        {"step": i + 1, "title": s.get("title", "").strip(), "content": s.get("content", "").strip()}
        for i, s in enumerate(chain_of_thought)
    ]
    return json.dumps({"reasoning": steps, "answer": answer.strip()}, ensure_ascii=False)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_output(text: str) -> dict[str, Any] | None:
    """모델 출력 → {"reasoning": [...], "answer": str}. 형식이 틀리면 None."""
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    steps, answer = obj.get("reasoning"), obj.get("answer")
    if not isinstance(steps, list) or not steps or not isinstance(answer, str) or not answer.strip():
        return None
    norm = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or not isinstance(s.get("content"), str):
            return None
        norm.append({"step": i + 1, "title": str(s.get("title", "")).strip(), "content": s["content"].strip()})
    return {"reasoning": norm, "answer": answer.strip()}
