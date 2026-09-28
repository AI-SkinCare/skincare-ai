"""규칙 기반 폴백 추천.

쓰이는 곳:
  1) sLLM이 두 번 연속 JSON 형식을 어겼을 때
  2) GPU 없이 개발할 때 (REC_BACKEND=mock) — 백엔드·프론트가 모델 학습 완료를 기다리지 않고 연동 가능
  3) 평가 시 베이스라인 (모델이 이것보다 나아야 의미가 있다)
"""
from __future__ import annotations

import json
from pathlib import Path

from .ingredients import canonical
from .schema import RecInput

ROOT = Path(__file__).resolve().parents[2]

# processed 데이터가 없을 때(예: 백엔드 팀원 PC)의 최소 기본값 — 실제 데이터 빈출 성분에서 발췌
_DEFAULT_TOP = {
    "과각질/악건성": ["쑥잎 추출물", "대나무추출물", "세라마이드"],
    "모공": ["리치 씨앗 가루", "여주 열매 추출물", "유황"],
    "미백(색소침착/기미/칙칙함)": ["알로에신", "헥사펩타이드-2", "댕댕이나무열매즙"],
    "민감성(트러블/자극감)": ["쑥잎 추출물", "대나무추출물", "대나무수"],
    "붉어짐(홍조)": ["쑥잎 추출물", "대나무추출물", "세라마이드"],
    "여드름/뾰루지": ["알로에 베라 잎즙 파우더", "서양고추냉이뿌리추출물", "키틴"],
    "주름": ["자두추출물", "헥사펩타이드-9", "리치 씨앗 가루"],
    "피부처짐/탄력저하": ["대나무추출물", "자두추출물", "레티놀"],
}


class RuleBasedRecommender:
    version = "rule_fallback_v1"

    def __init__(self, top_path: str | Path | None = None):
        p = Path(top_path) if top_path else ROOT / "data/processed/sllm/concern_top_ingredients.json"
        top = json.loads(p.read_text(encoding="utf-8")) if p.exists() else _DEFAULT_TOP
        self.top = {c: canonical(names) for c, names in top.items()}   # 예전 데이터의 '항산화제' 등 제거

    def ingredients_for(self, x: RecInput, k: int = 3) -> list[str]:
        out: list[str] = []
        for c in x.concerns:  # 첫 번째 고민 우선, 중복 제거
            for name in self.top.get(c, []):
                if name not in out:
                    out.append(name)
                if len(out) >= k * len(x.concerns):
                    break
        return out[: max(k, 3)]

    def recommend(self, x: RecInput) -> dict:
        ings = self.ingredients_for(x)
        concerns = ", ".join(x.concerns)
        ext = ", ".join(e.factor for e in x.external_factors) or "특이 외부 요인 없음"
        return {
            "reasoning": [
                {"step": 1, "title": "문제 분석 및 원인 파악",
                 "content": f"{x.age}세 {x.gender}, {x.skin_type} 피부로 주요 고민은 {concerns}입니다. 외부 요인으로 {ext}이(가) 영향을 줄 수 있습니다."},
                {"step": 2, "title": "성분 선택 및 근거 제시",
                 "content": f"같은 고민을 가진 사례에서 자주 권장된 성분인 {', '.join(ings)}을(를) 우선 고려할 수 있습니다."},
                {"step": 3, "title": "사용법 및 관리방안",
                 "content": "저녁에는 순한 클렌저로 세안한 뒤 해당 성분이 포함된 제품을 얇게 바르고, 아침에는 보습 후 자외선 차단제를 꼼꼼히 사용하세요. 새 제품은 팔 안쪽에 먼저 발라 자극 여부를 확인하세요."},
            ],
            "answer": f"{concerns} 관리를 위해 {', '.join(ings)} 성분을 중심으로 한 기본 루틴을 제안합니다.",
        }
