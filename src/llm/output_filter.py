"""생성 결과 후처리 필터. 규칙은 configs/filter_rules.yaml에서 관리한다.

모델을 아무리 잘 학습시켜도 금칙 표현이 0%가 된다는 보장은 없다.
그래서 '학습으로 줄이고, 필터로 막는' 이중 구조를 둔다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)\s*|\n+")


@dataclass
class FilterReport:
    dropped_sentences: list[str] = field(default_factory=list)
    removed: int = 0
    softened: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.dropped_sentences or self.removed or self.softened)


class OutputFilter:
    def __init__(self, rules: dict[str, Any]):
        self.drop_terms = rules.get("drop_sentence_terms", [])
        self.keep_if = rules.get("keep_sentence_if", [])
        self.remove = [re.compile(p) for p in rules.get("remove_patterns", [])]
        self.soften = [(re.compile(a), b) for a, b in rules.get("soften", [])]
        self.disclaimer = rules.get("disclaimer", "")

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "OutputFilter":
        path = Path(path) if path else ROOT / "configs" / "filter_rules.yaml"
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    def clean_text(self, text: str, report: FilterReport) -> str:
        kept = []
        for sent in [s for s in _SENT_SPLIT.split(text) if s and s.strip()]:
            if any(t in sent for t in self.drop_terms) and not any(k in sent for k in self.keep_if):
                report.dropped_sentences.append(sent.strip())
                continue
            kept.append(sent.strip())
        out = " ".join(kept)
        for pat in self.remove:
            out, n = pat.subn("", out)
            report.removed += n
        for pat, rep in self.soften:
            out, n = pat.subn(rep, out)
            report.softened += n
        return re.sub(r"\s{2,}", " ", out).strip()

    def apply(self, parsed: dict[str, Any]) -> tuple[dict[str, Any], FilterReport]:
        """parse_output() 결과를 받아 정제본과 리포트를 반환한다."""
        report = FilterReport()
        steps = [{**s, "content": self.clean_text(s["content"], report)} for s in parsed["reasoning"]]
        answer = self.clean_text(parsed["answer"], report)
        return {"reasoning": steps, "answer": answer, "disclaimer": self.disclaimer}, report
