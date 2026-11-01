"""학습 정답문 정제 → v2 학습 데이터.

사용 (PowerShell):
  python -m src.llm.clean_targets                        # data/processed/sllm → data/processed/sllm_v2
  python -m src.llm.clean_targets --src ... --dst ...

왜 필요한가 (v1 평가에서 확인):
  - 학습 정답문의 40.3%에 '개선합니다' 같은 효능 단정 표현이 있어, 모델이 그대로 따라 해
    서빙 필터가 답변의 36~42%에 개입했다. 필터는 문장을 지우거나 고쳐 쓰므로 답변 품질이 떨어진다.
  - 2.2%에 '치료·질환·피부염·복용' 같은 의료 표현이 있다.
  - 15.8%에 'LITCHI CHINENSIS SEED POWDER' 같은 영문 INCI명이 섞여, 사용자 답변에도 영문이 나온다.
  → 서빙 필터(configs/filter_rules.yaml)와 같은 규칙을 '학습 데이터에 미리' 적용해, 모델이 처음부터
    규정에 맞는 표현을 쓰도록 학습시킨다. ('학습으로 줄이고, 필터로 막는' 이중 구조의 앞단)

검증 데이터(val/val_eval)는 원본 그대로 복사한다. 정답 유사도를 v1과 같은 기준으로 비교하기 위해서다.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

from .ingredients import IngredientExtractor, canonical, repair_lexicon, _key
from .output_filter import FilterReport, OutputFilter

ROOT = Path(__file__).resolve().parents[2]

# 의료 표현을 문장째 버리기 전에, 뜻이 통하는 화장품 표현으로 먼저 바꾼다.
# (문장을 버리면 그 안의 성분 설명까지 사라지므로.) 긴 것부터 적용.
MEDICAL_REWRITE = [
    ("피부 질환", "피부 고민"), ("피부질환", "피부 고민"), ("질환", "피부 고민"), ("질병", "피부 고민"),
    ("치료제", "관리 제품"), ("치료", "관리"),
    ("피부염", "피부 자극"),
    ("복용", "섭취"), ("처방", "권장"), ("투약", "사용"),
]

_Q = "['‘’\"“”]"
_EN = r"[A-Za-z0-9][A-Za-z0-9\-/,\.]*(?:[ \t][A-Za-z0-9][A-Za-z0-9\-/,\.]*){0,8}"
P_KO_EN = re.compile(rf"\s?\(\s*({_EN})\s*\)")                               # 알로에신(Aloesin) → 알로에신
P_EN_KO = re.compile(rf"({_EN})\s?\(([가-힣][가-힣0-9 \-/]{{0,30}})\)")      # SULFUR(유황) → 유황
P_QUOTED_EN = re.compile(rf"({_Q})({_EN})({_Q})")                           # 'SULFUR' → '유황'
P_BARE_EN = re.compile(rf"(?<![A-Za-z])({_EN})(?![A-Za-z(])")               # SULFUR → 유황 (사전에 있을 때만)

# 치환한 이름 뒤 조사를 받침에 맞춘다: '유황'는 → '유황'은. 치환 지점에만 표시(\x00..\x01)를 달아 적용한다.
_S, _E = "\x00", "\x01"
_JOSA = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"),
         "을": ("을", "를"), "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와")}
P_JOSA = re.compile(rf"{_S}([^{_E}]+){_E}([)'‘’\"“”]*)(은|는|이|가|을|를|과|와)(?![가-힣])")


def _has_batchim(ch: str) -> bool | None:
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    return None


def _fix_josa(m: re.Match) -> str:
    name, q, josa = m.groups()
    b = _has_batchim(name[-1])
    if b is not None:
        josa = _JOSA[josa][0 if b else 1]
    return f"{name}{q}{josa}"


class TargetCleaner:
    def __init__(self, filt: OutputFilter, lexicon: dict[str, str]):
        self.filt = filt
        self.lex = repair_lexicon(lexicon)
        keys = sorted((k for k in self.lex if re.search(r"[가-힣]", k)), key=len, reverse=True)
        ko = "|".join(r"[ \t]*".join(re.escape(c) for c in k) for k in keys)
        self.paired = re.compile(rf"(?<![가-힣])({ko})\s?\(\s*({_EN})\s*\)", re.I)

    def _ko(self, en: str) -> str | None:
        name = self.lex.get(_key(en))
        return canonical([name])[0] if name and canonical([name]) else None

    def conflicting_pairs(self, text: str) -> list[str]:
        """같은 괄호에 서로 다른 성분이 적혔으면 자동으로 한쪽을 선택하지 않는다."""
        bad = []
        for m in self.paired.finditer(text):
            a, b = self._ko(m[1]), self._ko(m[2])
            if a and b and a != b:
                bad.append(m[0])
        for m in P_EN_KO.finditer(text):
            a, b = self._ko(m[1]), self._ko(m[2])
            if a and b and a != b:
                bad.append(m[0])
        return bad

    def korean_names(self, text: str, st: collections.Counter) -> str:
        def paired(m):
            a, b = self._ko(m[1]), self._ko(m[2])
            if a and a == b:
                st["inci"] += 1
                return f"{_S}{a}{_E}"
            return m[0]

        def parenthesized(m):
            # '(SULFUR)는'처럼 한글 이름 없는 경우도 이름을 보존한다.
            ko = self._ko(m[1])
            if ko:
                st["inci"] += 1
                return f"({_S}{ko}{_E})"
            return m.group(0)

        def en_ko(m):
            a, b = self._ko(m[1]), self._ko(m[2])
            if a and a == b:
                st["inci"] += 1
                return f"{_S}{a}{_E}"
            return m.group(0)

        def quoted(m):
            ko = self._ko(m.group(2).rstrip(",."))
            if ko:
                st["inci"] += 1
                return f"{m.group(1)}{_S}{ko}{_E}{m.group(3)}"
            return m.group(0)

        def bare(m):
            # 'SULFUR, MOMORDICA CHARANTIA FRUIT EXTRACT' 같은 연속 영문 안에서
            # 사전에 있는 가장 긴 단어 묶음을 찾아 바꾼다 (끝 구두점은 떼고 조회, 원문에는 남김)
            words = m.group(1).split(" ")
            out, i = [], 0
            while i < len(words):
                for j in range(len(words), i, -1):
                    cand = " ".join(words[i:j])
                    core = cand.rstrip(",.")
                    ko = self._ko(core) if len(re.sub(r"[^A-Za-z]", "", core)) >= 4 else None
                    if ko:
                        st["inci"] += 1
                        out.append(f"{_S}{ko}{_E}{cand[len(core):]}")
                        i = j
                        break
                else:
                    out.append(words[i])
                    i += 1
            return " ".join(out)

        # 충돌된 병기는 검토 대상으로 그대로 보존한다.
        conflicts = self.conflicting_pairs(text)
        for i, original in enumerate(conflicts):
            text = text.replace(original, f"\x02{i}\x03")
        text = P_EN_KO.sub(en_ko, text)
        text = self.paired.sub(paired, text)
        text = P_KO_EN.sub(parenthesized, text)
        text = P_QUOTED_EN.sub(quoted, text)
        text = P_BARE_EN.sub(bare, text)
        text = P_JOSA.sub(_fix_josa, text)
        text = text.replace(_S, "").replace(_E, "")
        for i, original in enumerate(conflicts):
            text = text.replace(f"\x02{i}\x03", original)
        return text

    def medical(self, text: str, st: collections.Counter) -> str:
        for a, b in MEDICAL_REWRITE:
            n = text.count(a)
            if n:
                text = text.replace(a, b)
                st["medical_rewrite"] += n
        return text

    def clean_text(self, text: str, st: collections.Counter) -> str:
        text = self.medical(self.korean_names(text, st), st)
        rep = FilterReport()
        # 남은 금칙어 문장 제거 + 단정 표현 완화 (서빙과 같은 규칙). 필터는 줄바꿈을 공백으로 합치므로
        # 번호 목록 형식을 살리려고 줄 단위로 적용한다.
        lines = [self.filt.clean_text(ln, rep) for ln in text.split("\n")]
        out = "\n".join(ln for ln in lines if ln)
        st["dropped_sentences"] += len(rep.dropped_sentences)
        st["softened"] += rep.softened
        st["removed"] += rep.removed
        return out

    def clean(self, target: str, st: collections.Counter) -> str:
        t = json.loads(target)
        for s in t["reasoning"]:
            s["content"] = self.clean_text(s["content"], st)
        t["answer"] = self.clean_text(t["answer"], st)
        return json.dumps(t, ensure_ascii=False)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/processed/sllm"))
    ap.add_argument("--dst", default=str(ROOT / "data/processed/sllm_v2"))
    args = ap.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    lexicon = json.loads((src / "ingredient_dict.json").read_text(encoding="utf-8"))["lexicon"]
    ext = IngredientExtractor(lexicon)
    cleaner = TargetCleaner(OutputFilter.from_yaml(), lexicon)
    filt = OutputFilter.from_yaml()

    st: collections.Counter = collections.Counter()
    rows_out, lost_gold, gold_total = [], 0, 0
    for line in open(src / "train.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r["target"] = cleaner.clean(r["target"], st)
        t = json.loads(r["target"])
        text = " ".join(s["content"] for s in t["reasoning"]) + " " + t["answer"]
        found = set(ext.extract(text))
        gold = set(ext.extract(" ".join(r["gold_ingredients"])))
        gold_total += len(gold)
        lost_gold += len(gold - found)
        # 정제 후에도 서빙 필터가 건드릴 게 남았는지 (0에 가까워야 함)
        st["still_filtered"] += int(filt.apply(t)[1].changed)
        rows_out.append(r)

    with open(dst / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("val.jsonl", "val_eval.jsonl", "ingredient_dict.json", "concern_top_ingredients.json"):
        shutil.copy2(src / name, dst / name)

    stats = {
        "train": len(rows_out),
        "inci_to_korean": st["inci"],
        "medical_rewrites": st["medical_rewrite"],
        "dropped_sentences": st["dropped_sentences"],
        "softened": st["softened"],
        "removed_patterns": st["removed"],
        "still_filtered_rate": round(st["still_filtered"] / len(rows_out), 4),
        "gold_mention_lost_rate": round(lost_gold / gold_total, 4) if gold_total else 0,
    }
    (dst / "clean_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"[saved] {dst}")


if __name__ == "__main__":
    main()
