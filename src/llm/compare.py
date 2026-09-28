"""평가 결과 비교표 (중간발표 자료용).

사용 (PowerShell, 레포 루트):
  python -m src.llm.compare

outputs/eval/*.json 에서 모델별 '가장 최근' 결과를 모아
outputs/eval/comparison.md 로 저장하고 화면에도 출력한다.

표 순서는 규칙 베이스라인 → 원본(파인튜닝 전) 모델 → 파인튜닝 모델.
이 순서로 봐야 '파인튜닝이 효과가 있었다'는 주장이 성립한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "outputs" / "eval"

COLUMNS = [
    ("n", "평가 건수", "int"),
    ("json_ok_first_try", "형식 준수(1차)", "pct"),
    ("fallback_rate", "폴백률", "pct"),
    ("precision_at_3", "precision@3", "pct"),
    ("recall_all", "recall", "pct"),
    ("answer_similarity_f1", "정답 유사도", "f3"),
    ("filter_intervention_rate", "필터 개입률", "pct"),
    ("latency_ms_p50", "응답 p50", "sec"),
    ("output_tokens_mean", "출력 토큰", "int"),
]


def _fmt(v, kind: str) -> str:
    if v is None:
        return "-"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "sec":
        return f"{v / 1000:.1f}s" if v >= 100 else f"{v}ms"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _order(tag: str) -> tuple[int, str]:
    if tag.startswith("rule"):
        return (0, tag)
    if tag.startswith("base_"):
        return (1, tag)
    if tag == "smoke":
        return (3, tag)
    return (2, tag)


def latest_results(eval_dir: Path = EVAL_DIR) -> dict[str, dict]:
    """모델 태그별 가장 최근 결과 파일 전체(summary + items). 파일명 끝의 YYYYMMDD-HHMM으로 최신을 판단한다."""
    best: dict[str, tuple[str, dict]] = {}
    for p in eval_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            s = d["summary"]
        except (KeyError, json.JSONDecodeError):
            continue
        stamp = p.stem.rsplit("_", 1)[-1]
        tag = s.get("model", p.stem)
        if tag not in best or stamp > best[tag][0]:
            best[tag] = (stamp, d)
    return {t: d for t, (_, d) in sorted(best.items(), key=lambda kv: _order(kv[0]))}


def latest_summaries(eval_dir: Path = EVAL_DIR) -> dict[str, dict]:
    return {t: d["summary"] for t, d in latest_results(eval_dir).items()}


def paired_vs_rule(results: dict[str, dict]) -> str:
    """규칙 베이스라인과 '같은 문항'만 짝지어 precision@3 차이와 95% 신뢰구간을 낸다.

    전체 표는 모델마다 평가 건수(n)가 달라 직접 비교하면 안 된다(60건 vs 201건).
    문항별 결과(items)가 있는 평가 파일끼리만 계산된다.
    """
    ref = next((d for t, d in results.items() if t.startswith("rule")), None)
    if not ref or not ref.get("items"):
        return "(규칙 베이스라인의 문항별 결과가 없습니다. `python -m src.llm.evaluate --mock`을 다시 실행하세요.)"
    ref_by_id = {i["id"]: i for i in ref["items"] if "p3_den" in i}
    lines = ["| 모델 | 짝지은 문항 | 규칙 기반 | 모델 | 차이 (95% CI) |", "|---|---:|---:|---:|---:|"]
    for tag, d in results.items():
        if tag.startswith("rule") or not d.get("items"):
            continue
        pairs = [(ref_by_id[i["id"]], i) for i in d["items"] if "p3_den" in i and i["id"] in ref_by_id]
        if not pairs:
            continue
        r = [a["p3_hit"] / a["p3_den"] for a, _ in pairs]
        m = [b["p3_hit"] / b["p3_den"] for _, b in pairs]
        diff = [y - x for x, y in zip(r, m)]
        n = len(diff)
        mean = sum(diff) / n
        sd = (sum((x - mean) ** 2 for x in diff) / (n - 1)) ** 0.5 if n > 1 else 0.0
        ci = 1.96 * sd / n ** 0.5
        mark = " ✅" if mean - ci > 0 else (" ❌" if mean + ci < 0 else " (유의하지 않음)")
        lines.append(f"| {tag} | {n} | {sum(r) / n * 100:.1f}% | {sum(m) / n * 100:.1f}% | "
                     f"{mean * 100:+.1f}%p (±{ci * 100:.1f}){mark} |")
    if len(lines) == 2:
        return "(짝지을 수 있는 모델 결과가 없습니다. 문항별 결과는 이번 수정 이후 평가한 파일에만 있습니다.)"
    return "\n".join(lines)


def build_table(summaries: dict[str, dict]) -> str:
    head = "| 모델 | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    rows = [f"| {tag} | " + " | ".join(_fmt(s.get(k), kind) for k, _, kind in COLUMNS) + " |"
            for tag, s in summaries.items()]
    return "\n".join([head, sep, *rows])


def build_concern_table(summaries: dict[str, dict]) -> str:
    """고민별 precision@3. 표본 30건 미만은 괄호로 표시(통계적으로 의미 없음)."""
    concerns: list[str] = []
    for s in summaries.values():
        for c in s.get("per_concern", {}):
            if c not in concerns:
                concerns.append(c)
    tags = list(summaries)
    lines = ["| 고민 | " + " | ".join(tags) + " |", "|---|" + "|".join("---:" for _ in tags) + "|"]
    for c in concerns:
        cells = []
        for t in tags:
            pc = summaries[t].get("per_concern", {}).get(c)
            if not pc or pc.get("precision_at_3") is None:
                cells.append("-")
                continue
            v = f"{pc['precision_at_3'] * 100:.0f}%"
            cells.append(v if pc.get("reliable") else f"({v})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


NOTES = """\
- **precision@3·recall·정답 유사도·필터 개입률은 모델이 직접 생성한 답만으로 계산**했습니다. 형식 실패로 규칙 기반 폴백이 나간 건은 제외합니다(폴백 점수가 섞이면 실패한 모델도 베이스라인 점수를 받아 실패가 가려지기 때문).
- 이 데이터는 고민별 추천 성분이 거의 고정되어 있어, '고민별 빈출 성분' 규칙만으로도 precision@3가 높게 나옵니다. 따라서 파인튜닝 효과는 **원본 모델 대비 precision@3 상승**과 **규칙 템플릿 대비 정답 유사도 상승**으로 판단합니다.
- 위 전체 표는 모델마다 평가 건수가 다를 수 있으니, 규칙 기반과의 우열은 '같은 문항끼리 비교' 표로 판단합니다. 신뢰구간이 0을 포함하면 차이가 우연일 수 있습니다.
- '항산화제', '레티노이드' 같은 성분 계열명과 균·생체 물질은 정답·예측 모두에서 제외하고 채점합니다.
- 정답 유사도는 정답 추론문과의 글자 2-gram F1로, 어휘 수준의 거친 대리 지표입니다.
- 고민별 표에서 괄호 값은 평가 표본 30건 미만으로 통계적 의미가 없습니다.
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = latest_results()
    summaries = {t: d["summary"] for t, d in results.items()}
    if not summaries:
        sys.exit("outputs/eval 에 평가 결과가 없습니다. 먼저 python -m src.llm.evaluate 를 실행하세요.")
    md = "\n\n".join([
        "# 성분 추천 모델 비교", build_table(summaries),
        "## 규칙 기반 대비 precision@3 (같은 문항끼리 비교)", paired_vs_rule(results),
        "## 고민별 precision@3", build_concern_table(summaries),
        "## 해석 시 주의", NOTES,
    ])
    out = EVAL_DIR / "comparison.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[saved] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
