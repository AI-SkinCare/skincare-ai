"""실측 결과를 비교표로 정리. 문구 검토 표시는 자동 임상 판정이 아니다."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    records = [json.loads(line) for line in (args.directory / "responses.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary_path = args.directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    generated = [r for r in records if r["kind"] == "label_fixture_generation" and r["structure"]]
    groups = {}
    for r in generated:
        names = tuple(sorted(i["name"] for i in r["response"]["recommendation"]["ingredients"]))
        groups.setdefault(names, []).append(r["id"])
    shared = [{"ingredients": list(names), "case_ids": ids} for names, ids in groups.items() if len(ids) > 1]
    review = {
        "clinical_validation": False,
        "completed_cases": len(records), "generation_cases": len(generated),
        "same_ingredient_sets": shared,
        "empty_purpose_count": sum(r["structure"]["empty_purpose_count"] for r in generated),
        "ingredient_total_count": sum(r["structure"]["ingredient_count"] for r in generated),
        "fragment_purpose_examples": [{"case_id": r["id"], "ingredient": item["name"], "purpose": item["purpose"]}
                                      for r in generated for item in r["response"]["recommendation"]["ingredients"]
                                      if item.get("purpose") and re.match(r"^(및|와|과|을|를)\s|^['‘\"“].+['’\"”](?:을|를)?\s*추천", item["purpose"])],
        "empty_answer_cases": [r["id"] for r in generated if not r["structure"]["answer_nonempty"]],
        "empty_reasoning_cases": [r["id"] for r in generated if not r["structure"]["reasoning_nonempty"]],
        "empty_routine_cases": [r["id"] for r in generated if not r["structure"]["routine_count"]],
        "review_candidate_counts": dict(Counter(k for r in generated for k,v in r["review_candidates"].items() if v)),
        "note": "Review candidates are search hits, not adjudicated errors; negated warnings may match. No medical efficacy score.",
    }
    (args.directory / "content_review_metrics.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 질환 분류 → Qwen2.5 실제 PC 품질 점검", "",
             "- 실제 localhost HTTP 서버, DenseNet121 CPU + Qwen2.5-7B v2 4-bit GPU.",
             "- 질환별 추천 입력은 시험용 라벨 fixture다. 실제 환자 분류 결과가 아니다.",
             "- 얼굴 아닌 이미지는 의도적으로 만든 음성 대조군이다.",
             "- 임상 판별 정확도·질환별 성분 효능 정확도는 측정하지 않았다.",
             "- [원본 GitHub 이슈 #3](https://github.com/AI-SkinCare/skincare-ai/issues/3)에 학습/test 이미지를 커밋하지 않았다고 명시돼 있다.", "",
             "## 실제 추천 응답", "", "| 사례 | HTTP | 시간(초) | 성분 수 | 효능 설명 없음 | 폴백 |", "|---|---:|---:|---:|---:|---|"]
    for r in generated:
        s = r["structure"]
        lines.append(f"| {r['id']} | {r['http_status']} | {r['elapsed_s']:.1f} | {s['ingredient_count']} | {s['empty_purpose_count']} | {s['used_fallback']} |")
    lines += ["", "## 비얼굴 음성 대조군", "", "| 입력 | 라벨 | 최고 점수 | 불확실 |", "|---|---|---:|---|"]
    for r in records:
        if r["kind"] == "nonface_negative_control":
            d = r["response"]
            lines.append(f"| {r['id']} | {d.get('label')} | {d.get('confidence',0):.6f} | {d.get('uncertain')} |")
    lines += ["", "## 오류와 추천 보류", "", "| 사례 | HTTP | 추천 상태 |", "|---|---:|---|"]
    for r in records:
        if r["kind"] in {"invalid_input", "recommendation_gate", "nonface_full_pipeline"}:
            lines.append(f"| {r['id']} | {r['http_status']} | {r['response'].get('recommendation_status','-')} |")
    lines += ["", "## 입력별 실제 답변 및 검토 후보", "", "아래 문구 표시는 수동 검토를 위한 검색 결과다. 부정/주의 문장도 검색될 수 있어 오답률로 해석하지 않는다.", ""]
    for r in generated:
        rec = r["response"]["recommendation"]
        lines += [f"### {r['id']}", "", "- 고민: " + ", ".join(rec["concerns"]),
                  "- 성분: " + ", ".join(i["name"] for i in rec["ingredients"]), "", rec["answer"], "",
                  "관리법:", ""]
        lines += [f"- {item['action']}" for item in rec["routine"]]
        for kind, candidates in r["review_candidates"].items():
            if candidates:
                lines += ["", f"검토 후보 `{kind}`:", ""] + ["- " + sentence for sentence in candidates]
        lines.append("")
    lines += ["## 동일 성분 묶음", ""]
    for group in shared:
        lines.append("- " + ", ".join(group["case_ids"]) + ": " + ", ".join(group["ingredients"]))
    if summary:
        lines += ["", "## 실측 요약 원문", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
    findings_path = args.directory / "manual_findings.json"
    if findings_path.exists():
        findings = json.loads(findings_path.read_text(encoding="utf-8"))
        section = ["## 판정과 수동 내용 검토", "", findings["verdict"], "",
                   "HTTP 상태가 예상과 같다는 수치는 수송/검증 동작만 의미하며, 내용 품질 통과율이 아니다.", ""]
        for item in findings["findings"]:
            section += [f"### {item['title']}", "", item["evidence"], "", "조치: " + item["next_step"], ""]
        lines[2:2] = section
    lines += ["## 내용 평가 범위", "",
              "형식/지연/입력 보류는 실측 검증이다. 성분 효능의 타당성은 이 시험의 정답 라벨이 없어 정확도로 환산하지 않았다.",
              "보습·저자극 관련 내용 검토의 참고 기준:", "",
              "- [AAD 아토피 피부 관리](https://www.aad.org/public/diseases/eczema/types/atopic-dermatitis/atopic-dermatitis-coping)",
              "- [AAD 주사 피부 관리](https://www.aad.org/public/diseases/rosacea/triggers/tips)",
              "- [AAD 건선 관리](https://www.aad.org/public/diseases/psoriasis/insider/tips)", ""]
    (args.directory / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
