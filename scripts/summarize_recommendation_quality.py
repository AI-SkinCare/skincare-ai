"""Summarize completed local recommendation trials without grading efficacy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from evaluate_recommendation_quality import read_rows, summarize, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reprocessed", action="store_true", help="Compare captured outputs after final pure postprocessing")
    args = parser.parse_args()
    trials = {}
    for name in ("v2", "v3"):
        folder = args.root / f"{name}_final"
        saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if saved.get("phase") != "complete" or (folder / "failure.json").exists():
            raise SystemExit(f"Not a completed successful execution: {folder}")
        rows = read_rows(folder / ("reprocessed.jsonl" if args.reprocessed else "responses.jsonl"))
        if len(rows) != 16 or len({r["case_id"] for r in rows}) != 16:
            raise SystemExit(f"Expected 16 unique completed cases: {folder}")
        stats = summarize(rows)
        stats["elapsed_s"] = saved["elapsed_s"]
        stats["raw_generation_tokens"] = sum(r["generation"]["result"]["output_tokens"] for r in rows if r["generation"])
        stats["input_kinds"] = dict(Counter(r["kind"] for r in rows))
        trials[name] = {"rows": rows, "stats": stats, "provenance": json.loads((folder / "provenance.json").read_text(encoding="utf-8"))}
    left, right = trials["v2"], trials["v3"]
    if left["provenance"]["code_sha256"] != right["provenance"]["code_sha256"]:
        raise SystemExit("Different serving code; comparison needs review")
    if [r["case_id"] for r in left["rows"]] != [r["case_id"] for r in right["rows"]]:
        raise SystemExit("Different cases")
    for a, b in zip(left["rows"], right["rows"]):
        if a.get("request") != b.get("request"):
            raise SystemExit(f"Different input request: {a['case_id']}")
        if a["generation"] and b["generation"] and a["generation"]["input"] != b["generation"]["input"]:
            raise SystemExit(f"Different Qwen input: {a['case_id']}")
    queue = []
    for model, trial in trials.items():
        for row in trial["rows"]:
            rec = row["response"].get("recommendation")
            if not rec:
                continue
            texts = rec["reasoning"] + [rec["answer"]] + [i["purpose"] for i in rec["ingredients"] if i["purpose"]]
            for text in texts:
                for sentence in re.split(r"(?<=[.!?])\s+", text):
                    if re.search(r"원인|장벽|강력|탁월|예방|재발|해독|안전|항염|보톡스|알레르기|스크럽|필링", sentence):
                        queue.append({"model": model, "case_id": row["case_id"], "kind": row["kind"],
                                      "candidate_sentence": sentence, "review_kind": "keyword_candidate_not_an_error_label",
                                      "expert_verdict": None, "evidence_source": None, "approved_for_training": False})
    unique = {json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in queue}
    with (args.root / "review_queue.jsonl").open("w", encoding="utf-8") as f:
        for row in unique.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    comparison = {name: trial["stats"] for name, trial in trials.items()}
    comparison["review_candidates"] = len(unique)
    comparison["paired_input_and_serving_code_match"] = True
    comparison["mode"] = "final_postprocessing_of_real_generations" if args.reprocessed else "original_live_API"
    comparison["efficacy_accuracy"] = None
    comparison["model_default_changed"] = False
    write(args.root / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
