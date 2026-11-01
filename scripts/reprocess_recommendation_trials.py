"""Apply final pure postprocessing to every captured real generation.

Keeps original raw/API records intact. Latency is inherited from real generation,
not a claim of new GPU execution. Gated cases never gain a generated response.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evaluate_recommendation_quality import ROOT, digest, read_rows, summarize, write
from server.recommendation_response import prepare_recommendation
from server.schemas import DiseasePrediction
from src.llm.ingredients import IngredientExtractor
from src.llm.inference import RecResult
from src.llm.output_filter import OutputFilter
from src.llm.schema import RecInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    ext = IngredientExtractor.load(ROOT / "data/processed/sllm/ingredient_dict.json")
    for model in ("v2", "v3"):
        folder = args.root / f"{model}_final"
        state = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if state.get("phase") != "complete" or (folder / "failure.json").exists():
            raise SystemExit(f"Incomplete source run: {folder}")
        rows = read_rows(folder / "responses.jsonl")
        refreshed = []
        for source in rows:
            row = copy.deepcopy(source)
            generation = row.get("generation")
            if generation:
                disease = DiseasePrediction(**row["response"]["disease"])
                result = RecResult(**generation["result"])
                x = RecInput.from_dict(generation["input"])
                prior = row["response"].get("recommendation") or {}
                rec, findings = prepare_recommendation(result, x, prior.get("input_notes", []), ext,
                                                       OutputFilter.from_yaml(), disease)
                row["response"]["quality_issues"] = rec.quality_issues
                if rec.quality_status == "withheld":
                    row["response"]["recommendation"] = None
                    row["response"]["recommendation_status"] = "quality_rejected"
                    message = "생성 결과가 추천 품질 기준을 충족하지 못해 성분 추천을 보류했습니다."
                    if message not in row["response"]["guidance"]:
                        row["response"]["guidance"].append(message)
                else:
                    row["response"]["recommendation"] = rec.model_dump()
                    row["response"]["recommendation_status"] = "generated"
                row["guard_findings"] = findings
            row["mode"] = "captured_real_generation_replayed_through_final_postprocessing"
            refreshed.append(row)
        with (folder / "reprocessed.jsonl").open("w", encoding="utf-8") as f:
            for row in refreshed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats = summarize(refreshed)
        stats.update(phase="complete", mode="postprocess_replay", source_sha256=digest(folder / "responses.jsonl"))
        write(folder / "reprocessed_summary.json", stats)
        print(json.dumps({model: stats}, ensure_ascii=False))
    sources = list((ROOT / "server").glob("*.py")) + [ROOT / "src/llm/ingredients.py", ROOT / "configs/filter_rules.yaml"]
    write(args.root / "final_postprocess_provenance.json", {
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "source": "Same captured Qwen raw outputs, no extra model generation in this replay",
    })


if __name__ == "__main__":
    main()
