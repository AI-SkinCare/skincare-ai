"""Compare the four serving adapters on identical labeled inputs without exporting photos."""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.disease_classifier import DenseNetDiseaseClassifier, TrainedDiseaseClassifier
from src.vision.data import read_rows


def summarize(rows):
    def tally(items):
        return {"correct": sum(r["label"] == r["prediction"] for r in items), "total": len(items)}

    times = sorted(r["ms"] for r in rows)
    return {
        **tally(rows),
        "recall_counts": {label: tally([r for r in rows if r["label"] == label])
                          for label in sorted({r["label"] for r in rows})},
        "views": {view: tally([r for r in rows if r["view"] == view])
                  for view in sorted({r["view"] for r in rows})},
        "uncertain": sum(r["uncertain"] for r in rows),
        "warm_ms_p50": times[math.ceil(len(times) * .50) - 1],
        "warm_ms_p95": times[math.ceil(len(times) * .95) - 1],
    }


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets.resolve()
    torch.set_num_threads(4)
    internal = [r for r in read_rows(assets / "data/processed/disease_20261107/images.jsonl")
                if r["split"] == "holdout" and r["origin_split"] == "Validation"]
    external = read_rows(assets / "outputs/disease_independent_20261110/independent_images.jsonl")
    cohorts = {"aihub_validation": internal, "observed_external": external}
    if not all(cohorts.values()):
        raise ValueError("Both evaluation cohorts are required")
    for rows in cohorts.values():
        random.Random(42).shuffle(rows)
    result = {"device": "cpu", "threads": 4, "seed": 42,
              "timing": "warm predict(bytes), including decode, excluding disk, network, Grad-CAM and Qwen",
              "external_groups": len({r["group"] for r in external}),
              "external_distribution": dict(Counter(r["label"] for r in external)), "models": {}}
    for name in ("densenet", "densenet121", "efficientnet_b0", "dinov2_linear"):
        classifier = (DenseNetDiseaseClassifier(assets / "model/weights/densenet121_s2.pt")
                      if name == "densenet" else TrainedDiseaseClassifier(
                          assets / "outputs/disease_models_20261107" / name / "best.pt", name)).load()
        model = {"version": classifier.version, "parameters": sum(p.numel() for p in classifier.model.parameters())}
        with ExitStack() as stack:
            archives = {p: stack.enter_context(ZipFile(p)) for p in
                        sorted({r["image_zip"] for rows in cohorts.values() for r in rows})}
            for cohort, rows in cohorts.items():
                measured = []
                for index, row in enumerate(rows):
                    raw = archives[row["image_zip"]].read(row["image_member"])
                    if index == 0:
                        classifier.predict(raw)
                    start = time.perf_counter()
                    prediction = classifier.predict(raw)
                    elapsed = (time.perf_counter() - start) * 1000
                    measured.append(dict(label=row["label"], view=row["view"], prediction=prediction.label,
                                         uncertain=prediction.uncertain, ms=round(elapsed, 2)))
                model[cohort] = summarize(measured)
        result["models"][name] = model
        print(name, model["aihub_validation"]["views"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
