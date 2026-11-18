"""Score frozen DenseNet and a paired, exploratory survey comparison from local data."""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.disease_classifier import CLASS_NAMES, DenseNetDiseaseClassifier


# Baseline panel only. Horizontal bounds split published multi-panel figures.
CLINICAL = (
    ("PMC6994586", "13555_2019_335_Fig1_HTML.jpg", "acne", (1 / 3, 2 / 3)),
    ("PMC12906923", "cureus-0018-00000101702-i01.jpg", "psoriasis", (0.74, 1)),
    ("PMC7777451", "gr1.jpg", "psoriasis", (0, 0.25)),
    *(("PMC11139169", f"DIC-2024-1-4_BERNARDINI_TS_180424-Figure_{i}.jpg", "psoriasis", (0, 0.5))
      for i in range(1, 5)),
    ("PMC10412051", "dp1303a182g001.jpg", "rosacea", None),
    ("PMC10412051", "dp1303a182g002.jpg", "rosacea", None),
    ("PMC10412051", "dp1303a182g005.jpg", "rosacea", (1 / 3, 2 / 3)),
)

# One dermatologist-labelled photo per contribution. The face regions include crops
# and oblique views and are separate from the frontal comparison.
SCIN_FACE_REGIONS = (
    "001_1.png", "002_1.png", "005_1.png", "007_1.png", "008_1.png",
    "017_1.png", "021_1.png", "022_2.png", "023_1.png", "026_3.png", "029_1.png",
)


def clinical_image(archive_dir, archive_name, member, bounds):
    with ZipFile(archive_dir / f"{archive_name}.zip") as archive:
        raw = archive.read(member)
    if bounds is None:
        return raw
    with Image.open(io.BytesIO(raw)) as img:
        left, right = bounds
        panel = img.crop((round(left * img.width), 0, round(right * img.width), img.height))
        with io.BytesIO() as encoded:
            panel.convert("RGB").save(encoded, "PNG")
            return encoded.getvalue()


def summarize(truth, predictions):
    return {
        "total": len(truth),
        "correct": sum(a == b for a, b in zip(truth, predictions, strict=True)),
        "by_label": {
            name: {
                "total": sum(label == name for label in truth),
                "correct": sum(a == b == name for a, b in zip(truth, predictions, strict=True)),
                "predicted_as": dict(Counter(b for a, b in zip(truth, predictions, strict=True) if a == name)),
            }
            for name in sorted(set(truth))
        },
    }


def evaluate_clinical(model, folder):
    truth, predictions, groups = [], [], []
    uncertain = 0
    for article, member, expected, bounds in CLINICAL:
        result = model.predict(clinical_image(folder, article, member, bounds))
        truth.append(expected)
        predictions.append(result.label_en)
        groups.append(article)
        uncertain += int(result.uncertain)
    by_publication = {}
    for article in sorted(set(groups)):
        rows = [i for i, group in enumerate(groups) if group == article]
        by_publication[article] = summarize([truth[i] for i in rows], [predictions[i] for i in rows])
    return {**summarize(truth, predictions), "uncertain": uncertain, "by_publication": by_publication}


def symptoms(case):
    duration = case["condition_duration"]
    return [
        case["condition_symptoms_itching"] == "YES",
        case["textures_rough_or_flaky"] == "YES",
        duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"},
        duration == "ONE_TO_FOUR_WEEKS",
        duration in {"ONE_TO_THREE_MONTHS", "THREE_TO_TWELVE_MONTHS", "MORE_THAN_ONE_YEAR",
                     "MORE_THAN_FIVE_YEARS", "SINCE_CHILDHOOD"},
    ]


def evaluate_survey(model, folder):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cases = json.loads((folder / "scin_download.json").read_text(encoding="utf-8"))
    truth, photo, survey = [], [], []
    for name in SCIN_FACE_REGIONS:
        case = cases[int(name.split("_")[0]) - 1]
        assert name in case["images"]
        assert case["body_parts_head_or_neck"] == "YES"
        assert sum(value == "YES" for key, value in case.items() if key.startswith("body_parts_")) == 1
        result = model.predict((folder / "scin" / name).read_bytes())
        truth.append(case["label"])
        photo.append([np.log(max(result.probabilities[label], 1e-8)) for label in CLASS_NAMES])
        survey.append(symptoms(case))

    def predict(features):
        # Standardization and the linear head are both fit within each fold.
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
        return cross_val_predict(classifier, features, truth, cv=LeaveOneOut())

    only = predict(np.asarray(photo))
    combined = predict(np.column_stack((photo, survey)))
    transitions = Counter()
    for actual, before, after in zip(truth, only, combined, strict=True):
        transitions[f"{'correct' if actual == before else 'wrong'}_to_{'correct' if actual == after else 'wrong'}"] += 1
    return {
        "photo_only_head": summarize(truth, only),
        "photo_plus_survey_head": summarize(truth, combined),
        "paired_transitions": dict(transitions),
        "symptom_feature_positive_rows": np.asarray(survey, dtype=int).sum(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = DenseNetDiseaseClassifier(args.weights).load()
    result = {
        "model": model.version,
        "clinical_frontal": evaluate_clinical(model, args.inputs / "pmc"),
        "scin_paired_exploration": evaluate_survey(model, args.inputs),
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
