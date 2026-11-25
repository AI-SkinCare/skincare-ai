"""Prepare the public research inputs in a local, untracked directory."""
import argparse
import ast
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

from evaluate_frontal_symptoms import CLINICAL, SCIN_FACE_REGIONS

SCIN = "https://storage.googleapis.com/dx-scin-public-data/"
LABELS = {
    "Acne": "acne",
    "Rosacea": "rosacea",
    "Psoriasis": "psoriasis",
    "Seborrheic Dermatitis": "seborrheic",
    "Atopic Dermatitis": "atopic",
}
LIMIT = 30 * 1024**2
SCIN_METADATA = {
    "scin_cases.csv": "7923ee9ba9775af413ca115c4587dc1bd32259d7936aa97fa80d88dd616ca3c3",
    "scin_labels.csv": "616ad03c24c304f5f7ef01b4116e4d3274f1c71c9b19a0e3d5620698b081a9fe",
}


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=25) as response:
                data = response.read(LIMIT + 1)
        except (TimeoutError, URLError) as error:
            if attempt == 2 or (isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429):
                raise
            time.sleep(2 ** attempt)
            continue
        if len(data) > LIMIT:
            raise ValueError("The input exceeds the download limit")
        return data


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def select_scin(cases, labels):
    cases_by_id = {row["case_id"]: row for row in cases}
    selected = []
    for row in labels:
        case = cases_by_id[row["case_id"]]
        if case["body_parts_head_or_neck"] != "YES":
            continue
        weights = ast.literal_eval(row["weighted_skin_condition_label"] or "{}")
        if not weights:
            continue
        name = max(weights, key=weights.get)
        if name not in LABELS or weights[name] < 0.65:
            continue
        names = ast.literal_eval(row["dermatologist_skin_condition_on_label_name"] or "[]")
        confidence = ast.literal_eval(row["dermatologist_skin_condition_confidence"] or "[]")
        if any(label == name and score >= 4 for label, score in zip(names, confidence, strict=True)):
            selected.append((name, case))
    if len(selected) != 31:
        raise ValueError("SCIN release differs from the screened 31-case cohort")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    for subdir in ("pmc", "scin"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        metadata_urls = [SCIN + "dataset/" + name for name in SCIN_METADATA]
        cases_raw, labels_raw = pool.map(fetch, metadata_urls)
        for (name, expected), data in zip(SCIN_METADATA.items(), (cases_raw, labels_raw), strict=True):
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"SCIN release changed: {name}")
        selected = select_scin(rows(cases_raw), rows(labels_raw))
        metadata = [{} for _ in selected]
        downloads = {}
        for photo in SCIN_FACE_REGIONS:
            index, slot = map(int, Path(photo).stem.split("_"))
            label, case = selected[index - 1]
            member = case[f"image_{slot}_path"]
            downloads[root / "scin" / photo] = SCIN + quote(member)
            metadata[index - 1] = {
                key: value for key, value in case.items()
                if key.startswith("body_parts_") or key in {
                    "condition_symptoms_itching", "textures_rough_or_flaky", "condition_duration",
                }
            }
            metadata[index - 1].update(label=LABELS[label], images=[photo])
        for article in sorted({item[0] for item in CLINICAL}):
            downloads[root / "pmc" / f"{article}.zip"] = (
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{article}/supplementaryFiles"
            )
        pending = {path: url for path, url in downloads.items() if not path.is_file()}
        for path, data in zip(pending, pool.map(fetch, pending.values()), strict=True):
            path.write_bytes(data)
    (root / "scin_download.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
