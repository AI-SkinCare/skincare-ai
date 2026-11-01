"""출처 식별자·문헌 메타데이터 검사. 효능 승인이나 전문가 검토를 대체하지 않는다."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .prepare_quality_data import ROOT, read_rows

VERSION = "bibliography_review_v1"
DEFAULT_OUT = ROOT / "data/review_20261006"


def reference_ids(text: str) -> list[str]:
    """명시적 PMID/DOI만 추출. 숫자처럼 보인다는 이유로 PMID를 추측하지 않는다."""
    ids = ["pmid:" + x for x in re.findall(r"(?:PMID\s*[:：]?\s*|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{1,9})\b", text, re.I)]
    ids += ["doi:" + x.rstrip(".,;。\"'").lower() for x in re.findall(r"10\.\d{4,9}/[^\s<>\"']+", text, re.I)]
    if re.search(r"10\.xxxx|xxxxx|placeholder|10\.1234/example", text, re.I):
        ids.append("placeholder:" + hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16])
    return sorted(set(ids))


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "SkinCareBibliographyAudit/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=18) as response:
        return json.load(response)


def resolve_reference(key: str) -> dict:
    kind, value = key.split(":", 1)
    checked = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {"reference": key, "checked_at": checked, "claim_status": "not_expert_verified"}
    if kind == "placeholder" or key == "doi:10.1234/example":
        return {**record, "status": "placeholder"}
    url = ""
    try:
        if kind == "pmid":
            query = urllib.parse.urlencode({"query": f"EXT_ID:{value} AND SRC:MED", "format": "json", "resultType": "core"})
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + query
            results = fetch_json(url).get("resultList", {}).get("result", [])
            result = next((r for r in results if r.get("id") == value and r.get("source") == "MED"), None)
            if result is None:
                return {**record, "status": "identifier_not_found", "metadata_url": url}
            metadata = {"title": result.get("title"), "year": result.get("pubYear"), "doi": result.get("doi"),
                        "publication_types": result.get("pubTypeList", {}).get("pubType", []),
                        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{value}/"}
        else:
            url = "https://api.crossref.org/works/" + urllib.parse.quote(value, safe="")
            result = fetch_json(url)["message"]
            if result.get("DOI", "").lower() != value.lower():
                return {**record, "status": "metadata_identifier_mismatch", "metadata_url": url}
            metadata = {"title": html.unescape(re.sub(r"<[^>]+>", "", " ".join(result.get("title", [])))),
                        "year": result.get("published", {}).get("date-parts", [[None]])[0][0],
                        "publication_types": [result.get("type")], "source_url": "https://doi.org/" + value}
        flags = []
        title = metadata.get("title") or ""
        if re.search(r"editorial|correction|erratum|retraction", title, re.I):
            flags.append("non_efficacy_study_title_review")
        if re.search(r"guinea.?pig|\bmice\b|\bmouse\b|\brats?\b|in vitro|cell line", title, re.I):
            flags.append("preclinical_title_human_extrapolation_review")
        if key == "pmid:12345678":
            flags.append("known_unrelated_population_declaration")
        return {**record, **metadata, "status": "metadata_resolved", "metadata_url": url, "review_flags": flags}
    except urllib.error.HTTPError as exc:
        return {**record, "status": "identifier_not_found" if exc.code == 404 else "lookup_error",
                "http_status": exc.code, "metadata_url": url}
    except (OSError, ValueError, KeyError) as exc:
        return {**record, "status": "lookup_error", "error": type(exc).__name__, "metadata_url": url}


def inventory(rows: list[dict]) -> tuple[dict, dict]:
    refs, by_row = {}, {}
    for row in rows:
        keys = set()
        sources = row.get("evidence_sources") or []
        if isinstance(sources, str):
            sources = [sources]
        unparsed = []
        for raw in sources:
            parsed = reference_ids(str(raw))
            if not parsed:
                unparsed.append(str(raw))
            for key in parsed:
                keys.add(key)
                entry = refs.setdefault(key, {"reference": key, "row_ids": set(), "raw_examples": [], "concerns": set(), "priority": 2})
                entry["row_ids"].add(row["id"])
                entry["concerns"].add(row["concern"])
                entry["priority"] = min(entry["priority"], row.get("review_priority", 2))
                if len(entry["raw_examples"]) < 3 and str(raw) not in entry["raw_examples"]:
                    entry["raw_examples"].append(str(raw))
        by_row[row["id"]] = {"reference_ids": sorted(keys), "unparsed_sources": unparsed}
    for entry in refs.values():
        entry["row_ids"] = sorted(entry["row_ids"])
        entry["concerns"] = sorted(entry["concerns"])
        entry["row_count"] = len(entry["row_ids"])
    return refs, by_row


def row_status(keys: list[str], cache: dict) -> dict:
    records = [cache.get(k, {"status": "not_looked_up"}) for k in keys]
    flags = sorted({f for r in records for f in r.get("review_flags", [])})
    return {"statuses": dict(collections.Counter(r["status"] for r in records)), "review_flags": flags,
            "all_identifiers_resolved": bool(keys) and all(r["status"] == "metadata_resolved" for r in records),
            "efficacy_approved": False, "expert_review_status": "pending"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/sllm_v3")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resolve", type=int, default=0, help="빈출 출처부터 네트워크 조회할 식별자 수. 0은 캐시만 사용")
    args = ap.parse_args(argv)
    if args.resolve < 0:
        ap.error("--resolve must be nonnegative")
    rows = read_rows(args.data / "review_queue.jsonl") + read_rows(args.data / "quarantine.jsonl")
    refs, by_row = inventory(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "reference_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for key in refs:
        if key.startswith("placeholder:") or key == "doi:10.1234/example":
            cache[key] = resolve_reference(key)
    ordered = sorted((k for k in refs if not k.startswith("placeholder:") and k != "doi:10.1234/example"),
                     key=lambda k: (-refs[k]["row_count"], refs[k]["priority"], k))
    selected = ordered[:args.resolve]
    missing = [k for k in selected if k not in cache or cache[k]["status"] == "lookup_error"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(resolve_reference, missing), 1):
            cache[result["reference"]] = result
            # 중단 후에는 조회에 성공한 결과를 캐시에서 재사용한다.
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            if i % 10 == 0:
                print(f"[references] {i}/{len(missing)}", flush=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    entries = [{**refs[k], "resolution": cache.get(k, {"status": "not_looked_up", "claim_status": "not_expert_verified"})} for k in sorted(refs)]
    reviewed_rows = [{**r, **by_row[r["id"]], "bibliography_review": row_status(by_row[r["id"]]["reference_ids"], cache)} for r in rows]
    for name, values in (("reference_inventory", entries), ("claim_review_queue", reviewed_rows)):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    summary = {"version": VERSION, "source_rows": len(rows), "unique_identifiers": len(refs),
               "resolution_budget": args.resolve, "identifier_statuses": dict(collections.Counter(e["resolution"]["status"] for e in entries)),
               "rows_with_any_resolved_reference": sum(any(cache.get(k, {}).get("status") == "metadata_resolved" for k in r["reference_ids"]) for r in reviewed_rows),
               "rows_with_all_identifiers_resolved": sum(r["bibliography_review"]["all_identifiers_resolved"] for r in reviewed_rows),
               "rows_with_placeholder": sum("placeholder" in r["bibliography_review"]["statuses"] for r in reviewed_rows),
               "rows_with_known_unrelated_reference": sum("known_unrelated_population_declaration" in r["bibliography_review"]["review_flags"] for r in reviewed_rows),
               "rows_with_non_efficacy_study_title": sum("non_efficacy_study_title_review" in r["bibliography_review"]["review_flags"] for r in reviewed_rows),
               "rows_with_preclinical_title": sum("preclinical_title_human_extrapolation_review" in r["bibliography_review"]["review_flags"] for r in reviewed_rows),
               "rows_without_parseable_reference": sum(not r["reference_ids"] for r in reviewed_rows),
               "efficacy_approved_rows": 0, "expert_review_status": "pending",
               "source_sha256": {f: hashlib.sha256((args.data / f).read_bytes()).hexdigest() for f in ("review_queue.jsonl", "quarantine.jsonl")}}
    (args.out / "evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
