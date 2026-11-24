"""Build the offline search index from curated guidance and fixed public API queries."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "configs/skin_knowledge.sqlite3"
# These terms are used only for discovering study records, never as efficacy evidence.
STUDIES = [
    ("acne", "diet OR dairy OR glycemic", "여드름 식습관 식단 우유 유제품 혈당"),
    ("rosacea", "diet OR trigger OR skincare", "주사 홍조 식습관 유발요인 관리"),
    ('"atopic dermatitis"', "moisturizer OR emollient", "아토피 보습제 장벽"),
    ("psoriasis", "diet OR emollient", "건선 식단 보습"),
    ('"seborrheic dermatitis"', "skincare OR shampoo OR moisturizer", "지루 지루성피부염 세정 샴푸"),
    ("ceramide", "moisturizer OR epidermal", "세라마이드 보습 장벽"),
    ("niacinamide", "topical OR moisturizer", "나이아신아마이드 니아신아마이드 외용"),
    ('"hyaluronic acid"', "topical AND hydration", "히알루론산 보습 수분"),
    ("glycerol OR glycerin", "emollient OR skin hydration", "글리세린 글리세롤 보습"),
    ("sunscreen", "zinc oxide OR titanium dioxide", "선크림 자외선 산화아연 징크옥사이드 티타늄디옥사이드"),
    ("panthenol OR dexpanthenol", "topical AND skin", "판테놀 덱스판테놀 피부장벽"),
    ("retinol", "topical AND skin", "레티놀 레티노이드 피부"),
    ("azelaic acid", "acne OR rosacea", "아젤라산 여드름 주사"),
    ("salicylic acid", "acne AND topical", "살리실산 BHA 여드름"),
]


def fetch(endpoint, params):
    url = endpoint + "?" + urlencode(params)
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "SkinCare-Knowledge/1.0", "Accept": "application/json"})
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def study_records():
    papers = {}
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    for topic, focus, tags in STUDIES:
        topics = " OR ".join(term + "[ti]" for term in topic.split(" OR "))
        focuses = "".join(term if term in (" OR ", " AND ") else term + "[tiab]"
                          for term in re.split(r"( OR | AND )", focus))
        query = (f'({topics}) AND ({focuses}) AND (randomized controlled trial[pt] OR systematic review[pt] OR clinical trial[pt]) '
                 'NOT (retracted publication[pt] OR retraction of publication[pt] OR (animals[mh] NOT humans[mh]) '
                 'OR dogs[ti] OR canine[ti] OR feline[ti] OR murine[ti] OR mice[ti])')
        hits = fetch(endpoint + "esearch.fcgi", {"db":"pubmed","term":query,"retmode":"json",
                                                "retmax":60,"sort":"relevance"})["esearchresult"]["idlist"]
        if not hits:
            continue
        time.sleep(.45)
        data = fetch(endpoint + "esummary.fcgi", {"db":"pubmed","id":",".join(hits),"retmode":"json"})["result"]
        for pmid in hits:
            item = data[pmid]
            title = html.unescape(re.sub("<[^>]+>", "", item.get("title", ""))).strip()
            if not pmid.isdigit() or not title:
                continue
            if pmid not in papers:
                journal = item.get("fulljournalname") or item.get("source") or "학술지"
                year = item.get("pubdate", "")
                design = ", ".join(t for t in item.get("pubtype", []) if t != "Journal Article")
                description = f"{journal} · {year}." + (f" PubMed 분류 · {design}." if design else "")
                papers[pmid] = dict(id="PMID-" + pmid, kind="research", title=title,
                    publisher="미국 국립의학도서관 · PubMed", url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    text=description + " 연구 대상과 결과는 원문에서 확인하세요.",
                    tags="", date=year)
            papers[pmid]["tags"] += " " + tags
        time.sleep(.45)
    if not papers:
        raise ValueError("No valid study records returned")
    return list(papers.values())


def product_records():
    records = {}
    for substance in ("ZINC OXIDE", "TITANIUM DIOXIDE"):
        query = f'openfda.product_type:"HUMAN OTC DRUG" AND openfda.substance_name:"{substance}" AND purpose:sunscreen'
        data = fetch("https://api.fda.gov/drug/label.json", {"search": query, "limit": 100, "sort": "effective_time:desc"})
        for item in data["results"]:
            meta = item.get("openfda", {})
            if not meta.get("brand_name") or not meta.get("substance_name") or not item.get("set_id"):
                continue
            set_id = str(UUID(item["set_id"]))
            brand = meta["brand_name"][0].strip()
            active = sorted(set(s.strip().lower() for s in meta["substance_name"] if s.strip()))
            tags = "선크림 선스크린 자외선차단제 제품 라벨"
            if "zinc oxide" in active:
                tags += " 산화아연 징크옥사이드"
            if "titanium dioxide" in active:
                tags += " 이산화티타늄 티타늄디옥사이드"
            if not brand or not {"zinc oxide", "titanium dioxide"}.intersection(active):
                continue
            row = dict(id="SPL-" + set_id, kind="label", title=brand, publisher="미국 제품 라벨 · openFDA",
                url="https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=" + set_id,
                text="미국 제출 라벨의 활성 성분 · " + ", ".join(active) + ". 국내 판매 제형과 회피 성분은 실제 포장의 전성분에서 확인하세요.",
                tags=tags + " " + " ".join(active), date=item["effective_time"])
            if set_id not in records or row["date"] > records[set_id]["date"]:
                records[set_id] = row
    if not records:
        raise ValueError("No valid OTC sunscreen labels returned")
    return list(records.values())


def local_records():
    guides = json.loads((ROOT / "configs/knowledge_guides.json").read_text(encoding="utf-8"))
    store = json.loads((ROOT / "configs/ingredient_evidence.json").read_text(encoding="utf-8"))
    topics = {
        "AAD_AD": "아토피 아토피피부염 습진 목욕 보습 atopic eczema",
        "AAD_PS": "건선 목욕 샤워 보습 psoriasis",
        "AAD_SD": "지루 지루성 지루성피부염 세정 샴푸 seborrheic",
        "AAD_ROS": "주사 홍조 민감성 rosacea",
        "AAD_PET": "페트롤라툼 바셀린 petrolatum vaseline",
        "AAD_DRY": "건조 건성 무향 보습 보습제",
    }
    for sid, ref in store["sources"].items():
        claims = [c for c in store["claims"] if sid in c["sources"]]
        # Index what this source itself covers, not every ingredient on a multi-source card.
        tags = topics.get(sid, "") + " " + " ".join(a for c in claims
            if any(alias.lower() in ref["summary"].lower() for alias in c["aliases"]) for a in c["aliases"])
        guides.append(dict(id=sid, kind="guide", tags=tags, title=ref["title"], text=ref["summary"],
            publisher="미국피부과학회" if sid.startswith("AAD_") else "PubMed", url=ref["url"], date=""))
    return guides


def build(path, rows, as_of):
    rows = sorted(rows, key=lambda r: r["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".sqlite3", delete=False) as f:
        temp = Path(f.name)
    try:
        db = sqlite3.connect(temp)
        try:
            db.executescript("""
                CREATE TABLE docs(id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
                    publisher TEXT NOT NULL, url TEXT NOT NULL, text TEXT NOT NULL, tags TEXT NOT NULL, date TEXT NOT NULL);
                CREATE TABLE metadata(as_of TEXT NOT NULL, schema_version INTEGER NOT NULL);
                CREATE VIRTUAL TABLE search USING fts5(title, tags, text, tokenize='unicode61');
            """)
            db.execute("INSERT INTO metadata VALUES (?,1)", (as_of,))
            for r in rows:
                if not r["title"].strip() or not r["text"].strip() or not r["url"].startswith("https://"):
                    raise ValueError(f"Invalid source {r['id']}")
                cursor = db.execute("INSERT INTO docs VALUES (:id,:kind,:title,:publisher,:url,:text,:tags,:date)", r)
                db.execute("INSERT INTO search(rowid,title,tags,text) VALUES (?,?,?,?)",
                           (cursor.lastrowid, r["title"], " ".join(dict.fromkeys(r["tags"].split())), r["text"]))
            db.commit()
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Invalid database")
        finally:
            db.close()
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_DB)
    p.add_argument("--local-only", action="store_true", help="Re-index edited local guides using the existing public snapshot")
    args = p.parse_args()
    rows = local_records()
    if args.local_only:
        with closing(sqlite3.connect(args.output.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            as_of = db.execute("SELECT as_of FROM metadata WHERE schema_version=1").fetchone()[0]
            rows.extend(dict(r) for r in db.execute("SELECT * FROM docs WHERE id LIKE 'PMID-%' OR id LIKE 'SPL-%'"))
    else:
        rows.extend(study_records())
        rows.extend(product_records())
        as_of = datetime.now(timezone.utc).date().isoformat()
    build(args.output, rows, as_of)
    print(dict(Counter(r["kind"] for r in rows)))
