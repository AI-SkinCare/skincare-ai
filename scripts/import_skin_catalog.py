"""Build the optional Korean reference catalog from native provider exports."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data/knowledge-inbox"
CATALOG = ROOT / "data/skin-reference.sqlite3"
MFDS = "https://nedrug.mfds.go.kr/pbp/"
NUTRITION = "https://www.data.go.kr/data/15100064/standard.do"
AIHUB = "https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71961&topMenu=100"
EXPORTS = {"ingredients": ("CCBDE01", "ingredients.xlsx"),
           "reviewed-products": ("CCBDB01", "products.xlsx")}
FORMATS = {"ingredients": {".xlsx"}, "reviewed-products": {".xlsx"},
           "nutrition": {".csv"}, "makeup": {".json"}}


def clean(value):
    return "" if value is None else " ".join(str(value).split())


def identifier(prefix, *parts):
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:32]
    return prefix + digest


def sheet(path):
    from openpyxl import load_workbook
    with closing(load_workbook(path, read_only=True, data_only=True)) as workbook:
        tab = workbook.active
        # The provider writes a nominal A1 dimension even in full exports.
        tab.reset_dimensions()
        values = iter(tab.values)
        headers = [clean(v) for v in next(values)]
        for row in values:
            if any(v is not None for v in row):
                yield dict(zip(headers, map(clean, row)))


def cosmetic_rows(group, path):
    for r in sheet(path):
        if group == "ingredients":
            name, english, cas = r["표준명"], r["영문명"], r["CASNo"]
            if not name:
                raise ValueError("원료 표준명이 없습니다")
            body = [f"식약처 원료 표준명은 {name}입니다."]
            if english:
                body.append(f"영문명 · {english}.")
            if cas:
                body.append(f"CAS 번호 · {cas}.")
            url = MFDS + "CCBDE01/getList?" + urlencode(
                {"searchYn": "true", "searchCondition": "ingrKorName", "searchValue": name})
            yield (identifier("KR-ING-", name), name, english + " " + cas,
                   " ".join(body), "식품의약품안전처 · 화장품 원료", url, "")
        else:
            seq, name = r["품목일련번호"], r["제품명"]
            if not seq.isdecimal() or not name:
                raise ValueError("유효한 심사 품목번호와 제품명이 필요합니다")
            body = f"품목일련번호 {seq} · 업체 {r['업체명']} · 심사번호 {r['심사번호']}."
            yield ("KR-COS-" + seq, name, r["업체명"] + " " + seq, body,
                   "식품의약품안전처 · 기능성화장품 심사", MFDS + "CCBDB01/getItem?itemSeq=" + seq, r["심사일자"])


NUTRIENTS = {"에너지(kcal)": "열량", "단백질(g)": "단백질", "탄수화물(g)": "탄수화물",
             "당류(g)": "당류", "지방(g)": "지방", "나트륨(mg)": "나트륨", "식이섬유(g)": "식이섬유"}


def food_rows(path):
    with path.open("rb") as raw:
        first_line = raw.readline()
    try:
        first_line.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "cp949"
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [re.sub(r"\s+", "", k) for k in reader.fieldnames or []]
        if not set(NUTRIENTS).intersection(reader.fieldnames):
            raise ValueError("영양소 열을 찾지 못했습니다")
        for r in reader:
            code, title, basis = (clean(r[k]) for k in ("식품코드", "식품명", "영양성분함량기준량"))
            if not code or not title or not basis:
                raise ValueError("식품코드·식품명·함량 기준량이 필요합니다")
            facts = []
            for key, label in NUTRIENTS.items():
                value = clean(r.get(key))
                if re.fullmatch(r"\d+(?:\.\d+)?", value):
                    unit = key.split("(")[1][:-1]
                    facts.append(f"{label} {value} {unit}")
            if facts:
                maker = clean(r.get("업체명"))
                source = clean(r.get("출처명")) or "전국통합식품영양성분정보"
                detail = f"{title}" + (f" ({maker})" if maker else "")
                yield ("KR-FOOD-" + code, detail, title, f"{detail}의 {basis} 기준 " + ", ".join(facts) + f". 식품코드 {code}.",
                       source, NUTRITION, clean(r.get("데이터생성일자")))


def makeup_rows(path):
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict) or not isinstance(obj.get("Annotation_info"), dict) or "Source_info1" not in obj:
        raise ValueError("AI Hub 71961 질의응답 라벨 JSON이 필요합니다")
    # Take topics only from the cited source passage, never from the generated recommendation.
    topics = ("메이크업", "색조", "컨실러", "파운데이션", "자외선", "세안", "클렌징", "여드름", "홍조",
              "아토피", "피부염", "색소침착", "보습", "피지", "피부장벽", "논코메도제닉", "세라마이드",
              "나이아신아마이드", "살리실산", "판테놀", "히알루론산", "글리세린", "향료")
    for i in range(1, 6):
        ref = obj.get(f"Source_info{i}") or {}
        title = clean(ref.get(f"Title{i}"))
        if not title:
            continue
        author = clean(ref.get(f"Author Name{i}"))
        publisher = clean(ref.get(f"Publisher Name (Institution Name){i}"))
        year = clean(ref.get(f"Year of Publication{i}"))
        year = year if re.fullmatch(r"[12]\d{3}", year) else ""
        details = " · ".join(v for v in (author, publisher, year) if v)
        text = "AI Hub 라벨에 기재된 참고문헌입니다. 링크는 데이터셋 설명으로 연결됩니다."
        passage = clean(ref.get(f"Knowledge Data{i}")).replace(" ", "")
        aliases = " ".join(sorted(set(author.split()) | {t for t in topics if t in passage}))
        yield (identifier("AIHUB-REF-", title, author, publisher, year), title, aliases,
               (details + ". " if details else "") + text, "AI Hub · 문제성 피부 메이크업 참고문헌", AIHUB, year)


def download_mfds(inbox):
    for group, (route, filename) in EXPORTS.items():
        folder = inbox / group
        folder.mkdir(parents=True, exist_ok=True)
        request = Request(MFDS + route + "/getExcel", data=b"searchYn=true",
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".xlsx", delete=False) as tmp:
            temp = Path(tmp.name)
        try:
            with urlopen(request, timeout=120) as response, temp.open("wb") as target:
                shutil.copyfileobj(response, target)
            with closing(cosmetic_rows(group, temp)) as rows:
                next(rows)  # verify official columns before replacing the input
            os.replace(temp, folder / filename)
        finally:
            temp.unlink(missing_ok=True)


def download_nutrition(inbox):
    """Use the same metadata and paged records as the official CSV export button."""
    base = "https://www.data.go.kr/download/"
    with urlopen(base + "columList.json?pk=15100064&ext=CSV", timeout=40) as response:
        metadata = json.load(response)
    columns = metadata["columList"]
    codes = [column["columCode"] for column in columns]
    folder = inbox / "nutrition"
    folder.mkdir(parents=True, exist_ok=True)
    size = 2000
    total = int(metadata["totalCount"])
    if total < 1:
        raise ValueError("공공데이터포털에 표시된 식품 레코드가 없습니다")
    params = {"publicDataPk": "15100064", "svcTableNm": metadata["tableVO"]["svcTableNm"],
              "colNmList": metadata["tableVO"]["colNmList"], "totalCount": total, "perPage": size}
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([c["columNm"] for c in columns])
            for offset in range(0, total, size):
                url = base + "standard.json?" + urlencode(dict(params, page=offset // size + 1), doseq=True)
                with urlopen(url, timeout=100) as response:
                    page = json.load(response)
                if len(page) != min(size, total - offset):
                    raise ValueError("공식 CSV 페이지의 실제 건수가 예상과 다릅니다")
                for item in page:
                    writer.writerow([item.get(code) for code in codes])
        os.replace(temp, folder / "nutrition.csv")
    finally:
        temp.unlink(missing_ok=True)


def build_catalog(inbox, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".sqlite3", delete=False) as f:
        temp = Path(f.name)
    counts = Counter()
    try:
        with closing(sqlite3.connect(temp)) as db:
            db.executescript("""
                CREATE TABLE docs(id TEXT PRIMARY KEY, title TEXT, aliases TEXT, text TEXT,
                    publisher TEXT, url TEXT, date TEXT, section TEXT);
                CREATE VIRTUAL TABLE names USING fts5(title,aliases,content='docs',content_rowid='rowid');
                CREATE TABLE metadata(as_of TEXT);
            """)
            for group, suffixes in FORMATS.items():
                folder = inbox / group
                files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("~$"))
                for path in files:
                    if path.suffix.lower() not in suffixes:
                        raise ValueError(f"{path.name}: {group} 폴더는 {sorted(suffixes)} 원본 파일을 받습니다. 압축은 먼저 풀어주세요.")
                    if group == "nutrition":
                        rows = food_rows(path)
                    elif group == "makeup":
                        rows = makeup_rows(path)
                    else:
                        rows = cosmetic_rows(group, path)
                    try:
                        for row in rows:
                            record = (*row, group)
                            try:
                                db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)", record)
                                counts[group] += 1
                            except sqlite3.IntegrityError:
                                existing = db.execute("SELECT * FROM docs WHERE id=?", (row[0],)).fetchone()
                                if group == "makeup":
                                    combined = " ".join(sorted(set(existing[2].split()) | set(record[2].split())))
                                    db.execute("UPDATE docs SET aliases=? WHERE id=?", (combined, row[0]))
                                elif existing != record:
                                    raise ValueError(f"같은 ID의 값이 다릅니다: {row[0]}. 최신 원본만 남겨 주세요.")
                    except (KeyError, ValueError) as error:
                        raise ValueError(f"{path.name}: {error}") from error
                if files and not counts[group]:
                    raise ValueError(f"{group}: 색인할 레코드가 없습니다")
            if not counts:
                raise ValueError(f"원본 파일이 없습니다: {inbox}")
            if output.exists():
                with closing(sqlite3.connect(output.resolve().as_uri() + "?mode=ro", uri=True)) as previous:
                    missing = {r[0] for r in previous.execute("SELECT DISTINCT section FROM docs")} - counts.keys()
                if missing:
                    raise ValueError(f"기존 자료 폴더가 비었습니다: {sorted(missing)}")
            db.execute("INSERT INTO names(names) VALUES ('rebuild')")
            db.execute("INSERT INTO names(names,rank) VALUES ('integrity-check',1)")
            db.execute("INSERT INTO metadata VALUES (?)", (date.today().isoformat(),))
            db.commit()
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("색인 무결성 검사 실패")
        os.replace(temp, output)
        return counts
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=INBOX)
    parser.add_argument("--output", type=Path, default=CATALOG)
    parser.add_argument("--download-mfds", action="store_true", help="Refresh both public MFDS spreadsheets first")
    parser.add_argument("--download-nutrition", action="store_true", help="Refresh the public Korean nutrition snapshot")
    args = parser.parse_args()
    if args.download_mfds:
        download_mfds(args.inbox)
    if args.download_nutrition:
        download_nutrition(args.inbox)
    print(dict(build_catalog(args.inbox, args.output)))
