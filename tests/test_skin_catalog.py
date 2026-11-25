import csv
import json
import sqlite3
from contextlib import closing
from zipfile import ZipFile

import pytest
from openpyxl import Workbook

from scripts.import_skin_catalog import build_catalog
from server.knowledge import DEFAULT_DB, KnowledgeQuery, apply_selection, retrieve


def test_import_native_exports_and_query(tmp_path):
    inbox = tmp_path / "inbox"
    for folder in ("nutrition", "makeup", "ingredients", "reviewed-products"):
        (inbox / folder).mkdir(parents=True)
    for group, columns, values in [
        ("ingredients", ["표준명", "영문명", "CASNo"], ["예시원료", "ExampleExtract", "84012-19-1"]),
        ("reviewed-products", ["품목일련번호", "제품명", "업체명", "심사번호", "심사일자"],
         ["12345", "예시보습크림", "예시업체", "28", "2023-05-06"]),
    ]:
        book = Workbook()
        book.active.append(columns)
        book.active.append(values)
        book.save(inbox / group / "export.xlsx")
        book.close()
    with (inbox / "nutrition/food.csv").open("w", encoding="cp949", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["식품코드", "식품명", "영양성분함량 기준량", "당류(g)", "나트륨(mg)", "지방(g)"])
        writer.writerow(["TEST1", "예시두유", "100mL", "0", "18", ""])
        writer.writerow(["TEST2", "예시두유 라이트", "200mL", "3", "12", ""])
    label = dict(Source_info1={"Title1": "예시메이크업 문헌", "Year of Publication1": 2020},
                 Annotation_info={"Makeup Response": "GENERATED-PRIVATE"},
                 Human_info={"Gender": "SURVEY-PRIVATE"})
    for name in ("one", "two"):
        label["Source_info1"]["Knowledge Data1"] = "논코메도제닉" if name == "one" else "컨실러"
        (inbox / f"makeup/{name}.json").write_text(json.dumps(label), encoding="utf-8")
    dest = tmp_path / "catalog.sqlite3"
    counts = build_catalog(inbox, dest)
    assert counts == {"ingredients": 1, "reviewed-products": 1, "nutrition": 2, "makeup": 1}
    assert build_catalog(inbox, dest) == counts

    def search(kind, text):
        return retrieve(DEFAULT_DB, KnowledgeQuery(query=text, kind=kind), dest)

    assert search("label", "ExampleExtract").sources[0].title == "예시원료"
    assert search("label", "CAS 84012-19-1").sources[0].title == "예시원료"
    assert not search("label", "84012-19-2").sources
    assert search("label", "12345").sources[0].url.endswith("itemSeq=12345")
    food = search("guide", "예시두유의 당류").sources[0]
    assert "100mL" in food.text and "당류 0 g" in food.text and "지방" not in food.text
    multiple = search("guide", "예시두 당류")
    assert len(multiple.sources) == 2 and apply_selection(multiple, 1).answerSourceId is None
    assert search("guide", "예시두유가 여드름의 원인인가요?").answerSourceId != food.id
    assert search("guide", "가공식품 포장의 나트륨 수치를 어떻게 비교하나요?").answerSourceId == "KR_LABEL"
    ref = search("research", "예시메이크업").sources[0]
    assert ref.id.startswith("AIHUB-REF-") and "71961" in ref.url
    assert search("research", "컨실러").sources[0].id == ref.id
    assert search("research", "논코메도제닉").sources[0].id == ref.id
    assert search("guide", "약을 몇 mg 복용해요").origin == "no_match"
    with closing(sqlite3.connect(dest)) as db:
        assert "PRIVATE" not in str(db.execute("SELECT * FROM docs").fetchall())
    previous = dest.read_bytes()
    (inbox / "makeup/wrong.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="71961"):
        build_catalog(inbox, dest)
    assert dest.read_bytes() == previous


@pytest.mark.parametrize("query,expected", [
    ("면도날은 어느 방향으로 써요", "SHAVING"), ("메이크업 브러시 세척", "MAKEUP_BRUSH"),
    ("운동 중에 땀이 나면", "POST_WORKOUT"), ("갈라진 입술 립밤", "CHAPPED_LIPS"),
    ("아이가 아토피인데 수영장에 가도 될까요", "ECZEMA_SWIM"), ("색조 파운데이션", "ACNE_MAKEUP"),
])
def test_additional_care_topics(query, expected):
    assert retrieve(DEFAULT_DB, KnowledgeQuery(query=query)).answerSourceId == expected


def test_ceramide_search_omits_veterinary_trial():
    result = retrieve(DEFAULT_DB, KnowledgeQuery(query="세라마이드", kind="research"))
    assert any(s.id.startswith("PMID-") for s in result.sources)
    assert all("dogs with atopic dermatitis" not in s.title.lower() for s in result.sources)


def test_import_native_makeup_archives(tmp_path):
    inbox = tmp_path / "inbox"
    folder = inbox / "makeup"
    folder.mkdir(parents=True)
    citation = "예시 민감성 화장품 연구"
    label = {
        "Source_info1": {"Title1": "예시 민감성 화장품연구", "Knowledge Data1": "컨실러"},
        "Source_info2": {"Title2": "색조 블로그", "Knowledge Data2": "컨실러"},
        "Annotation_info": {"Makeup Response": "GENERATED-PRIVATE"},
    }
    with ZipFile(folder / "TL_example.zip", "w") as archive:
        archive.writestr("labels/sample.json", json.dumps(label))
    with ZipFile(folder / "Other.zip", "w") as archive:
        for i, (title, kind) in enumerate([
            (citation, "국내저널"),
            ("예시 민감성 화장품연구", "국내저널"),
            ("생쥐 피부 연구", "해외저널"),
            ("색조 블로그", "유튜브, 블로그"),
            ("예시 색조 도구에 관한 학위논문", "국내석사"),
        ]):
            data = {
                "Data_info": {"Title": title, "Author Name": "예시저자", "Publisher Name (Institution Name)": "예시학회"},
                "Source_info": {"Source": kind},
                "File_info": {},
                "Annotation_info": {"year": "2024", "Main Keywords": "파운데이션"},
            }
            archive.writestr(f"{i}.json", json.dumps(data))
        archive.writestr("sample.doc", "DOCUMENT-PRIVATE")
    dest = tmp_path / "catalog.sqlite3"
    assert build_catalog(inbox, dest) == {"makeup": 2}
    result = retrieve(DEFAULT_DB, KnowledgeQuery(query="컨실러", kind="research"), dest)
    assert result.sources[0].title == citation and "국내저널" in result.sources[0].text
    with closing(sqlite3.connect(dest)) as db:
        content = str(db.execute("SELECT title,text FROM docs").fetchall())
    assert "학위논문" in content
    for excluded in ("GENERATED-PRIVATE", "DOCUMENT-PRIVATE", "색조 블로그", "생쥐 피부 연구"):
        assert excluded not in content
