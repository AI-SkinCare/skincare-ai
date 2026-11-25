from contextlib import closing
import sqlite3

import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.knowledge import DEFAULT_DB, KnowledgeQuery, apply_selection, retrieve
from server.main import create_app


@pytest.mark.parametrize("question,expected", [
    ("건선일 때 샤워하는 법", "AAD_PS"), ("지루성피부염에는 어떤 세정이 좋을까", "AAD_SD"),
    ("소아 아토피 때문에 달걀을 제한할까", "ECZEMA_DIET"), ("주사와 뜨거운 음료", "ROS_DIET"),
    ("여드름과 유제품의 관계", "ACNE_DIET"), ("식품 나트륨 비교", "KR_LABEL"),
    ("AHA 화장품을 쓰면 햇빛에 민감해질까", "AHA_SUN"), ("세라마이드 연구 한계", "CER_RCT"),
    ("향료 알레르기 성분표", "FDA_FRAGRANCE"), ("매일 편안함 변화량", "DIARY"),
    ("지성 얼굴 기름종이", "OILY"), ("샴푸로 비듬을 관리하는 방법", "DANDRUFF"),
])
def test_cited_first_result(question, expected):
    result = retrieve(DEFAULT_DB, KnowledgeQuery(query=question))
    assert result.answerSourceId == expected
    assert result.answer == result.sources[0].text


def test_database_scope_and_abstention():
    with closing(sqlite3.connect(DEFAULT_DB)) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone() == db.execute("SELECT COUNT(*) FROM search").fetchone()
    for kind, prefix in [("research", "PMID-"), ("label", "SPL-")]:
        result = retrieve(DEFAULT_DB, KnowledgeQuery(query="징크옥사이드", kind=kind))
        assert result.answerSourceId is None
        assert any(s.id.startswith(prefix) for s in result.sources)
    for query in ["이번 주말 기차 시간표", "여드름 약을 몇 mg 복용할까"]:
        result = retrieve(DEFAULT_DB, KnowledgeQuery(query=query))
        assert result.origin == "no_match" and not result.sources


def test_model_can_only_select_a_returned_passage():
    result = retrieve(DEFAULT_DB, KnowledgeQuery(query="여드름 식단"))
    assert apply_selection(result, 99) == result
    abstained = apply_selection(result, 0)
    assert abstained.answerSourceId is None and abstained.sources == result.sources
    selected = apply_selection(result, 1)
    assert selected.answer == selected.sources[0].text and selected.origin == "model_selected"


def test_search_contract_and_auth():
    app = create_app(Settings(internal_token="knowledge-test", rec_backend="mock", skin_backend="stub"))
    headers = {"X-Internal-Token":"knowledge-test"}
    with TestClient(app) as client:
        assert client.post("/knowledge/search", json={"query":"세라마이드"}).status_code == 401
        for invalid in [{"query":" "}, {"query":"x"*161}, {"query":"세안","kind":"all"}]:
            assert client.post("/knowledge/search", headers=headers, json=invalid).status_code == 422
        result = client.post("/knowledge/search", headers=headers, json={"query":"세라마이드"}).json()
        assert result["answerSourceId"] == "CER_RCT"
        assert result["sources"][0]["url"].startswith("https://")
