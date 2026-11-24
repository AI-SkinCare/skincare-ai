"""Offline, citation-preserving retrieval for the mobile information library."""
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated

Kind = Literal["guide", "research", "label"]
DEFAULT_DB = Path(__file__).resolve().parents[1] / "configs/skin_knowledge.sqlite3"
WORDS = re.compile(r"[가-힣]+|[a-z0-9]+")
STOP = set("피부 정보 알려주세요 알려줘 궁금해요 대한 어떻게 무엇 어떤 대해 있나요 추천 자료 the and for with what about".split())
PARTICLE = re.compile(r"(?:에(?:서|는|도|게)?|이|가|의|을|를|은|는|와|과|랑|로|으로|도|만|일|인|이라면|라면|인데|인가요|이랑|처럼|까지|한|해요|해서|하면|하는|하는데|하려면|할까요)")
CLINICAL = re.compile(r"처방|복용량|투약|진단해|진단해줘|몇\s*(?:mg|밀리그램|알|그램)|스테로이드.{0,12}(?:용량|얼마|몇)|prescri(?:be|ption)|dosage", re.I)


class KnowledgeQuery(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=160)]
    kind: Kind = "guide"
    limit: int = Field(default=4, ge=1, le=6)


class KnowledgeSource(BaseModel):
    id: str
    title: str
    text: str
    publisher: str
    url: str
    date: str


class KnowledgeAnswer(BaseModel):
    kind: Kind
    asOf: str
    origin: Literal["retrieved", "model_selected", "no_match"]
    answer: str
    answerSourceId: str | None = None
    sources: list[KnowledgeSource]


def words(query, vocabulary):
    tokens = WORDS.findall(unicodedata.normalize("NFKC", query).lower())
    terms = []
    for token in tokens:
        if len(token) < 2 or token in STOP:
            continue
        stems = [tag for tag in vocabulary if token.startswith(tag) and PARTICLE.fullmatch(token[len(tag):])]
        terms.append(max(stems, key=len) if stems else token)
    return list(dict.fromkeys(terms))[:18]


def retrieve(path: Path, request: KnowledgeQuery):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        meta = db.execute("SELECT as_of,schema_version FROM metadata").fetchone()
        if not meta or meta["schema_version"] != 1:
            raise sqlite3.DatabaseError("Unsupported knowledge database")
        stamp = meta["as_of"]
        if CLINICAL.search(request.query):
            return KnowledgeAnswer(kind=request.kind, asOf=stamp, origin="no_match", sources=[],
                answer="이 자료실은 일반 피부 관리 정보를 제공합니다. 개인의 진단이나 약물 용량은 의료진에게 확인해 주세요.")
        vocabulary = {t for row in db.execute("SELECT DISTINCT tags FROM docs WHERE kind=?", (request.kind,))
                      for t in row["tags"].lower().split() if len(t) > 1}
        terms = words(request.query, vocabulary)
        sources = []
        if terms:
            expression = " OR ".join('"' + term + '"*' for term in terms)
            sql = """SELECT d.id,d.title,d.text,d.publisher,d.url,d.date FROM search
                     JOIN docs d ON d.rowid=search.rowid WHERE search MATCH ? AND kind=?
                     ORDER BY bm25(search, 4, 8, 1), d.id LIMIT ?"""
            sources = [KnowledgeSource(**dict(row)) for row in db.execute(sql, (expression, request.kind, request.limit))]
    if not sources:
        return KnowledgeAnswer(kind=request.kind, asOf=stamp, origin="no_match", sources=[],
            answer="선택한 자료에서 질문에 맞는 근거를 찾지 못했어요. 성분명이나 다른 검색어로 찾아보세요.")
    first = sources[0]
    if request.kind == "guide":
        answer, source_id = first.text, first.id
    elif request.kind == "research":
        answer, source_id = "관련 연구의 서지 목록입니다. 제목을 눌러 연구 설계, 대상과 결과를 확인하세요.", None
    else:
        answer, source_id = "제조사 성분표와 미국 제품 라벨을 검색한 결과예요. 국내에서 구매하는 제품은 해당 포장의 전성분을 확인하세요.", None
    return KnowledgeAnswer(kind=request.kind, asOf=stamp, origin="retrieved", answer=answer,
                           answerSourceId=source_id, sources=sources)


def apply_selection(result, index):
    """Model chooses only an existing passage. Never relay generated medical text or URLs."""
    if index == 0:
        return result.model_copy(update=dict(origin="no_match", answerSourceId=None,
            answer="이 질문에 직접 답하는 문단을 찾지 못했어요. 아래에서 관련 자료를 살펴볼 수 있어요."))
    if isinstance(index, int) and 1 <= index <= len(result.sources):
        chosen = result.sources[index - 1]
        return result.model_copy(update=dict(origin="model_selected", answer=chosen.text, answerSourceId=chosen.id))
    return result
