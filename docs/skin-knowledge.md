# 피부 자료 검색

앱 홈에서 **피부 자료 검색**을 열고 관리 안내, 연구 문헌, 제품 라벨 중 하나를 선택한다. 사진이나 설문 제출 없이 질문만으로 자료를 찾을 수 있다. 검색어와 개인의 사진·설문·분석 이력은 자료 DB에 저장하지 않는다.

| 검색 범위 | 자료와 제공 방식 |
|---|---|
| 관리 안내 | 국내 식품의약품안전처, WHO, 미국피부과학회, 미국 FDA와 기존 성분 연구의 한국어 요약을 검색한다. 피부 기록의 점수 계산은 앱 코드에 연결한다. |
| 연구 문헌 | PubMed에서 질환·성분별 14가지 검색식을 실행해 임상시험과 체계적 문헌고찰의 제목, 학술지, 발행일, 연구 유형, PMID를 수집한다. 제목을 누르면 연구 대상과 결과를 원문에서 확인할 수 있다. |
| 제품 라벨 | 미국 openFDA에 제출된 산화아연·이산화티타늄 함유 일반의약품 자외선차단 라벨과 제조사의 성분표 예시를 검색한다. 미국 라벨의 활성 성분만으로 개인 알레르기, 국내 제형 또는 구매 적합성을 판정할 수 없다. |

Spring은 사용자 인증 후 `POST /api/v1/knowledge/search`를 FastAPI의 `POST /knowledge/search`로 전달한다. 팀 서버 앞에 경로 허용 목록이 있다면 이 POST 경로에도 기존 내부 토큰 인증을 적용한다. 앱에는 내부 토큰이 들어가지 않는다.

```json
{"query":"아이 아토피 때문에 음식을 제한해야 할까요?", "kind":"guide"}
```

`kind`에는 `guide`, `research`, `label`을 쓴다. 검색어는 2~160자다. 응답의 `answerSourceId`는 실제 답변 문단의 `sources[].id`다. 문헌·제품 목록이나 답변 보류에서는 이 값이 비어 있다. `sources`에는 제목, 설명, 발행처, 원문 URL과 자료에 기록된 날짜가 있다. `asOf`는 DB 수집일이다.

SQLite FTS5가 한국어 검색어·성분 별칭·영문 제목으로 후보를 찾는다. 관리 안내에서는 서버의 Qwen2.5가 반환된 후보에서 질문에 답하는 문단을 고른다. 선택한 문단과 출처는 DB 원문대로 응답한다. 관련 문단이 질문에 답하지 못하면 답변을 보류한다. GPU가 바쁘거나 모델을 사용하지 않는 환경에서는 검색 순위와 자료 요약을 제공한다. 응답의 `origin`은 각각 `model_selected`, `no_match`, `retrieved`다. 이 검색 결과를 개인 성분 추천의 효능 근거로 자동 승인하지 않는다.

## DB 갱신과 검사

AI 저장소에서 다음을 실행한다. 기본 DB는 `configs/skin_knowledge.sqlite3`이며 서비스 요청 중 외부 API 연결은 필요하지 않다.

```powershell
python scripts/refresh_skin_knowledge.py
python -m pytest tests/test_skin_knowledge.py -q
```

갱신 작업은 공개 API에서 문헌·라벨을 다시 수집하고, PMID와 라벨 세트 ID로 중복을 정리한다. 새 인덱스의 무결성 검사가 끝나야 기존 파일을 교체한다. 직접 확인한 한국어 요약과 검색 별칭은 `configs/knowledge_guides.json`, 기존 성분 출처는 `configs/ingredient_evidence.json`에서 관리한다. 이번 스냅샷은 전체 PubMed 문헌이나 국내 판매 화장품 전수 목록을 대표하지 않는다. 공개 수집 인터페이스는 [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)와 [openFDA 제품 라벨](https://open.fda.gov/apis/drug/label/)을 사용한다.
