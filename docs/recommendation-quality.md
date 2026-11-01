# 성분 추천 품질 처리와 팀 연동

2026-10-21. DenseNet 분류 → Qwen2.5 성분 추천 경로의 후처리 규약이다.
실측 결과는 별도 결과 보고서에 기록하며, 아래 상태는 임상적 효능 판정이 아니다.

## 응답 처리

성분명은 기존 사전의 교정된 별칭으로 추출한다. 한글·영문 및 띄어쓰기를
대조하고, 단순한 나열보다 해당 성분을 설명하는 문장을 우선한다.
서로 다른 식물이나 부위의 성분을 반복이라는 이유로 하나로 합치지 않는다.
설명이 없거나 어느 성분의 설명인지 판단할 수 없으면 `purpose: null`을 유지한다.
효능을 추정해 채우거나 다른 성분의 설명을 복사하지 않는다.

`ingredients[]`에는 다음 필드가 추가된다. 기존 `name`, `purpose`, `priority`는 유지한다.

| 필드 | 값 | 프론트 처리 |
|---|---|---|
| `purpose_status` | `provided`, `unavailable` | unavailable이면 설명 영역에 ‘설명 정보 부족’ 표시 |
| `purpose_source` | `model`, `model_shared`, `unavailable` | model_shared는 모델이 성분 묶음에 공통으로 쓴 설명 |

`model` 및 `model_shared`는 생성 문장 출처이며 과학적 효능 검증을 뜻하지 않는다.

`recommendation`에는 다음 필드가 추가된다.

| 필드 | 값 / 의미 |
|---|---|
| `quality_status` | `ok`: 설정된 검사에서 문제 미검출, `limited`: 누락·문장 제거 등 제한, `withheld`: 제공 보류 |
| `quality_issues` | 기계 판독 가능한 사유 코드 목록 |
| `postprocess_version` | 출력 처리 버전; 모델 버전과 별도로 저장 |

분류 경로의 `recommendation_status`에 **`quality_rejected`**가 추가된다.
필터 이후 성분, 실행 가능한 루틴, 답변 중 하나라도 남지 않으면
`recommendation: null`, `recommendation_status: "quality_rejected"`로 반환한다.
최상위 `quality_issues`와 `guidance`로 사유를 전달한다. 기존 `uncertain`
(분류 불확실), `no_concern`(정상·선택 고민 없음), `generated`는 유지한다.

Swift 상태 분기에 `quality_rejected`를 추가하고 추천 카드 대신 `guidance`를
보여준다. Spring Boot는 해당 응답을 모델 장애나 HTTP 재시도 대상으로
취급하지 않고 보류 상태로 저장한다. 자동 재요청을 반복하면 GPU 대기열만 늘어난다.
`/infer/recommend` 단독 경로는 기존 응답 형태를 유지하므로
`quality_status == "withheld"`를 확인해야 한다.

예시:

```json
{
  "recommendation": null,
  "recommendation_status": "quality_rejected",
  "quality_issues": ["no_supported_ingredients"],
  "guidance": ["생성 결과가 추천 품질 기준을 충족하지 못해 성분 추천을 보류했습니다."]
}
```

실제 응답에는 기존 `disease` 객체도 포함한다. 위 예시는 추가 상태를 설명하기 위한 발췌다.

## 생성과 검사의 범위

서빙 전용 시스템 지침은 성분별 설명, 간결한 관리법, 입력에 근거한 표현을
요청한다. 학습·일반 비교평가의 기존 프롬프트는 바꾸지 않는다. 따라서
새 서빙 지침을 쓴 결과는 과거 학습 평가 점수와 직접 비교하지 않는다.

생성 원문에서 강한 효능·안전 보장, 근거 없는 개인 원인·병변 부위 단정,
민감성 문맥에서의 각질 제거 권유 등을 검사한 뒤 일반 표현 필터를 적용한다.
삭제 후의 동일한 문장 집합으로 성분·루틴·답변을 구성하여 제거된 문구가
다른 응답 필드에 다시 나타나지 않도록 한다. 부정·회피 문장과 긍정 권유를
구분하는 회귀 테스트가 포함되지만 문자열 검사로 모든 표현을 판정할 수는 없다.

효능 문장이 삭제되어도 남아 있는 실제 루틴에서 이름을 명시한 성분은 목록에
포함하되 설명을 복원하지 않는다. 같은 성분의 사용·회피가 모순되거나 후보가
5개를 넘어서 목록과 본문이 어긋나면 `ingredient_reference_inconsistent`로
보류한다. 생성 JSON의 세 단계가 불완전하면 `invalid_reasoning_structure`로 보류한다.

설문 `note`에 명시한 민감성·따가움도 보수적인 관리 제약 검사에 반영한다.
따라서 분류가 여드름이어도 사용자가 민감하다고 적은 경우 각질 제거 권유를
그대로 통과시키지 않는다. 이 처리는 질환을 새로 진단하거나 분류 라벨을 바꾸지 않는다.
쑥과 용담을 대나무의 별칭처럼 병기하는 등 확인된 이름 혼동은
`ingredient_identity_conflict`로 응답 전체를 보류한다.
민감성 제약이 있는데 성분 선택 근거 자체가 각질 제거 효능인 경우에는 문장만
지워 같은 성분을 계속 권하지 않고 `restricted_ingredient_purpose`로 보류한다.

이 보수적 관리 정책은 기존 프로젝트 제약과
[AAD 아토피 피부 관리](https://www.aad.org/public/diseases/eczema/types/atopic-dermatitis/atopic-dermatitis-coping),
[AAD 주사 피부 관리](https://www.aad.org/public/diseases/rosacea/triggers/tips)를 참고한다.
이 자료는 Qwen이 제안하는 개별 식물 추출물의 효능을 승인하는 근거가 아니다.

규칙 폴백에서는 여러 고민을 번갈아 반영한다. 폴백은 계속 `used_fallback: true`로
구분하며 모델 자체 성능에 합산하지 않는다. 같은 입력에 임의로 다른 성분을
뽑아 다양성 점수를 높이지 않는다.

## 재현

레포 루트 PowerShell, 기존 `.venv`와 로컬 모델·데이터가 필요하다.

```powershell
$env:PYTHONUTF8='1'
$env:HF_HOME='D:/hf_cache'
$env:HF_HUB_CACHE='D:/hf_cache/hub'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'

# 저장된 96건의 이미 필터된 응답을 새 처리기로 재처리. 새 이미지 추론은 아니다.
.\.venv\Scripts\python.exe scripts/evaluate_recommendation_quality.py --mode replay --output-dir outputs/quality_replay_new

# 실제 모델: 분류 fixture 12건 + 원본 이미지 4건. 설문은 공통/변형 가상 설문.
.\.venv\Scripts\python.exe scripts/evaluate_recommendation_quality.py --mode live --adapter outputs/rec_sllm_v2/adapter --image-root 'C:/Users/User/Downloads/test_sample_96/test_sample_96' --output-dir outputs/quality_live_new

.\.venv\Scripts\python.exe -m pytest -q
```

평가 도구는 실제 FastAPI 앱에 TestClient로 요청하고 실제 GPU Qwen과 CPU
DenseNet을 실행한다. 외부 네트워크·Spring·Swift 지연을 측정하는 부하 시험은 아니다.
사진을 파일로 복사하거나 전송하지 않는다. 결과는 `outputs/`에만 저장한다.

실제 생성 이후 순수 후처리만 보강한 경우 `reprocess_recommendation_trials.py`로
저장한 원문 전체를 다시 검사한다. `responses.jsonl`은 최초 실제 API 기록으로 보존하고,
최종 처리 결과는 `reprocessed.jsonl`에 저장한다. 재처리 결과를 새 GPU 추론으로
세지 않으며, 최종 API 경로는 `--cases` 옵션의 추가 실모델 회귀 시험으로 확인한다.

## 추가 정답이 필요한 작업

v3 정제 데이터에서 민감성은 고유 학습 사례 7건이다. 입력이 비슷할 때 반복되는
성분군은 데이터 분포와 질환→고민 매핑의 영향도 받는다. 성분 추출기만 고쳐서
개인화 성능이 검증되었다고 볼 수 없다.

기존 `data/review_20261006/`의 검토 목록에는 출처, 희소·복합 고민, 외부 평가
양식이 있다. 전문가가 검토한 성분·효능·사용 조건 정답과 독립된 새 사용자 사례가
준비되기 전에는 생성 응답을 새 정답으로 승격하거나 임상 정확도를 계산하지 않는다.
기존 144건 개발 비교에서 v2와 v3의 성분 점수 차이는 작아, v3라는 이름만으로
서비스 기본 모델을 교체하지 않는다.
