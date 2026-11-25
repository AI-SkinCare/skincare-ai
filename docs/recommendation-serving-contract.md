# 추천 보류·출처 연동 계약

FastAPI → Spring → Swift가 같은 추천 상태를 전달하기 위한 서버 변경이다. 기존 미공개 연구 변경 중 추천 응답에 필요한 모듈만 포함한다. 분류 모델 재학습·실험 분류기·Grad-CAM·환자 데이터·가중치는 포함하지 않는다.

## 응답과 처리 순서

1. DenseNet 분류의 불확실성 또는 정상 무고민이면 추천을 호출하지 않고 `uncertain`/`no_concern`과 `generation_origin=not_generated`를 반환한다.
2. `avoidIngredients`로 근거 후보를 제외한다. 남은 근거가 없으면 생성하지 않고 `quality_rejected`, `recommendation=null`, `generation_origin=not_generated`, `quality_issues=["evidence_no_eligible_claims"]`를 반환한다.
3. 생성한 답변의 성분 근거, 분류 점수의 개인 상태 오해, 제품별 사용 조건, 출력 형식을 검사한다. 필요 시 제한된 교정 또는 근거 카드 대체를 사용하고 `model`/`source_cards`/`rule_fallback` 출처를 구별한다.
4. 생성 후 품질 미달로 추천이 null이 되더라도 최상위 `generation_origin`은 실제 생성 출처를 유지한다. `not_generated`는 recommendation_status가 아니다.

Spring은 보류를 `WITHHELD`로 저장하며, 상세 응답의 `generation.origin`, `qualityStatus`, `evidenceStatus`, `qualityIssues`로 전달한다. Swift는 보류 시 폴링을 종료하고 안내문을 보여준다. 생성하지 않은 응답을 모델 성공으로 표시하지 않는다.

## 설정과 배포

`.env.example`의 `REC_EVIDENCE_MODE=auto`는 실제 SLLM에서 근거 검사 경로를 활성화한다. 이 경로에서는 `REC_EVIDENCE_USE_ADAPTER=false`가 기본값이며 기존 Qwen2.5 베이스에 근거 문맥을 제공한다. 기존 v2 어댑터는 로드하되 이 생성 구간에서만 비활성화한다. 분류기는 기존 `DISEASE_BACKEND=densenet`을 유지한다.

`configs/ingredient_evidence.json`은 출처 URL과 한정된 효능·사용 조건을 담은 자료다. 임상 검증이나 개별 환자 진단의 정답이 아니다. 선택적인 `DISEASE_KNOWLEDGE`는 로컬에서 준비한 클래스 일반 정의만 읽으며, 파일이 없으면 정의 문맥을 추가하지 않는다. 해당 파일·사진·모델 가중치를 Git에 올리지 않는다.

추천 제한시간은 대기와 실행의 합이다. HTTP 취소/시간 초과 뒤에도 이미 실행 중인 GPU 작업이 실제 끝날 때까지 슬롯을 유지한다. 업로드 크기는 읽기 단계에서 제한하며 설문 검증 오류에 원문 값을 돌려주지 않는다.

권장 적용 순서는 AI → Spring(DB V3 포함) → Swift다. 구버전 AI 출처 필드가 없으면 Spring은 출처 미상으로 남긴다.

## 검증 범위

이번 PR만 분리한 새 작업 트리에서 `python -m pytest -q`: **460 통과, 12 건너뜀**. 로컬 과거 산출물 등이 없는 검사는 건너뛰며 해당 자료는 업로드하지 않는다. 선행 연구 전체 작업 트리의 600 통과/4 건너뜀과 구별한다.

앞서 실제 모델과 사진 3장으로 Swift 통신 코드 → Spring HTTP → FastAPI HTTP 4개 시나리오를 검증했다. 보류 2건, 근거 카드 대체 제공 2건이었으며 iOS UI 실행 및 임상 성능 검증은 아니다. 상세는 [연동 시험 기록](swift-spring-flow-20261112.md) 참조.

PR 분리 시 원래 작업 트리에서 실험 분류기 로딩과 Grad-CAM 엔드포인트만 제외했다. 이번 실제 시험에 사용한 DenseNet·추천 생성·응답 처리 경로는 유지했다.
