# Swift → Spring → FastAPI 연동 검증 (2026-11-12)

프론트와 백엔드의 실제 구현 브랜치를 확인하고, `not_generated`를 포함한 생성 출처와 추천 품질 보류 처리를 연결했다. 실제 사진으로 실행한 4개 시나리오와 인증·이력·삭제 흐름이 통과했다.

**PR 검토 범위:** 이 문서는 선행 로컬 시험 기록이다. AI 검토 브랜치에는 추천 서비스 의존 코드만 분리했으며 실험 분류기·Grad-CAM·학습 자료·개별 응답 로그는 포함하지 않는다. 분리된 코드만으로 다시 실행한 테스트는 **460 통과, 12 건너뜀**이다. 아래 600 통과/4 건너뜀은 연구 코드까지 있던 원래 작업 트리의 결과다. [배포 계약](recommendation-serving-contract.md)을 함께 확인한다.

## 적용 내용

- Frontend: `origin/chore/ios-project-setup`을 기준으로 로컬 `fix/ai-response-flow`에서 작업했다. 품질 보류를 분류 불확실과 구별하고, `not_generated` 응답은 추천 성공 카드로 표시하지 않는다. 근거 카드/규칙 대체 답변을 구별하는 안내와 회피 성분 입력을 추가했다.
- Backend: `origin/develop`을 기준으로 같은 이름의 로컬 브랜치에서 작업했다. `quality_rejected`를 WITHHELD로 저장한다. AI `guidance`, 생성 출처·품질 사유, 성분 purpose 출처를 보존하고 `avoidIngredients`를 AI에 전달한다.
- FastAPI: `/infer/disease-recommend`가 추천 객체를 null로 바꾸는 경우에도 최상위 `generation_origin`을 반환한다. 불확실/고민 없음/근거 후보 없음은 `not_generated`, 생성 후 검사에서 보류되면 실제 생성 출처를 보존한다.
- DB V3: 기존 행을 보존하며 생성 정보 JSONB와 품질 보류 상태를 추가한다. 최신 모델 버전이 저장되도록 길이 제한을 40자에서 160자로 확장한다.
- `not_generated`는 **생성 출처**이며 `recommendationStatus` enum 값이 아니다. `generated`와 `not_generated`/`withheld`가 함께 오는 모순된 AI 응답은 Spring에서 거절하고, Swift도 성공 표시를 막는다.

## 실제 실행 범위

Swift 6.1.3 Linux에서 **앱의 실제 APIClient, 토큰 관리, HTTPTransport, Endpoints, DTO 코드**를 컴파일하여 실행했다. Spring Boot/Java 21과 실제 HTTP로 통신하고, Spring 작업자는 실제 FastAPI/Uvicorn 서버를 호출했다. PostgreSQL 18의 별도 시험 DB를 사용했다. Windows–WSL 구간은 localhost 전용 일회성 SSH 터널을 사용했다. 목 응답으로 바꾼 경로가 아니다.

이미지는 다운로드 `test_sample_96`에서 기존에 사용한 정면 사진 3장이다. 아토피 사진을 회피 조건 유무에 따라 두 번 사용하여 총 4개 요청을 실행했다. PNG 원본은 읽기만 하고 JPEG 전환은 메모리에서 수행했다. 사진 복사본이나 이미지 본문은 저장하거나 외부로 전송하지 않았다.

DenseNet121이 실제로 예측한 분류를 Spring이 보존하고 그대로 추천 요청에 전달했다. 정답 라벨을 추천 입력으로 넣지 않았다. 여성·30대·중성·자가 고민 없음은 공통 **가상 설문**이다. 앱 연령대 계약에 따라 AI는 30대의 중간값인 35세를 사용하고 이를 `inputNotes`에 표시했다. 사진 속 인물의 인적 정보가 아니다.

Qwen2.5-7B와 v2 어댑터를 로드했다. 기존 근거 적용 설정(`REC_EVIDENCE_MODE=enforce`, `REC_EVIDENCE_USE_ADAPTER=false`)을 유지했으므로 근거 문맥 생성에서는 어댑터를 비활성화하는 현재 경로를 시험했다. 이번 작업에서 재학습하거나 모델을 교체하지 않았다.

| 시험 | 최종 상태 | 출처 | 제출부터 종료까지 | 결과 |
|---|---|---|---:|---|
| 아토피 사진 + 근거 후보 성분 전부 회피 | WITHHELD / quality_rejected | not_generated | 1.23초 | 추천 null, 보류 안내·사유 보존, 성공 카드 숨김, 재요청 409 |
| 정상 사진 + 자가 고민 없음 | WITHHELD / no_concern | not_generated | 1.13초 | 추천 null, 보류 안내, 조회 종료, 재요청 409 |
| 건선 사진 | COMPLETED / generated | source_cards | 77.62초 | 근거 카드 대체 안내, 성분 2개, 사용 조건과 출처 보존 |
| 아토피 사진 | COMPLETED / generated | source_cards | 39.34초 | 근거 카드 대체 안내, 성분 2개, 사용 조건과 출처 보존 |

추천 제공 2건은 모두 **모델 출력 검사 후 근거 카드로 구성한 대체 안내**다. 모델 원문 생성 성공 2건이라고 계산하지 않는다. 응답의 `usedFallback=true`와 `generation.origin=source_cards`가 일치했다. 예를 들어 아토피 응답은 세라마이드 보습제에 “무향의 완제품 보습제를 제품 표시대로 사용한다”를 명시했고, 성분의 `purposeSource=source_card`, `purposeStatus=provided`도 앱 응답에 남았다. 이 검토는 문구·연동 확인이며 성분 효능에 대한 임상 검증이 아니다.

회원가입·로그인 → 제출 → 대기/생성 상태 조회 → 종료 → 이력 조회 → 리프레시 토큰 갱신 → 분석 삭제/404 확인 → 시험 계정 삭제까지 통과했다. 네 요청 모두 최초 분류와 최종 조회의 분류 값이 같았고, 회피 성분도 보존됐다. 시험 계정과 분석은 정리했다.

## 자동 시험

| 항목 | 결과 |
|---|---:|
| Spring 단위 테스트 | 150 통과 |
| Spring HTTP + PostgreSQL 통합 테스트 | 88 통과 |
| Spring Spotless 및 bootJar | 통과 |
| Swift 응답·출처·토큰 갱신·폴링 정책 테스트 | 23 통과 |
| Swift 전체 앱 소스 구문 검사 | 통과 (타입 검사·iOS 빌드 아님) |
| AI 전체 회귀 테스트 | 600 통과, 4 건너뜀 |
| 실제 모델을 연결한 Swift HTTP 시나리오 | 4/4 통과 |

회귀 시험에는 기존 보류·실패·시간 초과·재요청·개인정보 로그 방지 검사도 포함된다. 실제 사진 연동에서는 오류 응답을 인위적으로 만들지 않았으며, 분류 불확실·시간 초과 등은 자동 시험으로 확인했다. 실시간 시험 클라이언트는 1초 간격으로 조회했고, 앱의 기존 3초/10초 간격 정책은 별도 테스트로 확인했다.

## 남은 검증과 재현

Windows에서는 Xcode/iOS 시뮬레이터를 실행할 수 없다. **SwiftUI 화면·실제 촬영 전처리·iOS Keychain·앱 전체 빌드·SwiftLint 실행은 이번에 검증하지 못했다.** 통신 코드 실행 성공과 iOS 앱 전체 검증을 구별해야 한다. Mac에서 새 회피 성분 입력과 보류/출처 안내 화면을 확인해야 한다.

Frontend에서 `swift test -c release`, Backend에서 별도 이름에 `test`가 들어간 DB를 설정한 후 `gradlew spotlessCheck test bootJar integrationTest`로 재현한다. Backend의 기존 V2는 과거 분석 삭제를 포함하므로 신규 시험 DB를 사용했다. 이번 V3에는 삭제가 없다.

상세 결과는 로컬 `outputs/swift_spring_20261112/`의 `live-summary.json`, `live-plan.json`, `*-response.json`, `backend-tests.log`, `swift-tests.log`, `ai-tests.log`에 있다. Frontend `Tools/LiveFlow.swift`가 실제 통신 시험 코드이며, 그림이나 원본 사진을 포함하지 않는다.

이 시험은 기존 사진에 대한 기능 연동 확인이다. 새 환자 일반화, 분류 정확도 또는 의학적 추천 정확도 측정이 아니다. 원본 응답과 로그는 로컬에 보관하고 코드·문서만 PR 검토 대상으로 공유한다.

시험이 끝난 뒤 전용 Spring/FastAPI·터널을 종료하고 임시 키·토큰을 제거했다. 시험 PostgreSQL 클러스터는 중지하고 수동 시작으로 바꿨다. 설치한 새 SSH 서비스도 비활성화했으며 기존 사용자 SSH 키·설정은 건드리지 않았다. 시험 도구와 DB는 로컬에 남겨 재현에 사용할 수 있다.
