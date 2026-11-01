# API 명세서

**문서 번호** SC-03 · **버전** v1.0 (2026-10-01) · **작성** 김우주 · **검토** 이현승, 정도희

이 문서가 프론트·백엔드·AI 서버 사이의 계약입니다. 바꿀 때는 [협업 규칙 §5](06-collaboration.md#5-명세-변경-절차)를 따릅니다.
백엔드 구현 후에는 Swagger(`/api/docs`)가 실제 동작 기준이 되며, 이 문서와 다르면 둘 중 하나를 고치는 PR을 올립니다.

**2026-10-21 AI 연동 변경:** 현재 질환 분류→추천 경로는 [파이프라인 계약](../disease-pipeline.md)과
[추천 품질 계약](../recommendation-quality.md)을 함께 적용합니다. FastAPI의
`recommendation_status`는 `generated | uncertain | no_concern | quality_rejected`입니다.
`quality_rejected`는 HTTP 오류가 아니며 추천 객체는 `null`입니다. Spring은 상태와
`quality_issues`를 보존하고 Swift는 추천 카드 대신 `guidance`를 표시해야 합니다.
성분의 `purpose_status`, `purpose_source`와 추천의 `quality_status`,
`postprocess_version`도 저장·전달합니다. 아래 기존 JSON 예시는 이 추가 필드를 생략한 예시입니다.

## 1. 공통 규약

| 항목 | 규칙 |
|---|---|
| Base URL | `https://app.<도메인>/api/v1` (로컬: `http://localhost:3000/api/v1`) |
| 인증 | `Authorization: Bearer <accessToken>`. 표의 "인증" 열이 `-`인 API만 예외 |
| 형식 | 요청·응답 `application/json; charset=utf-8` (업로드만 `multipart/form-data`) |
| 필드 이름 | camelCase. 열거형 **값**은 소문자 snake_case (`full_face`, `cheek_left`) |
| 시간 | ISO 8601 UTC (`2026-10-15T09:12:33Z`). 화면 표시는 프론트가 KST로 변환 |
| ID | 분석 ID는 ULID 26자 (`01J9Z3K5V8Q2M4N6P8R0S2T4V6`) |
| 요청 추적 | 요청 헤더 `X-Request-Id`(선택). 응답에 항상 `X-Request-Id` 포함 |
| 페이지 | `?page=1&size=20` (size 최대 50). 응답 `data.page` 객체 포함 |

### 1.1 응답 포맷

성공
```json
{ "success": true, "data": { } }
```

실패
```json
{
  "success": false,
  "error": { "code": "E2001", "message": "얼굴을 찾지 못했어요. 정면에서 다시 찍어 주세요.", "detail": null }
}
```

- `message`는 **사용자에게 그대로 보여줘도 되는 문장**입니다. 프론트는 코드별 문구를 따로 만들지 않아도 됩니다.
- `detail`은 개발용(검증 실패 필드 목록 등). 화면에 보여주지 않습니다.
- `204 No Content`는 본문이 없습니다.

### 1.2 에러 코드

| 코드 | HTTP | 의미 | 발생 | 재시도 | 프론트 처리 |
|---|---|---|---|---|---|
| E1001 | 401 | 토큰 없음·만료·위조 | 백엔드 | 토큰 재발급 후 1회 | 재발급 실패 시 로그인 화면 |
| E1002 | 403 | 권한 없음 | 백엔드 | 안 함 | 공통 오류 |
| E1003 | 401 | 이메일 또는 비밀번호 불일치 | 백엔드 | 안 함 | 로그인 폼 아래 문구 |
| E1004 | 409 | 이미 가입된 이메일 | 백엔드 | 안 함 | 가입 폼 이메일 필드 오류 |
| E2001 | 422 | 얼굴 미검출 | AI → 백엔드 | 안 함 | 재촬영 화면 |
| E2002 | 422 | 이미지 품질 미달(흐림·과노출·저조도) | AI → 백엔드 | 안 함 | 재촬영 화면 + 사유 |
| E2003 | 413 | 이미지 10MB 초과 | 백엔드(nginx 12MB), AI | 안 함 | 업로드 전 차단이 원칙 |
| E2004 | 415 | JPEG·PNG가 아님 | 백엔드, AI | 안 함 | 업로드 전 차단이 원칙 |
| E3001 | 503 | AI 서버 응답 없음·연결 설정 오류, 또는 60분 넘게 처리되지 못함 | 백엔드 | 30초 후 | "분석 서버 점검 중" |
| E3002 | 504 | 추론 타임아웃 | 백엔드, AI | 추천 재요청 가능 | 측정 결과 + 추천 재시도 버튼 |
| E3003 | 503 | 대기열 가득 참 | 백엔드, AI | 1분 후 | "요청이 많아요" |
| E4001 | 500 | 내부 오류 | 공통 | 안 함 | 공통 오류 (요청 ID 표시) |
| E4002 | 422 | 요청 형식 오류(검증 실패) | 백엔드, AI | 안 함 | 폼 필드 오류 (`detail` 참고) |
| E4003 | 404 | 리소스 없음 (남의 것 포함) | 백엔드 | 안 함 | "결과를 찾을 수 없어요" |
| E4004 | 429 | 요청 제한 초과 (진행 중 분석 1건, 일 20건(KST 기준), 로그인 분당 10회) | 백엔드 | `Retry-After` 헤더 | 안내 문구 |
| E4005 | 409 | 지금 상태에서는 할 수 없는 요청 (예: 실패하지 않은 추천 재요청) | 백엔드 | 안 함 | 화면 새로고침 |

AI 서버가 돌려준 에러는 백엔드가 이 표의 코드로 바꿔서 전달합니다([시스템 설계서 §4.3](02-system-design.md#43-ai-서버-에러-변환)). 예를 들어 AI 서버의 401(내부 토큰 불일치)은 클라이언트에 503 E3001로 나갑니다.

## 2. 엔드포인트 목록 — 클라이언트 ↔ 백엔드

| 메서드 | 경로 | 설명 | 인증 | 요구사항 |
|---|---|---|---|---|
| POST | `/auth/signup` | 회원가입 | - | FR-AUTH-01 |
| POST | `/auth/login` | 로그인 | - | FR-AUTH-02 |
| POST | `/auth/refresh` | 액세스 토큰 재발급 (쿠키) | - | FR-AUTH-02 |
| POST | `/auth/logout` | 로그아웃 | O | FR-AUTH-03 |
| GET | `/users/me` | 내 정보 | O | — |
| DELETE | `/users/me` | 회원 탈퇴 | O | FR-AUTH-04 |
| POST | `/analyses` | 분석 요청 (측정 + 추천 대기열 등록) | O | FR-ANA-01, 02, 05 |
| GET | `/analyses/{id}` | 분석 결과·진행 상태 | O | FR-ANA-03, FR-REP-01, 02 |
| POST | `/analyses/{id}/recommendation/retry` | 실패한 추천 재요청 | O | FR-ANA-04 |
| GET | `/analyses` | 이력 목록 | O | FR-HIS-01 |
| GET | `/analyses/{id}/trend` | 직전 분석 대비 변화 | O | FR-HIS-02 |
| DELETE | `/analyses/{id}` | 이력 삭제 | O | FR-HIS-03 |
| GET | `/system/status` | AI 가용성, 대기열 | - | FR-SYS-01 |

## 3. 인증 · 사용자

### 3.1 POST /auth/signup

```json
{
  "email": "user@example.com",
  "password": "Skin!2026",
  "nickname": "우주",
  "agreements": { "terms": true, "privacy": true, "imageProcessing": true, "over14": true }
}
```

| 필드 | 규칙 |
|---|---|
| `email` | 이메일 형식, 최대 255자, 소문자로 정규화 |
| `password` | 8–64자, 영문·숫자·특수문자 중 2종 이상 |
| `nickname` | 2–20자 |
| `agreements.*` | 4개 모두 `true`여야 함. 아니면 E4002 |

응답 `201`
```json
{ "success": true, "data": { "userId": 42 } }
```

### 3.2 POST /auth/login

요청 `{ "email": "...", "password": "..." }`

응답 `200` + `Set-Cookie: refresh_token=…; HttpOnly; Secure; SameSite=Strict; Path=/api/v1/auth; Max-Age=1209600`
```json
{
  "success": true,
  "data": {
    "accessToken": "eyJhbGciOi…",
    "expiresIn": 3600,
    "user": { "id": 42, "email": "user@example.com", "nickname": "우주" }
  }
}
```

### 3.3 POST /auth/refresh

본문 없음. 쿠키의 리프레시 토큰을 검증하고 **새 리프레시 토큰으로 교체**합니다(재사용 감지 시 해당 사용자의 모든 리프레시 토큰 폐기). 응답은 로그인과 같은 구조.

### 3.4 POST /auth/logout → `204`

### 3.5 GET /users/me

```json
{ "success": true, "data": { "id": 42, "email": "user@example.com", "nickname": "우주", "createdAt": "2026-10-03T02:11:00Z", "analysisCount": 3 } }
```

### 3.6 DELETE /users/me → `204`

계정, 분석, 지표, 설문, 추천, 리프레시 토큰을 모두 즉시 삭제합니다. 진행 중인 추천 작업도 취소합니다.

## 4. 분석

### 4.1 POST /analyses

요청 `multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `image` | file | O | JPEG/PNG, 10MB 이하, 짧은 변 720px 이상 |
| `survey` | string(JSON) | O | 아래 구조 |
| `deviceType` | string | O | `smartphone` \| `tablet` \| `camera` |

`survey`
```json
{
  "gender": "female",
  "ageBand": "20s",
  "skinType": "combination",
  "sleepHours": 6,
  "waterIntakeLevel": "low",
  "sunExposureLevel": "high",
  "stressLevel": "medium",
  "concerns": ["pore", "pigmentation"],
  "currentProducts": ["토너", "수분크림"],
  "note": "여름에 번들거림이 심해요"
}
```

| 필드 | 필수 | 값 |
|---|---|---|
| `gender` | O | `male` \| `female` |
| `ageBand` | O | `10s` … `70s` |
| `skinType` | - | 피부 타입 코드 (§6.3). 모르면 생략 또는 `null` |
| `sleepHours` | - | 0–14 |
| `waterIntakeLevel`, `sunExposureLevel`, `stressLevel` | - | `low` \| `medium` \| `high` |
| `concerns` | - | 고민 코드 배열 (§6.4), 최대 3개 |
| `currentProducts` | - | 문자열 배열, 최대 5개, 각 30자 |
| `note` | - | 300자 이하 |

처리 순서: 인증 → 요청 제한(E4004) → 대기열 상한(E3003) → 형식 검증(E2003·E2004·E4002) → AI 측정(E2001·E2002·E3001) → 저장 → 응답

응답 `202 Accepted`
```json
{
  "success": true,
  "data": {
    "analysisId": "01J9Z3K5V8Q2M4N6P8R0S2T4V6",
    "status": "queued",
    "queuePosition": 2,
    "estimatedWaitSec": 110
  }
}
```

측정이 실패하면 분석을 만들지 않고 `422 E2001/E2002` 또는 `503 E3001`을 바로 돌려줍니다.

### 4.2 GET /analyses/{id}

`status`가 `queued`·`recommending`이면 프론트는 다시 호출합니다 — 처음 3분은 3초, 이후 10초 간격, 15분이 지나면 중단.

응답 `200` (완료)
```json
{
  "success": true,
  "data": {
    "analysisId": "01J9Z3K5V8Q2M4N6P8R0S2T4V6",
    "status": "done",
    "createdAt": "2026-10-15T09:12:33Z",
    "completedAt": "2026-10-15T09:13:31Z",
    "deviceType": "smartphone",
    "queuePosition": null,
    "estimatedWaitSec": null,
    "measurement": {
      "isDemo": false,
      "confidence": 0.87,
      "calibrationApplied": true,
      "overallScore": 63,
      "summary": [
        { "metric": "moisture",     "value": 61, "score": 61, "grade": "normal" },
        { "metric": "elasticity",   "value": 74, "score": 74, "grade": "good" },
        { "metric": "pigmentation", "value": 38, "score": 62, "grade": "normal" },
        { "metric": "pore",         "value": 45, "score": 55, "grade": "normal" }
      ],
      "byRoi": [
        { "roi": "forehead",   "moisture": 58, "elasticity": 71, "pigmentation": 35, "pore": 51, "overallScore": 61, "grade": "normal" },
        { "roi": "cheek_left", "moisture": 66, "elasticity": 76, "pigmentation": 41, "pore": 38, "overallScore": 66, "grade": "normal" }
      ],
      "modelVersion": "skin_reg_v1"
    },
    "recommendation": {
      "status": "done",
      "skinType": { "code": "combination", "label": "복합성" },
      "concerns": [
        { "code": "pore", "label": "모공" },
        { "code": "pigmentation", "label": "미백(색소침착/기미/칙칙함)" }
      ],
      "ingredients": [
        { "name": "나이아신아마이드", "purpose": "피지 분비와 모공 관리에 도움을 줄 수 있습니다.", "priority": 1 },
        { "name": "판테놀", "purpose": "피부 장벽을 지키는 데 도움을 줄 수 있습니다.", "priority": 2 },
        { "name": "알로에신", "purpose": "칙칙한 피부 톤 관리에 도움을 줄 수 있습니다.", "priority": 3 }
      ],
      "routine": [
        { "step": 1, "time": "morning", "action": "약산성 클렌저로 가볍게 세안합니다." },
        { "step": 2, "time": "morning", "action": "자외선 차단제를 바릅니다." },
        { "step": 3, "time": "evening", "action": "나이아신아마이드 세럼을 바릅니다." }
      ],
      "reasoning": [
        "복합성 피부로 T존의 피지 분비가 많아 모공이 도드라져 보일 수 있습니다. 자외선 노출이 많은 생활 습관은 피부 톤이 칙칙해지는 데 영향을 줄 수 있습니다. …",
        "나이아신아마이드는 피지 분비와 모공 관리에 도움을 줄 수 있습니다. 판테놀은 … 알로에신은 …",
        "아침에는 약산성 클렌저로 세안한 뒤 자외선 차단제를 바르고, 저녁에는 … 세럼을 사용하세요. …"
      ],
      "answer": "복합성 피부에 모공과 톤이 고민이라면 …",
      "inputNotes": ["나이는 연령대(20s) 중간값 25세로 추정"],
      "disclaimer": "본 결과는 참고용 정보이며 의학적 진단이나 치료를 대신하지 않습니다. 피부 이상이 지속되면 피부과 전문의와 상담하세요.",
      "filtered": false,
      "usedFallback": false,
      "modelVersion": "rec_sllm_v2"
    },
    "survey": {
      "gender": "female", "ageBand": "20s", "skinType": "combination", "sleepHours": 6,
      "waterIntakeLevel": "low", "sunExposureLevel": "high", "stressLevel": "medium",
      "concerns": ["pore", "pigmentation"], "currentProducts": ["토너", "수분크림"], "note": "여름에 번들거림이 심해요"
    },
    "error": null
  }
}
```

진행 중이면 `recommendation`은 `{ "status": "pending" }`만 있고, `queuePosition`·`estimatedWaitSec`이 채워집니다.
추천이 실패하면 `status: "failed"`, `recommendation: { "status": "failed" }`, `error: { "code": "E3002", "message": "…" }`입니다. 실패 코드는 E3001·E3002·E3003·E4001 중 하나입니다. **측정 결과는 실패와 무관하게 항상 포함**됩니다.

| 필드 | 설명 |
|---|---|
| `measurement.isDemo` | `true`면 회귀 모델 연결 전 데모 값. 화면에 "데모 값" 배지 필수 |
| `measurement.summary[].value` | 원래 지표 값(0–100). 색소침착·모공은 높을수록 나쁨 |
| `measurement.summary[].score` | 좋은 방향으로 통일한 값. 색소침착·모공은 `100 − value` |
| `measurement.summary[].grade` | `score` 기준 등급 (§6.6) |
| `measurement.byRoi` | 검출된 ROI만 포함. `full_face`는 `summary`에 있으므로 제외. 데모 값이면 빈 배열일 수 있음. `overallScore`·`grade`는 부위별 점 색에 사용 |
| `recommendation.ingredients` | 0–5개. `purpose`는 `null`일 수 있음 |
| `recommendation.routine` | 0개 이상. 비어 있으면 루틴 영역을 숨김 |
| `recommendation.reasoning` | 3개 문단 (문제 분석·원인 / 성분 선택·근거 / 사용법·관리 방안). 각 문단은 여러 문장 |
| `recommendation.answer` | 모델이 쓴 요약 문단. 결과 화면 "한눈에 보기"에 사용 |
| `recommendation.inputNotes` | 추정해서 채운 입력 내역. "이렇게 가정했어요"로 접어서 표시 |
| `recommendation.filtered`, `usedFallback` | 화면 비표시, 로그·분석용 |
| `survey` | 제출한 설문(클라이언트 코드). 다음 측정 때 설문 기본값으로 사용 |

### 4.3 POST /analyses/{id}/recommendation/retry

`status=failed`일 때만 허용. 아니면 `409 E4005`. 응답 `202`는 4.1과 같은 구조. 대기열 상한·요청 제한도 4.1과 같이 적용합니다.

### 4.4 GET /analyses

`?page=1&size=20`
```json
{
  "success": true,
  "data": {
    "items": [
      { "analysisId": "01J9…", "createdAt": "2026-10-15T09:12:33Z", "status": "done",
        "overallScore": 63, "isDemo": false, "concerns": ["pore", "pigmentation"] }
    ],
    "page": { "page": 1, "size": 20, "totalItems": 3, "totalPages": 1 }
  }
}
```

### 4.5 GET /analyses/{id}/trend

같은 사용자의 **직전 `done` 분석**과 비교합니다. 데모 값 분석끼리는 비교하지 않습니다.
```json
{
  "success": true,
  "data": {
    "current":  { "analysisId": "01J9…", "date": "2026-10-15", "overallScore": 63,
                  "summary": { "moisture": 61, "elasticity": 74, "pigmentation": 38, "pore": 45 } },
    "previous": { "analysisId": "01J8…", "date": "2026-09-28", "overallScore": 60,
                  "summary": { "moisture": 54, "elasticity": 72, "pigmentation": 40, "pore": 47 } },
    "delta":    { "overallScore": 3, "moisture": 7, "elasticity": 2, "pigmentation": -2, "pore": -2 }
  }
}
```
이전 분석이 없으면 `previous`, `delta`가 `null`입니다. `delta`는 원래 값의 차이이므로 색소침착·모공은 **음수가 좋아진 것**입니다(화면 표시 시 주의).

### 4.6 DELETE /analyses/{id} → `204`

### 4.7 GET /system/status

```json
{ "success": true, "data": { "aiAvailable": true, "queueLength": 3, "estimatedWaitSec": 165, "checkedAt": "2026-10-15T09:12:00Z" } }
```
백엔드가 30초마다 AI `/health`를 확인한 결과입니다. `aiAvailable=false`면 프론트는 분석 제출을 막습니다.

## 5. 백엔드 ↔ AI 서버

| 항목 | 값 |
|---|---|
| Base URL | 환경변수 `AI_BASE_URL` (예: `https://ai.<도메인>`) |
| 인증 | `X-Internal-Token: <AI_INTERNAL_TOKEN>`. `/health`만 예외 |
| 필드 이름 | 응답은 snake_case(Python 관례). **백엔드가 camelCase로 변환해 저장·응답** |
| 응답 형식 | 성공은 본문 그대로(`{success, data}` 감싸지 않음), 실패는 §1.1과 같은 `{success: false, error}` |
| 실제 스키마 | `skincare-ai/server/schemas.py`, 서버 실행 후 `http://localhost:8000/docs` |

| 메서드 | 경로 | 백엔드 사용 | 타임아웃 | 재시도 |
|---|---|---|---|---|
| GET | `/health` | 30초마다 상태 확인 | 5초 | — |
| GET | `/model-info` | 기동 시 1회 로그 | 5초 | — |
| POST | `/infer/skin-metrics` | `POST /analyses` 안에서 동기 호출 | 20초 | 연결 오류만 1회 |
| POST | `/infer/recommend` | 추천 워커에서 호출 | 130초 | 연결 오류·503만 1회. **504는 재시도 금지** (이전 생성이 GPU를 계속 점유) |
| POST | `/infer/analyze` | 사용 안 함 (테스트용) | — | — |

### 5.1 POST /infer/skin-metrics

요청 `multipart/form-data`: `image`(file), `device_type`(`smartphone`|`tablet`|`camera`)

응답 `200`
```json
{
  "roi_detected": ["full_face", "forehead", "glabella", "cheek_left", "cheek_right", "chin"],
  "metrics": {
    "full_face": { "moisture": 0.61, "elasticity": 0.74, "pigmentation": 0.38, "pore": 0.45 },
    "forehead":  { "moisture": 0.58, "elasticity": 0.71, "pigmentation": 0.35, "pore": 0.52 }
  },
  "confidence": 0.87,
  "calibration_applied": true,
  "is_stub": false,
  "model_version": "skin_reg_v1",
  "inference_ms": 840
}
```
- 값은 0–1. 백엔드는 **0–1 원본을 그대로 저장**하고, 클라이언트에 응답할 때 `round(v × 100)`으로 바꿉니다.
- `is_stub: true`(현재 기본)면 해시로 만든 가짜 값입니다. 백엔드는 `analyses.is_demo=true`로 저장합니다.
- 에러: `401 E1001`, `413 E2003`, `415 E2004`, `422 E2001/E2002`(회귀 모델 연결 후), `503 E3003`, `504 E3002`

### 5.2 POST /infer/recommend

요청
```json
{
  "metrics": { "full_face": { "moisture": 0.61, "elasticity": 0.74, "pigmentation": 0.38, "pore": 0.45 } },
  "survey": {
    "gender": "female",
    "ageBand": "20s",
    "skinType": "복합성",
    "sleepHours": 6,
    "waterIntakeLevel": "low",
    "sunExposureLevel": "high",
    "stressLevel": "medium",
    "currentProducts": ["토너", "수분크림"],
    "selfReportedConcerns": ["모공", "미백(색소침착/기미/칙칙함)"],
    "note": "여름에 번들거림이 심해요"
  }
}
```

측정값이 데모 값(`is_demo`)인 분석은 `"metrics": null`로 보냅니다. 그러면 AI 서버는 설문만으로 피부 타입·고민을 정합니다.

**백엔드 변환 규칙** (클라이언트 설문 → AI 설문)

| 클라이언트 | AI 서버 | 변환 |
|---|---|---|
| `skinType: "combination"` | `skinType: "복합성"` | §6.3 코드 → 라벨 (영문 코드도 인식하지만 라벨로 통일) |
| `concerns: ["pore"]` | `selfReportedConcerns: ["모공"]` | §6.4 코드 → 라벨. **코드를 그대로 보내면 인식되지 않음** |
| (저장된 측정값) | `metrics` (0–1) | DB에 저장한 0–1 원본을 ROI별로 그대로 전달 |
| 나머지 | 같은 이름 | 그대로 |

응답 `200`
```json
{
  "skin_type": "복합성",
  "concerns": ["모공", "미백(색소침착/기미/칙칙함)"],
  "ingredients": [{ "name": "나이아신아마이드", "purpose": "…", "priority": 1 }],
  "routine": [{ "step": 1, "time": "morning", "action": "약산성 클렌저로 가볍게 세안합니다." }],
  "reasoning": ["(문제 분석·원인 문단)", "(성분 선택·근거 문단)", "(사용법·관리 방안 문단)"],
  "answer": "…",
  "disclaimer": "본 결과는 참고용 정보이며 …",
  "filtered": false,
  "used_fallback": false,
  "input_notes": ["나이는 연령대(20s) 중간값 25세로 추정"],
  "model_version": "rec_sllm_v2",
  "inference_ms": 48100
}
```
- `skin_type`, `concerns`는 **한글 라벨**로 옵니다. 백엔드가 §6.3·6.4로 코드를 붙여 저장합니다. 표에 없는 값이 오면 E4001로 처리하고 로그를 남깁니다.
- `ingredients`는 최대 5개(`purpose`는 `null` 가능), `routine[].time`은 §6.5입니다. 현재 SLLM 서버는 성분 사전이 없으면 초기화에 실패합니다. 생성 후 성분·루틴 등이 비면 품질 보류로 처리하며, 단독 `/infer/recommend`에서는 `quality_status: "withheld"`와 빈 목록이 반환됩니다. 질환 통합 경로에서는 `recommendation: null`입니다.
- 추천 모델은 측정 수치·부위를 직접 보지 않으므로 `reasoning`에 수치가 나오지 않습니다.
- 에러: `401 E1001`, `422 E4002`, `503 E3003`, `504 E3002`

### 5.3 GET /health

```json
{ "status": "ok", "rec_loaded": true, "skin_loaded": true, "rec_backend": "sllm", "skin_backend": "stub" }
```
`status`가 `ok`가 아니거나 5초 안에 응답이 없으면 `aiAvailable=false`.

## 6. 열거형 (enum)

### 6.1 상태

| 필드 | 값 |
|---|---|
| 분석 `status` | `queued`(추천 대기) · `recommending`(추천 생성 중) · `done` · `failed` |
| `recommendation.status` | `pending`(queued·recommending일 때) · `done` · `failed` |

### 6.2 ROI `roi`
`full_face`, `forehead`, `glabella`(미간), `eye_left`, `eye_right`, `cheek_left`, `cheek_right`, `lip`, `chin`

### 6.3 피부 타입

| 코드 | 라벨 (AI 서버 값) |
|---|---|
| `dry` | 건성 |
| `normal` | 중성 |
| `oily` | 지성 |
| `combination` | 복합성 |

### 6.4 피부 고민 (8종)

| 코드 | 라벨 (AI 서버 값) | 화면 표기 |
|---|---|---|
| `pigmentation` | 미백(색소침착/기미/칙칙함) | 피부 톤·잡티 |
| `pore` | 모공 | 모공 |
| `wrinkle` | 주름 | 주름 |
| `acne` | 여드름/뾰루지 | 트러블 |
| `redness` | 붉어짐(홍조) | 붉은기 |
| `dryness` | 과각질/악건성 | 건조·각질 |
| `sagging` | 피부처짐/탄력저하 | 탄력 |
| `sensitivity` | 민감성(트러블/자극감) | 민감함 |

"화면 표기"는 의료 용어로 읽히지 않게 고른 사용자용 표현입니다. `acne` 라벨의 "여드름"은 질환명이므로 화면에는 "트러블"로 씁니다.

### 6.5 루틴 시간 `time`
`morning` · `evening` · `daily` · `weekly` · `anytime`

### 6.6 등급 `grade`

| 코드 | `score` 범위 | 화면 표기 |
|---|---|---|
| `good` | 70–100 | 양호 |
| `normal` | 40–69 | 보통 |
| `care` | 0–39 | 관리 필요 |

### 6.7 기타
`deviceType`: `smartphone` · `tablet` · `camera` / `gender`: `male` · `female` / `ageBand`: `10s` … `70s` / 수준: `low` · `medium` · `high`

## 7. 개발용 고정 응답 (fixture)

`AI_MODE=fixture`일 때 백엔드는 AI 서버 대신 `test/fixtures/ai/` 아래 JSON을 돌려줍니다. 어떤 응답을 줄지는 환경변수 `AI_FIXTURE_SCENARIO`로 고르고, **운영(`NODE_ENV=production`)이 아닐 때만** 요청 헤더 `X-Fixture-Scenario`로 요청마다 바꿀 수 있습니다(테스트용).

| 시나리오 | 측정 | 추천 |
|---|---|---|
| `ok` (기본) | `skin-metrics.ok.json` (§5.1, `is_stub: true`) | `recommend.ok.json` (§5.2), 1초 지연 |
| `no-face` | `422 E2001` | — |
| `rec-timeout` | 정상 | `504 E3002` 즉시 반환 |
| `rec-slow` | 정상 | 60초 지연 후 정상 (대기열 테스트용) |
| `ai-down` | 연결 실패 | 연결 실패 |

`recommend.ok.json`은 김우주가 v2 모델 실제 출력으로 만들어 10/3까지 `skincare-ai/docs/fixtures/`에 올립니다.
