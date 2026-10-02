# 협업 규칙 · 개발 가이드

**문서 번호** SC-06 · **버전** v1.0 (2026-10-01) · **작성** 김우주 · **관리** 손승우

## 1. 저장소

| 레포 | 담당 | 리뷰어 | 내용 |
|---|---|---|---|
| `skincare-frontend` | 정도희 | 이현승 | React SPA |
| `skincare-backend` | 이현승 | 정도희 | NestJS API, DB 마이그레이션, Docker Compose |
| `skincare-ai` | 김우주 | 박성근 | 학습·평가 코드, 추론 서버, **이 문서 묶음(`docs/project/`)** |

- 프로젝트 공통 문서는 `skincare-ai/docs/project/`에 둡니다. 프론트·백엔드 README 첫 줄에 이 폴더 링크를 겁니다.
- 세 레포 모두 Public입니다. 비밀값과 AI허브 데이터는 절대 커밋하지 않습니다(§7).

## 2. 브랜치

```
main      ← 배포·발표 시점. 직접 푸시 금지
 └ develop ← 통합. 직접 푸시 금지, PR로만 병합
    ├ feat/23-login-page
    ├ fix/41-polling-timeout
    └ docs/12-api-retry
```

| 접두어 | 용도 | 출발 → 병합 |
|---|---|---|
| `feat/` | 기능 | develop → develop |
| `fix/` | 버그 | develop → develop |
| `docs/` | 문서·명세 | develop → develop |
| `chore/` | 설정, 의존성, CI | develop → develop |
| `hotfix/` | 발표 직전 운영 장애 | main → main + develop |

이름 형식: `{접두어}/{이슈번호}-{영문-요약}` (예: `feat/23-login-page`)

**브랜치 보호 (각 레포 Settings → Branches)** — `main`, `develop`: PR 필수, 승인 1명, CI 통과 필수, force push 금지.

## 3. 커밋

```
<타입>: <무엇을 했는지 한 줄, 한국어> (#이슈번호)

(선택) 왜 바꿨는지
```

| 타입 | 예 |
|---|---|
| `feat` | `feat: 분석 결과 폴링 훅 추가 (#31)` |
| `fix` | `fix: 추천 실패 시 측정 결과가 사라지는 문제 수정 (#44)` |
| `docs` | `docs: API 명세에 재시도 엔드포인트 추가 (#12)` |
| `refactor` | 동작 변경 없는 구조 개선 |
| `test` | 테스트 추가·수정 |
| `chore` | 빌드, 의존성, 설정 |

한 커밋에는 한 가지 변경만 담습니다. "여러 가지 수정" 같은 커밋은 리뷰에서 반려합니다.

## 4. 이슈 · PR

### 4.1 이슈

- 모든 작업은 이슈에서 시작합니다. 템플릿: 기능 / 버그 / 명세 변경 (`templates/github/ISSUE_TEMPLATE/` → 각 레포 `.github/ISSUE_TEMPLATE/`로 복사).
- 라벨

| 종류 | 라벨 |
|---|---|
| 유형 | `feat` `bug` `docs` `spec` `chore` |
| 영역 | `fe` `be` `ai` `infra` |
| 우선순위 | `P0`(발표·시연 차단) `P1`(이번 마일스톤) `P2`(여유 시) |

- 마일스톤은 [WBS](07-wbs-schedule.md#3-마일스톤)의 M1–M6와 같은 이름으로 만듭니다.
- 조직 Projects 보드 1개(`스킨케어 개발`)에 세 레포 이슈를 모읍니다. 열: Todo → In Progress → Review → Done.

### 4.2 PR

| 규칙 | 내용 |
|---|---|
| 대상 | `develop` (hotfix만 `main`) |
| 크기 | 변경 400줄 이하 권장. 넘으면 나눠서 올림 |
| 본문 | 템플릿 작성 (무엇을, 왜, 테스트 방법, 화면 캡처) |
| 리뷰 | §1 표의 리뷰어 1명 승인. **24시간 안에** 첫 리뷰 |
| 병합 | CI 통과 + 승인 후 작성자가 **Squash merge**. 병합 후 브랜치 삭제 |
| API·DB 영향 | `spec` 라벨을 달고 영향받는 레포 담당자를 리뷰어로 추가 |

### 4.3 완료 기준 (Definition of Done)

- [ ] 요구사항 ID의 수용 기준을 만족한다
- [ ] 단위 테스트가 있고 CI가 통과한다
- [ ] API·DB·화면이 바뀌었다면 해당 문서도 같은 PR에서 고쳤다
- [ ] 로그에 이미지·설문 원문·토큰이 남지 않는다
- [ ] 화면 문구가 [문구 규칙](05-screen-design.md#4-문구-규칙)을 지킨다

## 5. 명세 변경 절차

API·DB·열거형을 바꾸면 프론트·백엔드·AI가 동시에 영향받기 때문에 **문서 먼저, 코드 나중**입니다.

1. `spec` 라벨로 이슈를 연다 (무엇을, 왜, 영향 레포).
2. `skincare-ai`에 `docs/{이슈번호}-…` 브랜치로 03·04 문서를 고치는 PR을 올리고, 영향받는 담당자 전원을 리뷰어로 지정한다.
3. 전원 승인 후 병합한다. 문서 맨 위 버전을 올린다(호환 깨짐: v1.0 → v1.1, 단순 추가: v1.0.1).
4. 각 레포는 그 PR 번호를 본문에 적고 구현한다.

시연 1주 전(10/13, 12/2)부터는 호환이 깨지는 명세 변경을 하지 않습니다.

## 6. 코드 스타일

| 레포 | 도구 | 설정 |
|---|---|---|
| frontend | ESLint(typescript-eslint, react-hooks) + Prettier | `printWidth 100`, `singleQuote`, 저장 시 자동 정렬 |
| backend | NestJS 기본 ESLint + Prettier | 위와 동일 |
| ai | ruff + pytest | `line-length 120` |

| 대상 | 규칙 | 예 |
|---|---|---|
| API 경로 | 소문자 복수형 kebab-case | `/analyses/{id}/recommendation/retry` |
| JSON 필드 | camelCase | `overallScore` |
| 열거형 값 | 소문자 snake_case | `full_face`, `cheek_left` |
| DB | snake_case | `skin_metrics.analysis_id` |
| TS 파일 | kebab-case | `analysis-detail.page.tsx`, `ai-client.service.ts` |
| Python | snake_case | `predict_metrics()` |
| 모델 산출물 | `{모델}_v{버전}` | `rec_sllm_v2`, `skin_reg_v1` |

## 7. 환경변수 · 비밀값

- 각 레포에 `.env.example`(가짜 값)만 커밋하고, 실제 `.env`는 `.gitignore`에 넣습니다.
- 실제 값(토큰, DB 비밀번호, JWT 비밀키)은 **담당자에게 1:1로만** 전달합니다. 단체 채팅방·이슈·PR에 붙이지 않습니다.
- 유출이 의심되면 즉시 새 값으로 바꾸고 팀에 알립니다. 비밀키 생성: `openssl rand -base64 48`
- CI에서 필요한 값은 GitHub Actions Secrets에 넣습니다.

### 7.1 백엔드 `.env`

| 변수 | 예 | 설명 |
|---|---|---|
| `PORT` | `3000` | |
| `DATABASE_URL` | `postgres://skincare:***@db:5432/skincare` | |
| `JWT_ACCESS_SECRET` / `JWT_REFRESH_SECRET` | 48바이트 난수 | 서로 다른 값 |
| `CORS_ORIGIN` | `https://app.<도메인>` | 로컬: `http://localhost:5173` |
| `AI_MODE` | `fixture` \| `real` | 로컬 개발 기본값 `fixture` |
| `AI_BASE_URL` | `https://ai.<도메인>` | |
| `AI_INTERNAL_TOKEN` | 48바이트 난수 | AI 서버와 동일 값 |
| `AI_SKIN_TIMEOUT_MS` | `20000` | |
| `AI_REC_TIMEOUT_MS` | `130000` | |
| `AI_HEALTH_INTERVAL_MS` | `30000` | |
| `REC_WORKER_CONCURRENCY` | `1` | 1보다 크게 하지 않음 |
| `REC_QUEUE_LIMIT` | `10` | 추천 대기 상한 (AI 서버의 `QUEUE_LIMIT`과 별개) |
| `REC_QUEUE_MAX_WAIT_MIN` | `60` | 이보다 오래 대기하면 `failed` E3001 |
| `ANALYSIS_DAILY_LIMIT` | `20` | 하루 기준 KST 00:00. 스테이징 테스트 때는 `100` |
| `LOGIN_RATE_PER_MIN` | `10` | IP당, `CF-Connecting-IP` 기준 |
| `TRUST_PROXY` | `true` | nginx·터널 뒤에서 실제 IP를 읽기 위해 |
| `AI_FIXTURE_SCENARIO` | `ok` | `AI_MODE=fixture`일 때 응답 시나리오 ([API §7](03-api-spec.md#7-개발용-고정-응답-fixture)) |

### 7.2 프론트엔드 `.env`

| 변수 | 예 |
|---|---|
| `VITE_API_BASE_URL` | `/api/v1` (로컬도 동일 — Vite proxy가 `localhost:3000`으로 전달) |
| `VITE_POLL_INTERVAL_MS` | `3000` |
| `VITE_POLL_SLOW_AFTER_MS` / `VITE_POLL_SLOW_INTERVAL_MS` | `180000` / `10000` |
| `VITE_POLL_TIMEOUT_MS` | `900000` (15분) |

### 7.3 AI 서버 `.env`

`skincare-ai/.env.example` 참고. 핵심: `AI_INTERNAL_TOKEN`, `REC_BACKEND=sllm`, `REC_ADAPTER_DIR=outputs/rec_sllm_v2/adapter`, `SKIN_BACKEND=stub`(회귀 모델 연결 후 `model`), `REC_TIMEOUT_S=120`.

## 8. CI (GitHub Actions)

각 레포 `.github/workflows/ci.yml` — `develop`·`main` 대상 PR과 푸시에서 실행.

| 레포 | 단계 |
|---|---|
| frontend | `npm ci` → `npm run lint` → `npm run test -- --run` → `npm run build` |
| backend | `npm ci` → `npm run lint` → `npm run test` → `npm run build` (+ PostgreSQL 서비스 컨테이너로 `npm run test:e2e`) |
| ai | `pip install -r requirements-dev.txt` → `ruff check .` → `pytest` (GPU 없이 동작) |

```yaml
# skincare-backend/.github/workflows/ci.yml (예시)
name: ci
on:
  pull_request: { branches: [develop, main] }
  push: { branches: [develop, main] }
jobs:
  build:
    runs-on: ubuntu-latest
    services:
      db:
        image: postgres:16-alpine
        env: { POSTGRES_USER: test, POSTGRES_PASSWORD: test, POSTGRES_DB: test }
        ports: ["5432:5432"]
        options: --health-cmd pg_isready --health-interval 5s --health-retries 10
    env:
      DATABASE_URL: postgres://test:test@localhost:5432/test
      AI_MODE: fixture
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npm run lint
      - run: npm run test
      - run: npm run build
      - run: npm run test:e2e
```

## 9. 로컬 실행

| 도구 | 버전 |
|---|---|
| Node.js | 20 LTS |
| Python | 3.11 (AI 레포만) |
| Docker Desktop | 최신 (PostgreSQL 로컬 실행용) |

**백엔드·프론트 개발 (GPU 불필요)**
```bash
# skincare-backend
docker compose up -d db
cp .env.example .env            # AI_MODE=fixture
npm ci && npm run migration:run && npm run seed
npm run start:dev               # http://localhost:3000/api/docs

# skincare-frontend
cp .env.example .env
npm ci && npm run dev           # http://localhost:5173  (vite.config.ts의 server.proxy: /api → http://localhost:3000)
```

**실제 AI 서버와 연동 (김우주 PC가 켜져 있을 때)**
1. 김우주: AI 서버 기동 + 터널 ([시스템 설계서 §7.2](02-system-design.md#72-ai-서버-김우주-pc))
2. 백엔드 `.env`: `AI_MODE=real`, `AI_BASE_URL`, `AI_INTERNAL_TOKEN` 설정 후 재시작
3. `GET /api/v1/system/status`가 `aiAvailable: true`인지 확인

## 10. 회의 · 소통

| 항목 | 규칙 |
|---|---|
| 정기 회의 | 주 1회 30분 (요일은 팀 합의). 지난주 완료 / 이번주 계획 / 막힌 것 |
| 주간보고서 | 손승우가 회의 후 작성, 다음날까지 공유 |
| 진행 공유 | 막힌 일은 **24시간 안에** 채널에 공유. 혼자 하루 넘게 붙잡지 않기 |
| AI 서버 가동 요청 | 연동 테스트가 필요하면 김우주에게 하루 전 요청 (PC 학습 일정과 겹침 방지) |
| 결정 기록 | 회의에서 정한 사항은 이슈 또는 이 문서 PR로 남김 |

## 11. 버전 · 릴리스

| 태그 | 시점 | 내용 |
|---|---|---|
| `v0.1.0` | 10/18 (M3) | 3계층 연동 프로토타입 |
| `v0.2.0` | 11/24 | 통합 테스트 통과 |
| `v1.0.0` | 12/8 | 최종 시연 버전 |

`develop` → `main` PR로 병합한 뒤 `main`에 태그를 붙이고 GitHub Release에 변경 사항을 적습니다.
