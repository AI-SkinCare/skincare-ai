# skincare-ai

스마트폰 촬영 영상 기반 피부 상태 정량화 및 맞춤 성분 추천 AI 서비스의 **AI 모델 / 추론 서버** 저장소입니다.

## 역할

| 구성 | 내용 | 담당 |
|---|---|---|
| 성분 추천 sLLM | Qwen2.5-7B-Instruct를 「스킨케어 성분-효능 추천 데이터」로 QLoRA 4-bit 파인튜닝. 3단계 추론 과정과 함께 성분·관리법 생성 | 김우주 |
| 추론 서버 | FastAPI. 피부 측정 → 추천을 백엔드에 제공 | 김우주 |
| 피부 상태 회귀 모델 | 안면 이미지 → 수분·탄력·색소침착·모공 예측 | 박성근 |

> **서비스 범위**: 피부 상태 측정과 스킨케어 성분 추천(웰니스)까지입니다. 여드름·아토피 등 **질환 진단은 의료기기 영역이므로 다루지 않으며**, 생성 결과의 치료·효능 단정 표현은 `configs/filter_rules.yaml` 규칙으로 걸러냅니다.

## 빠른 시작

GPU 없이 API 연동만 확인할 때 (규칙 기반 mock 모드):

```powershell
pip install -r requirements-dev.txt
$env:AI_INTERNAL_TOKEN = "dev-token"
.\scripts\run_server.ps1
# → http://localhost:8000/docs
```

학습·평가 전체(설치 → 점검 → 원본 모델 평가 → QLoRA 학습 → 평가 → 비교표)는 **`run_pipeline.bat` 더블클릭** 한 번으로 실행됩니다. 중단돼도 다시 실행하면 이어갑니다. 자세한 내용은 **[docs/ai-module.md](docs/ai-module.md)** 를 참고하세요.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 헬스체크 (인증 불필요) |
| GET | `/model-info` | 로드된 모델 버전 |
| POST | `/infer/skin-metrics` | 이미지 → ROI별 피부 지표 (0~1) |
| POST | `/infer/recommend` | 설문 + 측정값 → 성분·관리법 추천 |
| POST | `/infer/analyze` | 위 두 단계 통합 |

모든 추론 API는 `X-Internal-Token` 헤더가 필요합니다. 추천 생성은 수십 초가 걸리므로 백엔드는 백그라운드 작업에서 호출해야 합니다 ([연동 계약](docs/ai-module.md#6-백엔드-연동-계약--이현승정도희에게-전달할-것)).

## 구조

```
configs/        학습 설정, 출력 필터 규칙, 서빙 매핑 규칙
docs/           ai-module.md (실행 가이드), label_spec.md (데이터에서 자동 생성한 라벨 명세)
scripts/        run_pipeline.py (전체 실행기), check_env.py (GPU 환경 점검), run_server.ps1
server/         FastAPI 추론 서버
src/llm/        데이터 준비 · QLoRA 학습 · 추론 · 평가 · 비교표
tests/          pytest (GPU 없이 실행 가능)
data/, outputs/ 가공 데이터·모델·평가 결과 (git 제외)
```

## 테스트

```powershell
pytest
```

## 이미지 분류 → Qwen2.5 성분 추천

6종 DenseNet121 분류 결과를 기존 Qwen2.5 어댑터의 피부 고민 입력으로 변환해 추천합니다.
실행·API·분류가 불확실할 때의 처리는 [파이프라인 가이드](docs/disease-pipeline.md)를 참고하세요.
**Swift 앱 → Spring Boot → FastAPI** 구성과 담당자별 변경은 [개정 수행계획서](docs/project/09-revised-project-plan.md),
요청·응답 예시는 [계약 fixture](docs/fixtures/disease-pipeline/README.md), 실제 시험의 한계는 [품질 점검](docs/disease-pipeline-quality-20261015.md)에 정리했습니다.
기존 `run_server.ps1`은 기존 모드로, 새 `scripts/run_disease_pipeline.ps1`은 두 모델을 함께 실행합니다.
질환 치료제나 진단을 제공하는 모델로 재학습한 것은 아니며, 추천 범위는 화장품 수준의 보조 관리입니다.

## 관련 저장소

- [skincare-frontend](https://github.com/AI-SkinCare/skincare-frontend) — 프론트엔드
- [skincare-backend](https://github.com/AI-SkinCare/skincare-backend) — 백엔드

팀 소개 및 팀원 역할은 [AI-SkinCare 조직 프로필](https://github.com/AI-SkinCare)에서 확인해 주세요.
