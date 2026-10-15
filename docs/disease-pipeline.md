# 이미지 분류와 Qwen2.5 성분 추천 연결

2026-10-15 구현. 기존 미커밋 학습·데이터 개선 작업은 유지했다.

팀 전체의 구성은 **Swift iOS 앱 → Spring Boot → FastAPI → DenseNet121 → Qwen2.5**다.
앱은 Spring Boot만 호출하고, 백엔드는 이미지 분류 후 추천을 별도 작업자로 요청한다.
AI의 현재 API와 백엔드·앱에서 구현할 작업 상태·시간 제한·화면 처리는
[개정 수행계획서와 부록](project/09-revised-project-plan.md)에 정리했다.
[실행 가능한 JSON](fixtures/disease-pipeline/README.md)과 [실제 품질 점검의 범위·한계](disease-pipeline-quality-20261015.md)를 함께 확인한다.

## 새로 내려받은 팀원의 빠른 연동

AI/ 루트에서 `python -m venv .venv`로 환경을 만든 뒤 진행한다.
GPU·모델·원본 데이터가 없어도 JSON 계약은 아래 규칙 기반 개발 모드로 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:AI_INTERNAL_TOKEN = "<팀 내부에서 정할 개발용 값>"
$env:REC_BACKEND = "mock"
$env:DISEASE_BACKEND = "disabled"
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

다른 터미널에서 위와 같은 토큰을 설정하고 [fixture 호출 예](fixtures/disease-pipeline/README.md)를 실행한다.
이 모드에서는 `/infer/disease-recommend`의 JSON을 시험하며 사진 분류는 비활성이다.
새 clone에서도 내장 규칙 성분을 구조화하도록 원격 `fix/1-mock-ingredients`의 데이터 없는 mock 수정 취지를 반영했다.

실제 사진 분류에는 아래 DenseNet 의존성과 Git에 포함된 `model/weights/densenet121_s2.pt`가 필요하다.
실제 Qwen에는 별도로 준비할 `outputs/rec_sllm_v2/adapter/`와
`data/processed/sllm/{ingredient_dict.json,concern_top_ingredients.json}`가 추가로 필요하다.
161MB 규모 어댑터와 학습 데이터는 일반 Git 커밋에 포함하지 않는다. AI 담당 서버를 호출하거나
AI 담당자가 접근 권한·이용 조건에 맞춰 제공한 추론 산출물을 같은 경로에 준비한다.

## 실행 흐름

```text
사진 + 설문
  → DenseNet121 (CPU, 6종 분류)
  → 분류 점수/불확실성 확인
  → Qwen2.5 학습 어휘로 고민 변환 + 분류 참고 정보 추가
  → Qwen2.5-7B + 기존 rec_sllm_v2 LoRA (GPU, 4-bit)
  → 기존 출력 필터 → 성분/관리법 구조화 → JSON
```

`feat/3-skin-disease-classifier`의 커밋 `3ea9506bf72f096b3e2bd545f241a8c57fcbfe46`에서
`model/`을 가져왔다. 배포 가중치는 `model/weights/densenet121_s2.pt`다.
기존 분류기의 클래스 순서, DenseNet121 구조, 256×256 resize, ImageNet 정규화를 재현한다.
원본 추론 모듈이 `sys.path`와 전역 `config`, `model` 모듈을 수정하므로
서버에서는 충돌이 없는 `server/disease_classifier.py`로 같은 가중치를 읽는다.
추가 학습이나 가중치 변경은 하지 않았다.

## 실행

프로젝트 루트 `AI/`의 PowerShell 기준이다. 기존 CUDA PyTorch를 유지한다.
이 PC에서 확인한 조합은 `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`다.

```powershell
# 현재 PC의 torch 2.11.0+cu128에 맞는 torchvision만 추가
.\.venv\Scripts\python.exe -m pip install --no-deps torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "Pillow>=10.2,<13"

# 기존에 설정한 내부 토큰 사용. 없으면 백엔드와 공유하는 값을 설정
$env:AI_INTERNAL_TOKEN = "<백엔드와 공유할 비밀값>"
.\scripts\run_disease_pipeline.ps1
```

다른 torch 버전에서는 그 버전에 맞는 torchvision을 설치한다.
원본 `model/requirements.txt`의 오래된 torch/CUDA 버전을 현재 가상환경에 덮어 설치하지 않는다.
새 환경은 호환되는 torch/torchvision을 먼저 설치하고 `requirements-train.txt`,
`requirements-disease.txt`를 설치한다.

런처는 기본적으로 Qwen2.5 `rec_sllm_v2` 어댑터와 CPU 분류기를 사용한다.
`-AdapterDir outputs/rec_sllm_v3/adapter`로 다른 **Qwen2.5** 어댑터를 명시할 수도 있다.
기존 서버 기본 어댑터는 변경하지 않는다. GPU가 없는 개발 환경은 `-RecBackend mock`을 사용한다.
기본 바인딩은 `127.0.0.1:8000`이며 외부 백엔드 연결 시 `-ListenAddress 0.0.0.0`을 명시한다.

직접 uvicorn을 실행하는 경우 아래 환경변수를 지정한다.

| 변수 | 값/기본값 | 역할 |
|---|---|---|
| `REC_BACKEND` | `sllm` | 실제 Qwen 추천 사용 |
| `REC_BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 베이스 모델 |
| `REC_ADAPTER_DIR` | `outputs/rec_sllm_v2/adapter` | 기존 파인튜닝 어댑터 |
| `DISEASE_BACKEND` | `densenet` (일반 서버 기본은 `disabled`) | 분류기 활성화 |
| `DISEASE_WEIGHTS` | `model/weights/densenet121_s2.pt` | 가중치 경로 |
| `DISEASE_DEVICE` | `cpu` | Qwen과 VRAM을 공유하지 않음 |
| `DISEASE_THRESHOLD` | `0.6` | 원본 분류기의 추천 보류 임계값. 임상적으로 보정된 값 아님 |

`HF_HUB_CACHE`는 `HF_HOME`보다 우선한다. 기존 모델이 `D:\hf_cache\hub`에 있는데
다른 캐시 환경변수가 남아 있다면 `HF_HUB_CACHE=D:\hf_cache\hub`로 지정한다.
새 런처는 해당 캐시의 기존 Qwen2.5 모델이 있으면 재사용한다.

## API

보호된 모든 API는 기존과 같이 `X-Internal-Token` 헤더가 필요하다.
이미지는 JPEG/PNG, 기존 최대 용량은 10MB다. 업로드 이미지를 파일에 저장하지 않는다.

| API | 입력 | 용도 |
|---|---|---|
| `POST /infer/disease-analyze` | multipart `image`, `survey` JSON 문자열 | 사진에서 분류와 추천을 한 번에 실행 |
| `POST /infer/disease` | multipart `image` | 6종 분류 결과만 반환 |
| `POST /infer/disease-recommend` | JSON `disease`, `survey`, 선택 `metrics` | 별도 분류 서버의 출력 JSON을 Qwen으로 전달 |
| `POST /infer/analyze` | 기존 multipart 입력 | 분류기가 활성화되면 기존 측정 응답에 분류·추천도 포함 |

사진에서 한 번에 실행하는 예:

```powershell
$survey = '{"gender":"female","age":30,"skinType":"dry","selfReportedConcerns":[]}'
$survey | Set-Content -Encoding utf8 survey.json
curl.exe http://127.0.0.1:8000/infer/disease-analyze `
  -H "X-Internal-Token: $env:AI_INTERNAL_TOKEN" `
  -F "image=@face.png" -F "survey=<survey.json"
```

별도 분류 결과를 받는 요청 예:

```json
{
  "disease": {
    "label": "아토피",
    "label_en": "atopic",
    "confidence": 0.9,
    "uncertain": false,
    "probabilities": {"건선":0.02,"아토피":0.9,"여드름":0.02,"정상":0.02,"주사":0.02,"지루":0.02}
  },
  "survey": {"gender":"female","age":30,"skinType":"dry"},
  "metrics": null
}
```

6개 클래스 확률, 합계, 최상위 라벨, 한글·영문 라벨, confidence 일치 여부를 검증한다.
NaN/무한대, 음수, 범위 밖 값, 모르는 라벨은 `422 E4002`다.
외부 JSON이 `uncertain:false`여도 confidence가 서버 임계값보다 낮으면 추천하지 않는다.

응답의 주요 필드:

```json
{
  "disease": {"label":"아토피","label_en":"atopic","confidence":0.9,"uncertain":false,"probabilities":{},"model_version":"densenet121_s2_3ea9506","inference_ms":0},
  "recommendation_status": "generated",
  "recommendation": {"ingredients":[],"routine":[],"answer":"...","used_fallback":false,"model_version":"rec_sllm_v2"},
  "guidance": ["이미지 분류 참고 결과이며 의학적 진단이 아닙니다."]
}
```

위 응답은 구조를 간략히 보여주는 예시다. 실제 응답은 6개 확률과 기존 추천 필드를 모두 포함한다.

`recommendation_status`가 `uncertain`이면 자동 추천을 보류하고 `recommendation:null`과 안내를 반환한다.
정상 분류이며 선택한 고민/실제 측정값도 없으면 `no_concern`과 `recommendation:null`이다.
정상이어도 사용자가 고른 지원 고민이 있으면 그 고민에 대해 추천한다.
분류기를 비활성화한 상태에서 사진 분류 API를 호출하면 `503 E3001`이며 가짜 분류 결과를 만들지 않는다.

기존 `/infer/analyze`도 분류기 활성화 시 이 보류 정책을 적용하므로 프론트는
`recommendation:null`을 처리해야 한다. 분류기 비활성화 시 기존 추천 경로를 유지한다.
`skin.is_stub:true`인 측정값은 추천 입력으로 사용하지 않는다.

## 분류와 추천 사이의 매핑

| 이미지 라벨 | Qwen에 우선 전달하는 고민 |
|---|---|
| 건선 | 민감성(트러블/자극감) |
| 아토피 | 민감성(트러블/자극감) |
| 여드름 | 여드름/뾰루지 |
| 정상 | 사용자 자가응답/실제 측정값만 사용 |
| 주사 | 붉어짐(홍조), 민감성(트러블/자극감) |
| 지루 | 민감성(트러블/자극감) |

매핑은 `configs/disease_mapping.yaml` 한 곳에 있으며 서버의 라우팅 규칙이다.
아토피·건선·지루를 여드름이나 각질 제거 대상으로 자동 취급하지 않는다.
분류 라벨과 점수, 진단이 아니라는 점, 순한 보조 관리 범위를 `RecInput.user_note`에 추가하므로
Qwen은 **분류 결과를 실제 입력으로 받는다**. 응답에 라벨만 붙이는 방식이 아니다.

기존 Qwen의 학습 목표는 피부 고민별 성분 추천이며 질환별 치료 데이터로 재학습하지 않았다.
이 매핑이나 기존 데이터의 성분 효능을 임상적으로 검증한 것도 아니다.
보습/자극 최소화 안내는 [AAD 아토피 피부 관리](https://www.aad.org/public/diseases/eczema/types/atopic-dermatitis/atopic-dermatitis-coping)와
[AAD 주사 피부 관리](https://www.aad.org/public/diseases/rosacea/triggers/tips)를 참고했다.
기존 표현 필터는 그대로 적용하고 전문의 상담 안내는 서버에서 별도로 반환한다.
질환 라벨 연동에서는 실제 시험에서 관찰된 "민감성 피부에도 안전하게 사용" 같은
무조건적 안전 보장 문장도 추가로 제거한다. 모든 과장 표현을 판별하는 의학적 검증기는 아니다.
표현 필터 통과가 성분의 의학적 적합성 검증을 의미하지 않는다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q

# 실제 DenseNet 가중치와 원본 전처리 재현 검증
$env:RUN_DISEASE_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_disease_classifier.py -q

# 네트워크/외부 서버 없이 두 실제 모델을 한 프로세스에서 호출
$env:HF_HOME = "D:\hf_cache"
$env:HF_HUB_CACHE = "D:\hf_cache\hub"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts/check_disease_pipeline.py --synthetic --rec-backend sllm
```

`--synthetic`은 단색 이미지로 코드 연결만 확인한다. 사람 사진이 아니며 임상 정확도를 검증하지 않는다.
정상·불확실·다른 질환·얼굴 아닌 입력을 확실히 구분하는 OOD/얼굴 검출기가 추가된 것은 아니다.
6종 분류기는 얼굴 아닌 입력에도 높은 점수를 줄 수 있으므로 score 임계값만으로 안전한 입력임을 보증하지 않는다.
실제 결과는 `outputs/disease_pipeline_smoke.json`에 저장되며 원본 사진은 저장하지 않는다.

2026-10-15 로컬 실행 결과: HTTP 200, 실제 `densenet121_s2_3ea9506` +
`rec_sllm_v2`, 규칙 폴백 없음. 분류 약 40ms, Qwen 생성 약 42.2초.
단색 이미지가 정상으로 분류돼 설문의 건조 고민으로 추천을 생성했다.
고점수 정상 판정이 실제 정상 피부의 증거라는 뜻이 아니다.

아토피 라벨·점수 0.9를 **시험용 예시 JSON**으로 준 2단계 API도 HTTP 200,
`rec_sllm_v2`, 규칙 폴백 없이 약 47.5초에 민감성 고민 추천을 생성했다.
이 경우는 실제 환자 분류 결과가 아닌 문맥 전달 시험이다. 원문 결과는
`outputs/disease_atopic_smoke.json`에 있다. 이 출력에서 관찰한 안전 보장 문구를
추가 필터 회귀 테스트에 넣었다.
