# AI 모듈 가이드 — 성분 추천 sLLM · 추론 서버

담당: 김우주 (성분 추천 sLLM 파인튜닝, 추론 서버 구동)
연결: 박성근 (피부 상태 회귀 모델 → `server/skin_regressor.py`)

이 문서는 조장 PC(Windows, RTX 5070 Ti 16GB)에서 학습부터 서빙까지 그대로 따라 하는 순서입니다.
모든 명령은 **레포 루트(`AI\`)의 PowerShell** 기준입니다.

---

## 1. 현재 상태

| 단계 | 상태 | 산출물 |
|---|---|---|
| 데이터 준비 | 완료 | `data/processed/sllm/` (train 8,258 / val 1,000 / val_eval 201), `docs/label_spec.md` |
| 규칙 베이스라인 평가 | 완료 (201건) | precision@3 **68.8%**, 정답 유사도 **0.143** |
| 원본 모델 평가 | 완료 (60건) | precision@3 **0.9%**, 형식 준수 78.3% |
| v1 학습·평가 | 완료 (201건) | `outputs/rec_sllm_v1/adapter` — precision@3 78.0%, 필터 개입 35.8% |
| 학습 정답문 정제 | 완료 | `data/processed/sllm_v2/` (§3 ⑧) |
| **v2 학습·평가** | **완료 (201건) — 서빙 기본값** | `outputs/rec_sllm_v2/adapter` — precision@3 **78.4%**, 필터 개입 **0.5%** |
| 비교표 | 완료 | `outputs/eval/comparison.md` |
| 서버 | mock 모드 동작 (테스트 통과) | `-Backend sllm`이면 v2 어댑터 사용 |

### 결과 요약 (201건, 같은 문항 비교 포함)

| 모델 | precision@3 | recall | 정답 유사도 | 형식 준수 | 필터 개입 | 응답 p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| 규칙 베이스라인 | 68.8% | 51.0% | 0.143 | 100% | 0% | 즉시 |
| 원본 Qwen2.5-7B (60건) | 0.9% | 0.5% | 0.241 | 78.3% | 67.9% | 19.4초 |
| v1 | 78.0% | 77.3% | 0.508 | 99.5% | 35.8% | 49.3초 |
| **v2** | **78.4%** | **79.3%** | 0.482 | **100%** | **0.5%** | 48.1초 / 약 61초 |

- 같은 문항끼리 비교하면 v2는 규칙 베이스라인보다 precision@3 **+9.7%p (95% CI ±5.1)** 높습니다. 통계적으로 유의합니다.
- v1과 v2의 precision@3 차이(0.4%p)는 오차 범위 안이라 같은 수준입니다. v2의 이점은 **필터 개입 35.8% → 0.5%**입니다.
- v2의 정답 유사도가 조금 낮은 것은 기준 정답문이 원본(영문 INCI명·단정 표현 포함)이기 때문입니다.
- 입력 정보(성별·나이·피부타입·외부요인)로는 정답 성분이 거의 결정되지 않습니다. 입력 조합별 최빈 성분으로 예측해도 precision@3는 0.65~0.68에 그칩니다. 따라서 precision@3를 더 끌어올릴 여지는 작고, 개선은 표현 품질·안전·속도 쪽이 효과적입니다.

---

## 2. 설치 (최초 1회)

### 2.1 가상환경

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2.2 PyTorch — RTX 50 시리즈는 CUDA 12.8 이상 빌드가 필수

RTX 5070 Ti는 Blackwell 아키텍처(compute capability 12.0)입니다.
CUDA 12.1/12.4용 torch는 **설치와 import는 되지만 GPU 연산에서 `no kernel image is available` 오류**가 납니다.
반드시 cu128 이상 인덱스에서 설치하세요.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> [pytorch.org](https://pytorch.org/get-started/locally/)에서 더 높은 CUDA 버전(cu129, cu130 등)을 안내하면 그것을 써도 됩니다. 12.8 미만만 피하면 됩니다.

### 2.3 나머지 패키지 · 모델 캐시 위치

```powershell
pip install -r requirements-train.txt -r requirements-dev.txt
setx HF_HOME "D:\hf_cache"          # 7B 모델(약 15GB)을 D드라이브에 저장. 새 터미널부터 적용
$env:HF_HOME = "D:\hf_cache"        # 현재 터미널에도 즉시 적용
```

`Qwen/Qwen2.5-7B-Instruct`는 Apache-2.0 라이선스로 Hugging Face 로그인 없이 받을 수 있습니다.

### 2.4 환경 점검

```powershell
python scripts/check_env.py
```

import만 확인하는 게 아니라 **GPU에서 실제로 bf16 행렬곱과 4-bit(nf4) 연산을 한 번 실행**합니다.
`모든 점검 통과`가 나와야 다음 단계로 넘어갑니다.

---

## 3. 실행 순서

### 가장 쉬운 방법 — `run_pipeline.bat` 더블클릭

레포 루트의 **`run_pipeline.bat`을 더블클릭**하면 §2 설치부터 아래 ②~⑥까지 전부 순서대로 실행합니다.

- 가상환경(`.venv`) 생성 → torch(CUDA 12.8) 포함 패키지 설치 → 환경 점검 → 스모크 → 원본 모델 평가 → 본 학습(1에폭) → 파인튜닝 모델 평가 → 비교표
- 전체 **5~8시간 추정** (대부분 학습). 퇴근 전이나 자기 전에 시작하는 것을 권장합니다. 실행 중에는 절전 모드로 들어가지 않습니다.
- **창을 닫지 마세요.** 닫혔거나 PC가 꺼졌으면 다시 더블클릭만 하면 됩니다. 끝난 단계는 건너뛰고, 학습은 마지막 체크포인트부터 이어집니다.
- 진행 상황: `outputs/run_logs/status.json`, 전체 로그: `outputs/run_logs/pipeline.log`. Claude가 이 파일을 읽어 진행 상황 확인과 오류 진단을 할 수 있습니다.

| 옵션 (PowerShell에서 `.\run_pipeline.bat <옵션>`) | 의미 |
|---|---|
| `--status` | 진행 상태만 출력 |
| `--epochs 2` | 2에폭 학습 |
| `--from train` | 학습부터 다시 (이후 단계 포함) |
| `--only eval_base` | 한 단계만 실행 |

아래는 단계를 하나씩 직접 실행할 때의 명령입니다.

### ① 데이터 준비 (완료됨 — 데이터를 다시 받았을 때만)

```powershell
python -m src.llm.prepare_data --raw-root "D:\03.스킨케어 성분-효능 추천 데이터\3.개방데이터\2.데이터(NIA)"
```

### ② 스모크 테스트 (약 5분)

0.5B 모델로 10스텝만 돌려 GPU 경로(4-bit, bf16, paged optimizer)가 끝까지 도는지 봅니다.

```powershell
python -m src.llm.train_qlora --smoke
```

### ③ 원본 모델 평가 — 파인튜닝 효과를 증명할 기준선

```powershell
python -m src.llm.evaluate --base Qwen/Qwen2.5-7B-Instruct --n 60
```

- 1건당 30~75초로 추정되므로 201건 전체는 2~4시간입니다. 기준선은 `--n 60`(약 30~75분)으로 충분합니다.
- `--n`은 **고정 시드 무작위 표본**이라, 비교할 모델끼리 같은 값을 쓰면 같은 문항으로 평가됩니다.
- 원본 모델은 이 데이터의 성분 어휘(알로에신, 댕댕이나무열매즙 등)를 모르므로 precision@3가 낮게 나오는 게 정상입니다. 그 격차가 곧 파인튜닝 효과입니다.

### ④ 본 학습

```powershell
python -m src.llm.train_qlora --epochs 1
```

- **처음엔 1에폭만** 돌리세요. 1에폭 결과로 방향이 맞는지 확인한 뒤 2에폭을 결정하는 게 시간 대비 안전합니다.
- 학습 중에는 **추론 서버를 끄세요.** 같은 GPU를 두고 경쟁하면 OOM이 납니다.
- 소요 시간: 로그의 `s/it` × 516 (= 8,258건 ÷ 실효 배치 16) = 1에폭 시간.
  대략 **1에폭 3~5시간**으로 추정되지만, 첫 10스텝 로그로 실제 값을 확인하세요.
- 중단됐으면 `--resume`으로 마지막 체크포인트(10스텝마다 저장)부터 이어집니다. `run_pipeline.bat`과 `scripts\train_v2.bat`은 학습이 죽으면 자동으로 이어서 재시도합니다.
- 실측: 1스텝 약 32~38초, 1에폭(505스텝) 약 4.5시간.
- 시작 로그의 `토큰 길이 p50/p95`와 `길이 초과 제외` 건수를 기록해 두세요 (발표 자료용).

### ⑤ 파인튜닝 모델 평가

```powershell
python -m src.llm.evaluate --adapter outputs/rec_sllm_v1/adapter --n 60     # 원본 모델과 같은 n
python -m src.llm.evaluate --adapter outputs/rec_sllm_v1/adapter            # 여유 있을 때 201건 전체
```

### ⑥ 비교표 생성 (중간발표 자료)

```powershell
python -m src.llm.compare
```

`outputs/eval/comparison.md`에 모델별 최신 결과가 **규칙 베이스라인 → 원본 모델 → 파인튜닝 모델** 순으로 정리됩니다.
평가 건수가 모델마다 다를 수 있으므로, 규칙 베이스라인과의 우열은 **"같은 문항끼리 비교" 표**(문항별로 짝지은 차이와 95% 신뢰구간)로 판단하세요.
이 표는 문항별 결과(`items`)가 저장된 평가 파일끼리만 계산됩니다.

### ⑧ v2 — 학습 정답문 정제 후 재학습

v1 평가에서 서빙 필터가 답변의 36%에 개입했습니다. 원인은 학습 정답문의 약 40%에 들어 있는 효능 단정 표현("개선합니다")이었고,
2.2%에는 의료 표현(치료·질환·피부염)이, 15.8%에는 영문 INCI명이 섞여 있었습니다. 서빙 필터와 같은 규칙을 **학습 데이터에 미리 적용**해 해결합니다.

```powershell
python -m src.llm.clean_targets      # data/processed/sllm → data/processed/sllm_v2 (train만 정제, 검증셋은 원본 복사)
.\scripts\train_v2.bat               # 1에폭 학습(자동 재시도) → 201건 평가 → 비교표. 로그: outputs\run_logs\v2.log
```

| 정제 내용 | 예 |
|---|---|
| 영문 성분명 → 한글 대표명, 조사 교정 | `'SULFUR(유황)'는` → `'유황'은` |
| 의료 표현 바꿔 쓰기 | 치료 → 관리, 피부 질환 → 피부 고민, 피부염 → 피부 자극 |
| 효능 단정 표현 완화 (`filter_rules.yaml`의 `soften`) | 억제합니다 → 억제하는 데 도움을 줄 수 있습니다 |

정제 통계는 `data/processed/sllm_v2/clean_stats.json`에 남습니다. `filter_rules.yaml`을 바꾸면 정제·재학습을 다시 해야 모델에 반영됩니다(서빙 필터에는 즉시 반영).

### ⑦ 서버 실행

```powershell
$env:AI_INTERNAL_TOKEN = "<백엔드와 공유할 비밀값>"
.\scripts\run_server.ps1                 # mock: 규칙 기반 (GPU 불필요, 연동 개발용)
.\scripts\run_server.ps1 -Backend sllm   # 파인튜닝 모델 (기본 v2 어댑터. v1은 REC_ADAPTER_DIR=outputs/rec_sllm_v1/adapter)
```

확인: `http://localhost:8000/health`, API 문서: `http://localhost:8000/docs`

---

## 4. 평가 지표 읽는 법

| 지표 | 의미 | 파인튜닝 모델에 기대하는 방향 |
|---|---|---|
| 형식 준수(1차) | 재시도 없이 JSON 형식을 지킨 비율 | 95% 이상 |
| 폴백률 | 두 번 모두 형식 실패 → 규칙 기반 답으로 대체된 비율 | 5% 이하 |
| precision@3 | 추천 상위 3개 성분 중 정답 성분 비율 (**모델이 직접 만든 답만**) | 원본 모델보다 크게 높고, 규칙 베이스라인(68.8%) 이상 |
| 정답 유사도 | 정답 추론문과의 글자 2-gram F1 | 규칙 템플릿(0.143)보다 높게 |
| 필터 개입률 | 금칙 표현(치료·완치 등) 필터가 개입한 비율 | 낮을수록 좋음 |
| 응답 p50 | 1건 생성 시간 | 측정해서 백엔드 타임아웃 결정에 사용 (§6) |

**왜 precision@3만으로는 부족한가.** 이 데이터는 고민별 추천 성분이 거의 고정되어 있습니다(미백 → 알로에신·헥사펩타이드-2·댕댕이나무열매즙이 대부분).
그래서 "고민별로 자주 나온 성분 3개"라는 단순 규칙만으로 68.8%가 나옵니다. 파인튜닝 모델의 가치는 성분 맞히기가 아니라
**사용자 조건(나이·피부 타입·외부 요인)에 맞춘 추론 과정을 생성하는 것**이고, 이는 정답 유사도와 원본 모델 대비 개선폭으로 보여줘야 합니다.

**폴백은 점수에서 뺍니다.** 폴백 답(=규칙 베이스라인)까지 점수에 넣으면 형식을 매번 어기는 망가진 모델도 68.8% 근처를 받아 실패가 가려집니다.
폴백을 포함한 "사용자가 실제로 받는 결과" 기준 수치는 `precision_at_3_with_fallback`에 따로 남습니다.

### 수행계획서 성능 목표와의 관계 (중간발표 전 조정 필요)

| 수행계획서 목표 | 실제 상황 | 발표 시 제안 |
|---|---|---|
| 추천 성분 Top-3 일치율 70% 이상 | 규칙 베이스라인이 이미 68.8% | "원본 모델 대비 +N%p, 규칙 베이스라인 이상"으로 상대 목표화 |
| 단일 요청 응답 3초 이내 | 추천 생성은 실측 p50 48초 (약 1,050토큰) | 피부 측정(`/infer/skin-metrics`)만 3초 목표, 추천은 비동기 처리로 명시 |

---

## 5. 설정 파일

| 파일 | 담당 | 내용 |
|---|---|---|
| `configs/sllm_qlora.yaml` | 김우주 | 데이터 경로, 베이스 모델, LoRA·학습 하이퍼파라미터 (v1) |
| `configs/sllm_qlora_v2.yaml` | 김우주 | v2: 정제 데이터(`sllm_v2`)·출력(`rec_sllm_v2`) 경로만 다름 |
| `configs/filter_rules.yaml` | 손승우(규제 검토) | 금칙 표현·완화 규칙·면책 문구 — 코드 수정 없이 이 파일만 수정 |
| `configs/serving_mapping.yaml` | 김우주 ↔ 박성근 | 측정값(0~1) → 피부 고민 변환 임계값. `[확인필요]` 값은 회귀 모델 출력 분포를 보고 보정 |
| `.env` (`.env.example` 복사) | 김우주 | 내부 토큰, 백엔드 모드, 타임아웃 — **커밋 금지** |

---

## 6. 백엔드 연동 계약 — 이현승·정도희에게 전달할 것

개발명세서 §5.3의 **"백엔드 → AI 서버 타임아웃 20초"는 추천 API에 맞지 않습니다.**
7B 4-bit 모델이 약 1,050토큰을 생성하는 데 **실측 p50 48초, p95 약 61초**가 걸립니다(v2, 201건).

| 엔드포인트 | 예상 소요 | 백엔드 HTTP 타임아웃 | 호출 방식 |
|---|---|---|---|
| `POST /infer/skin-metrics` | 수 초 | 20초 | 동기 호출 가능 |
| `POST /infer/recommend` | 48초(p50) / 61초(p95) 실측 | **130초 이상** | 백그라운드 작업에서 호출 |
| `POST /infer/analyze` | 위 두 개의 합 | **130초 이상** | 백그라운드 작업에서 호출 |

- 백엔드는 이미 `POST /analyses` → `202` + 폴링 구조이므로, **AI 호출을 요청 처리 스레드가 아닌 백그라운드 작업에서** 하면 됩니다.
- 프론트엔드 폴링 제한(명세서 §4.2의 30초)은 **150초 이상**으로 늘려야 합니다. 로딩 화면에 "맞춤 추천을 작성하는 중(최대 2분)" 같은 안내를 권장합니다.
- GPU는 한 번에 한 건씩 생성합니다(`REC_CONCURRENCY=1`). 대기열(`QUEUE_LIMIT=20`)이 차면 `503 E3003`을 반환하니 잠시 후 재시도하세요.
- AI 서버 자체 타임아웃은 `REC_TIMEOUT_S=120`이며 초과 시 `504 E3002`입니다.

### 명세서 §2.3 에러 코드표에 추가할 항목

| 코드 | HTTP | 의미 |
|---|---|---|
| E3003 | 503 | AI 서버 대기열 가득 참 (재시도 대상) |
| E4002 | 422 | 요청 형식 오류 (설문 JSON·측정값 해석 불가) |

### 응답에서 백엔드·프론트가 확인해야 할 필드

- `skin.is_stub: true` — 박성근 회귀 모델 연결 전의 **가짜 측정값**입니다. 사용자에게 수치를 보여주면 안 됩니다.
- `recommendation.used_fallback: true` — 모델 대신 규칙 기반 추천이 나갔다는 뜻 (로그·모니터링용).
- `recommendation.input_notes` — 나이·피부 타입 등을 추정한 경우 그 내역.
- `recommendation.disclaimer` — 결과 화면에 항상 표시.

---

## 7. 외부 공개 (Cloudflare Tunnel)

AI 서버는 조장 PC에서 돌고, 백엔드는 정도희 자택 서버에서 돌기 때문에 인터넷을 통한 연결이 필요합니다.

```powershell
cloudflared tunnel --url http://localhost:8000
```

- 출력된 `https://….trycloudflare.com` 주소를 백엔드의 AI 서버 주소로 설정합니다.
- 이 임시 주소는 **cloudflared를 재시작할 때마다 바뀝니다.** 고정 주소가 필요하면 Cloudflare에 연결된 도메인으로 named tunnel을 만들어야 합니다.
- 모든 추론 API는 `X-Internal-Token` 없이는 401을 반환합니다. `/health`만 인증 없이 열려 있습니다.

---

## 8. 박성근 회귀 모델 연결 지점

1. `src/models/` 아래에 `version` 속성과 `predict(image_bytes, device_type) -> SkinPrediction`을 가진 클래스를 만듭니다.
2. `server/skin_regressor.py`의 `load_skin_regressor()` TODO 자리에서 반환하도록 연결합니다.
3. `.env`에서 `SKIN_BACKEND=model`로 바꿉니다.
4. 회귀 모델의 실제 출력 분포를 보고 `configs/serving_mapping.yaml`의 `[확인필요]` 임계값을 함께 보정합니다.
   두 데이터셋은 피험자가 달라서 측정값 → 피부 고민 변환은 이 규칙표 하나에 전적으로 의존합니다.

---

## 9. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `no kernel image is available for execution on the device` | CUDA 12.8 미만 torch | §2.2대로 cu128 이상으로 재설치 |
| bitsandbytes 4-bit 연산 실패 | 구버전 bitsandbytes | `pip install -U bitsandbytes` |
| 학습 중 `CUDA out of memory` | 긴 샘플 또는 서버 동시 실행 | 서버와 GPU를 쓰는 프로그램(브라우저·IDE 등) 종료. `max_seq_len`은 이미 2048로 낮춰 둠 |
| 학습 중 `CUDA error: unknown error` (4~8스텝 등 불규칙) | 16GB VRAM이 빠듯할 때 Windows 드라이버에서 발생 | 체크포인트(10스텝마다)에서 이어서 재시작. 실행 스크립트가 자동 재시도함 |
| `expandable_segments not supported` 경고 | Windows 미지원 옵션 | 무시. 대신 `max_split_size_mb:128` 사용 중 |
| 평가 결과에 `[경고] 폴백 …` | 형식 준수 실패 다수 | 학습 부족 가능성. 문항별 결과(`outputs/eval/*.json`의 `items`)와 `samples`에서 출력 확인 |
| 필터 개입률이 높음 | 학습 정답문에 단정·의료 표현이 많음 | §3 ⑧ 정제 후 재학습 |
| C드라이브 용량 부족 | 모델 캐시가 C에 저장 | `HF_HOME`을 D드라이브로 (§2.3) |
| HF 캐시 심볼릭 링크 경고 | Windows 기본 설정 | 무시해도 동작함 (디스크를 조금 더 씀) |
