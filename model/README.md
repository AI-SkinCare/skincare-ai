# 안면부 피부질환 6종 분류 — NIA 베이스라인 재현

AI Hub「안면부 피부질환 이미지 합성 데이터」(dataSetSn=71863), DenseNet121 베이스라인.
클래스: 건선[0], 아토피[1], 여드름[2], 정상[3], 주사[4], 지루[5]

## 파일 구성
| 파일 | 역할 |
|---|---|
| `src/config.py` | 경로·하이퍼파라미터·클래스 매핑 (경로는 환경변수/CLI로 덮어쓰기) |
| `src/prepare_data.py` | 라벨 JSON 재귀 수집 → 층화 8:1:1 분할 → `splits/{train,val,test}.csv` |
| `src/dataset.py` | Dataset/DataLoader, train/eval transform |
| `src/model.py` | DenseNet121 (ImageNet pretrained) + Dropout 0.5 + Linear(1024→6) |
| `src/train.py` | 학습, `results/<run>/train_log.csv`, `checkpoints/<run>/{best,last}.pt` |
| `src/inference.py` | 단일 이미지 추론 → JSON (서버 연동용) |
| `src/evaluate.py` | ACC, 클래스별 P/R/F1, confusion matrix(CSV+PNG), 정면/측면별 ACC → `results/<run>/eval_test/` |

## 추론 (서버 연동용)
`src/inference.py` — 이미지 1장 → JSON. 가중치: `weights/densenet121_s2.pt` (28MB, test ACC 94.91%)

```python
import sys; sys.path.append("<repo>/src")
from inference import SkinClassifier

clf = SkinClassifier()              # 서버 시작 시 1회 로드. GPU 있으면 자동 사용 (device="cpu" 로 강제 가능)
result = clf.predict(image)         # 파일 경로 / bytes(업로드 바이트) / PIL.Image
```
```bash
python src/inference.py --image face.png   # CLI 테스트
```
출력:
```json
{
  "label": "아토피",
  "label_en": "atopic",
  "confidence": 0.9412,
  "uncertain": false,
  "probabilities": {"건선": 0.0213, "아토피": 0.9412, "여드름": 0.0101, "정상": 0.0098, "주사": 0.0105, "지루": 0.0071}
}
```
| 필드 | 의미 |
|---|---|
| `label` / `label_en` | 예측 클래스 (건선·아토피·여드름·정상·주사·지루 중 하나) |
| `confidence` | 예측 클래스의 확률 (0~1) |
| `uncertain` | `confidence < 0.6` 이면 true. 모델은 학습에 없는 질환·흐린 사진·비얼굴 사진도 6개 중 하나로 답하므로, true 일 땐 재촬영 안내 등 별도 처리 권장 (`SkinClassifier(threshold=...)` 로 조정) |
| `probabilities` | 6개 클래스 전체 확률 |

추론에 필요한 패키지: torch, torchvision, Pillow, pandas (`requirements.txt` 버전 기준)

주의: 정면 사진 정확도가 측면보다 낮음 (seed1 기준 정면 86.7% / 측면 98.5%), 지루↔주사 혼동이 가장 많음.

## 베이스라인 스펙 (공식값)
DenseNet121 pretrained / 8:1:1 층화(seed 42) / image 256 / batch 32 / epoch 11 / Adam lr 1e-3 / dropout 0.5
목표 test ACC ≥ 80% (공식 85.17%)

## 환경
```bash
conda activate skin-disease
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 데이터 경로 지정
우선순위: CLI 인자 > 환경변수 > 기본값(`./data`)
- `--data-root` / `SKIN_DATA_ROOT` : 이미지·라벨 루트를 한 번에
- `--image-root` / `SKIN_IMAGE_ROOT` : JSON의 `photograph.file_path`(예: `여드름/정면/xxx.png`)가 붙는 기준 디렉토리
- `--label-root` / `SKIN_LABEL_ROOT` : `.json`을 재귀 수집할 디렉토리

`image_root/file_path` 가 없으면 (상위 폴더 구조 차이, NFD 한글 파일명 등) image_root 아래 파일명으로 자동 탐색한다
(`lesion_area/` 마스크는 제외). JSON 배치(클래스별 1파일/이미지별 1파일)는 무관.

## 실행 순서 (항상 프로젝트 루트에서 실행)
```bash
# 1) 분할 생성
python src/prepare_data.py --image-root <원천데이터 경로> --label-root <라벨링데이터 경로>

# 2) 학습 (공용 GPU 메모리가 부족하면 --memory-efficient 추가: 결과 동일, 속도만 느려짐)
python src/train.py --gpu 0 --run-name baseline_s42

# 3) 평가
python src/evaluate.py --run-name baseline_s42 --gpu 0
```

## 실데이터 도착 후 체크리스트
- [ ] 압축 해제 후 디렉토리 구조 확인 (`tree -d -L 3`), 이미지 루트·라벨 루트 결정
- [ ] 한글 파일명 깨짐 여부 확인 (zip이 cp949면 `unzip -O cp949`)
- [ ] JSON 하나 열어서 스키마가 명세와 같은지 확인 (`diagnosis_info.diagnosis_name`, `photograph.file_path`)
- [ ] `python src/prepare_data.py --image-root ... --label-root ...` 실행
  - [ ] 클래스별 2000장 / 정면·측면 각 1000장 맞는지 (경고 메시지 확인)
  - [ ] "이미지 파일을 찾지 못함", "알 수 없는 클래스명", "읽기 실패 JSON" 경고가 있으면 원인 확인
  - [ ] 분할이 9600 / 1200 / 1200 인지 확인
- [ ] 샘플 이미지 몇 장 열어서 라벨과 맞는지 육안 확인
- [ ] 빈 GPU 확인 (`nvidia-smi`) → 여유 메모리 부족 시 `--memory-efficient`
- [ ] `python src/train.py --gpu <id> --run-name baseline_s42` (11 epoch)
- [ ] `python src/evaluate.py --run-name baseline_s42` → test ACC ≥ 80% 확인 (공식 85.17%)
- [ ] confusion matrix에서 혼동이 큰 클래스 쌍, 정면/측면별 ACC 차이 기록
- [ ] (선택) seed 바꿔 2~3회 반복해 편차 확인
