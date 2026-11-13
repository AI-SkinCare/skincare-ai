# 멘토 제안 모델과 Grad-CAM의 서비스 연결

2026-11-13. DINOv2-small 동결 + 선형 분류기, EfficientNet-B0, Grad-CAM의
로컬 연구 구현 중 독립 모델 학습·복원·설명에 필요한 코드를 서비스 PR로 분리한다.
평가 결과로 검증되지 않은 후속 보정 모델을 기본 분류기로 바꾸지 않는다.

`DISEASE_BACKEND=densenet|densenet121|efficientnet_b0|dinov2_linear`로
체크포인트 구조를 명시한다. 학습·가중치 파일·개인 사진은 Git에 포함하지 않는다.
DINOv2 인코더는 requires_grad=False와 eval 상태를 유지하고 작은 분류기만 학습한다.
체크포인트 복원은 weights_only=True와 클래스 순서 검사를 적용하며 네트워크 다운로드를
요청 처리 중 수행하지 않는다.

`/infer/disease`의 선택 form `include_explanation` 기본값은 false다.
true이면 같은 사진·모델의 Grad-CAM 히트맵을 `explanation`으로 함께 반환한다.
독립 `/infer/disease-explain`도 제공한다. 얼굴 원본·overlay를 포함하지 않는
256×256 grayscale PNG만 서비스 응답으로 전달한다. Grad-CAM을 만들 수 없으면
available=false이며 원래 분류·추천은 유지한다.

설명의 target_label·model_version은 분류와 일치해야 한다. 설명은 Qwen에
전달하지 않는다. Spring은 선택형 설명을 저장하고 Swift는 참고 영역으로 표시한다.
전체 앱 계약은 백엔드 `docs/gradcam-contract.md`에 기록한다.

## 동일 사진 비교와 적용 결정

학습된 기존 체크포인트를 CPU 4스레드로 새로 평가했다. 자료는 다운로드 폴더의
`test_sample_96` 이미지 96장과 `labels.csv`다. 클래스당 16장, 정면·측면 각 48장이다.
이 자료는 이미 여러 번 사용했으므로 새 환자 일반화 또는 독립 임상 시험이 아니다.
이번 작업에서는 재학습하지 않았다. 정답 라벨은 채점에만 사용한다.

| 모델 | 정답/96 | 정확도 | Macro F1 | 정면 정확도 | 건선 재현율 | 아토피 재현율 | 지루 재현율 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 DenseNet S2 | 90 | 93.75% | 0.9373 | 87.50% | 15/16 | 15/16 | 13/16 |
| 학습 DenseNet121 | 92 | 95.83% | 0.9583 | 91.67% | 14/16 | 15/16 | 15/16 |
| EfficientNet-B0 | 91 | 94.79% | 0.9480 | 91.67% | 15/16 | 14/16 | 14/16 |
| Frozen DINOv2-small + linear | 81 | 84.38% | 0.8518 | 70.83% | 13/16 | 14/16 | 12/16 |

전체 점수가 높은 DenseNet121도 건선 재현율이 감소했고 EfficientNet도 아토피
재현율이 감소했다. 사용자가 요구한 질환별 기존 성능 유지를 모두 만족하지 못하므로
기본 분류기는 기존 S2를 유지한다. DINOv2·EfficientNet은 명시적으로 선택할 수 있는
비교 후보로 제공한다. 후속 연구 보정·융합 모델은 이 서비스 변경에 포함하지 않는다.

실제 DINO 체크포인트에서 인코더 eval·동결과 학습 가능 파라미터 **4,614개**를 확인했다.
네 모델 각각 실제 사진 1장에 대한 HTTP Grad-CAM을 생성했고, 대상·분류 점수 일치와
가중치 불변을 검사했다. 이는 위치 정답과의 일치율 시험이 아니다.

## Swift → Spring → FastAPI 실제 연동

실제 앱의 Swift APIClient·DTO를 Linux Swift 6.1.3으로 컴파일하고, Spring HTTP와
FastAPI 실제 DenseNet S2·Qwen2.5-7B를 통해 사진 3건을 전송했다. 얼굴 사진은 메모리에서
JPEG로 변환했고 복사·외부 전송하지 않았다. 여성·30대·중성·자가 고민 없음 설문은
공통 가상 정보이며 사진 인적 정보가 아니다. 근거 경로의 기존 설정대로 LoRA는 비활성화했다.

| 경로 | 결과 | 생성 출처 | 전체 처리 시간 | 분류+히트맵 시간 |
|---|---|---|---:|---:|
| 근거 후보 성분 전부 회피 | quality_rejected / WITHHELD | not_generated | 1.34초 | 112ms |
| 정상·자가 고민 없음 | no_concern / WITHHELD | not_generated | 1.26초 | 86ms |
| 건선으로 분류된 사진 | generated / COMPLETED | source_cards | 38.52초 | 95ms |

추천 완료 건은 Qwen 경로 실행 후 근거 카드 대체 안내로 제공됐다. 성분은 글리세린·
징크옥사이드, 루틴은 5개다. 모델 원문 성공으로 집계하지 않는다. 처리 시간은 이 PC의
세 요청 값이며 전체 서비스 처리량 보장이 아니다. 성분 효능 정확도나 임상 검증은 측정하지 않았다.

세 건 모두 분류·모델 버전·히트맵이 접수와 최종 조회에서 동일했다. 히트맵의 실제 PNG
픽셀 디코딩까지 확인했고 크기는 약 6.7–7.4KiB였다. 회피 성분, 보류 사유, 생성 출처,
이력, 토큰 갱신, 삭제 후 404, 계정 정리도 통과했다. 이력 목록에는 히트맵을 싣지 않는다.
추천 재시도는 원래 저장한 분류를 사용하고 히트맵을 Qwen에 전달하지 않는다.

검증 산출물은 실행 PC의 `AI/outputs/mentor_service_20261113`에 있다.
`comparison.json`, `review-checks.json`, `live-summary.json`과 개별 응답은 로컬에만 두고
사진·가중치·토큰·개별 의료 응답 로그는 PR에 포함하지 않는다. 테스트 서버와 임시 키는 정리했다.

## 재현과 검사

기본 API 개발 설치와 실제 모델 설치를 구분한다. 실제 모델·학습에는
`requirements-vision.txt`와 별도로 준비한 승인된 가중치·데이터가 필요하다.
RTX 50 시리즈에서는 CUDA 12.8 이상용 PyTorch를 먼저 설치한다.

```powershell
python -m pytest -q
python scripts/evaluate_mentor_service.py --assets ../AI --photos C:/Users/User/Downloads/test_sample_96/test_sample_96 --output outputs/mentor-comparison-new

# 재학습이 필요할 때만 새 경로로 실행. 기존 실험은 덮어쓰지 않는다.
python -m src.vision.train --root 'D:/14.안면부 피부질환 이미지 합성데이터' --out outputs/mentor-new --manifest-out data/processed/mentor-new --architectures dinov2_linear efficientnet_b0 densenet121
```

원본 ZIP에서 이미지를 메모리로 읽고 추정 출처 그룹을 분리한다. 파일명 그룹은 검증된
환자 ID가 아니다. 이번에 공유한 데이터 준비 코드는 내부 dev·holdout을 제외한 train
분할의 일반 질환 설명만 추천 지식으로 구성한다. 개인 사진의 소견이나 성분 효능 근거로
그 설명을 확대하지 않는다. 실제 서비스 시험은 기존 train-only 지식 파일을 사용했다.

최종 서비스 작업 트리: **pytest 515 passed, 12 skipped**. 선택 의존성·로컬 산출물이 필요한
검사는 기본 개발 환경에서 건너뛴다. Grad-CAM 실패·상수 맵·대상 불일치·색상/초과 크기·
구버전 응답과 DINO 동결·학습·가중치 불변을 검사했다.
Spring은 형식·빌드, 단위 **154개**, 실제 PostgreSQL/HTTP 통합 **90개**를 통과했다.
Swift 릴리스 통신·DTO 테스트 **27개**와 수정 화면 소스 구문 검사를 통과했다.
macOS Xcode의 iOS 타입 검사·시뮬레이터 화면·실기기 확인은 이 Windows PC에서 수행하지 못했다.

적용 순서는 AI → Spring(**DB V4**, 기존 V3 포함) → Swift다. V4는 nullable 컬럼 추가로
기존 행을 보존한다. 검증한 서비스 작업 트리는 `AI-integration-pr`이며, 원래 `AI` 폴더의
미커밋 연구 변경은 이 작업으로 덮어쓰지 않았다.
