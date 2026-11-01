# 질환 분류와 추천 API 계약 예시

실제 환자 자료가 없는 환경에서 백엔드·앱의 요청·응답 구조를 확인하는 **시험용 fixture**다.
입력 라벨·점수·설문은 만들어 넣었다. `generated.response.json`은 이 입력으로 실제 Qwen2.5 v2가
2026-10-15 로컬 HTTP 시험에서 생성한 기록이다. 환자 사진 판정이나 전문가가 확인한 효능 정답이 아니다.
내용상 한계와 purpose 누락을 그대로 포함한다. [품질 점검 보고서](../../disease-pipeline-quality-20261015.md)를 함께 확인한다.

2026-10-21 추가 상태는 [추천 품질 규약](../../recommendation-quality.md)을 확인한다.
`quality_rejected.response.example.json`은 새 보류 화면 구현용 **합성 계약 예시**다.
실제 모델 응답 기록이 아니며, 특정 request가 항상 이 상태를 반환한다는 뜻도 아니다.
기존 2026-10-15 응답 파일은 당시 기록으로 보존한다.

| 파일 쌍 | 응답 상태 | GPU 생성 |
|---|---|---|
| `generated.request.json` / `generated.response.json` | generated | 실제 Qwen 생성 기록 |
| `uncertain.request.json` / `uncertain.response.json` | uncertain | 호출하지 않음 |
| `no_concern.request.json` / `no_concern.response.json` | no_concern | 호출하지 않음 |

`POST /infer/disease-recommend`에 request 파일을 보내면 같은 **스키마**의 응답을 받는다.
모델·환경·생성 과정에 따라 문장이 달라질 수 있다. mock 모드의 답변도 이 실제 생성 기록과 다르다.

```powershell
curl.exe http://127.0.0.1:8000/infer/disease-recommend `
  -H "X-Internal-Token: $env:AI_INTERNAL_TOKEN" `
  -H "Content-Type: application/json" `
  --data-binary "@docs/fixtures/disease-pipeline/generated.request.json"
```

사진 입력을 시험하려면 실제 DenseNet이 활성화된 서버의 `/infer/disease` 또는
`/infer/disease-analyze`를 사용한다. 이 JSON fixture 시험만으로 사진 분류를 검증했다고 보고하지 않는다.
