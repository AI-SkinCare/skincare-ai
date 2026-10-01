# AI 서버 고정 응답 (fixture)

백엔드 `AI_MODE=fixture`([API 명세서 §7](../project/03-api-spec.md#7-개발용-고정-응답-fixture))에서 쓰는 응답입니다.
백엔드 `test/fixtures/ai/`로 복사해서 사용하세요.

| 파일 | API | 만든 방법 |
|---|---|---|
| `recommend.ok.json` | `POST /infer/recommend` (§5.2) | **v2 모델(`rec_sllm_v2`) 실제 출력**. 2026-10-02, RTX 5070 Ti, 생성 58.4초 |
| `skin-metrics.ok.json` | `POST /infer/skin-metrics` (§5.1) | stub 회귀기(`is_stub: true`) 실제 응답 |

## recommend.ok.json을 만든 요청

명세서 §5.2 예시 요청 그대로입니다.

```json
{
  "metrics": { "full_face": { "moisture": 0.61, "elasticity": 0.74, "pigmentation": 0.38, "pore": 0.45 } },
  "survey": {
    "gender": "female", "ageBand": "20s", "skinType": "복합성", "sleepHours": 6,
    "waterIntakeLevel": "low", "sunExposureLevel": "high", "stressLevel": "medium",
    "currentProducts": ["토너", "수분크림"],
    "selfReportedConcerns": ["모공", "미백(색소침착/기미/칙칙함)"],
    "note": "여름에 번들거림이 심해요"
  }
}
```

## 알아 둘 점 (실제 모델 동작)

- **고민을 2개 이상 보내도 답변은 사실상 첫 번째 고민 위주입니다.** 위 요청은 모공·미백을 보냈지만 성분과 설명은 모공 중심입니다.
  학습 데이터가 문항당 고민 하나 기준이기 때문입니다. `concerns` 필드에는 두 고민이 모두 들어갑니다.
- `metrics: null`(데모 측정값)로 보내도 위 요청에서는 **같은 결과**가 나옵니다. 설문에 피부 타입·고민이 있으면 측정값은 추천 입력에 쓰이지 않고,
  설문에 없을 때만 측정값으로 피부 타입·고민을 추정합니다(`input_notes`에 기록).
- 생성 시간은 GPU를 이 서버 혼자 쓸 때 약 48~60초입니다. 다른 프로그램(Ollama, 다른 모델 서버 등)이 GPU를 쓰면 절반 속도로 떨어져 **120초 타임아웃(504 E3002)이 납니다.**
  `rec-slow` 시나리오(60초 지연)는 실제 정상 속도와 비슷합니다.
