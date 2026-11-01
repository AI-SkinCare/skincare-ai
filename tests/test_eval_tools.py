import json

import pytest
import yaml

from src.llm import compare, evaluate
from src.llm.inference import RecResult
from src.llm.schema import RecInput, build_target

CONCERN = "미백(색소침착/기미/칙칙함)"
TARGET = build_target(
    [{"title": "문제 분석 및 원인 파악", "content": "색소침착이 보입니다."},
     {"title": "성분 선택 및 근거 제시", "content": "알로에신을 권합니다."},
     {"title": "사용법 및 관리방안", "content": "저녁에 바르세요."}],
    "알로에신이 포함된 제품을 사용하세요.",
)


# ── bigram_f1 ──
def test_bigram_f1_bounds_and_whitespace():
    assert evaluate.bigram_f1("알로에신 추천", "알로에신 추천") == 1.0
    assert evaluate.bigram_f1("알로에신추천", "알로에신 추천") == 1.0   # 띄어쓰기 차이는 무시
    assert evaluate.bigram_f1("가나다", "라마바") == 0.0
    assert 0 < evaluate.bigram_f1("알로에신을 권합니다", "알로에신은 좋습니다") < 1


# ── evaluate: 폴백 답은 모델 점수에서 제외 ──
def _run_eval(tmp_path, monkeypatch, used_fallback: bool) -> dict:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "ingredient_dict.json").write_text(
        json.dumps({"lexicon": {"알로에신": "알로에신"}, "freq": {}}), encoding="utf-8")
    x = RecInput(gender="여성", age=30, skin_type="중성", concerns=[CONCERN])
    row = {"id": "1", "target_concern": CONCERN, "input": x.to_dict(), "target": TARGET,
           "gold_ingredients": ["알로에신"]}
    (proc / "val_eval.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"processed_dir": str(proc)},
                                   "model": {"base": "x", "load_in_4bit": False}}), encoding="utf-8")

    class Fake:
        version = "fake"

        def __init__(self, *a, **k):
            pass

        def recommend(self, _x):
            # 폴백이 '우연히 정답'을 내도 모델 점수에 들어가면 안 된다
            return RecResult(json.loads(TARGET), "fake", used_fallback, 2 if used_fallback else 1, "", 0, 1)

    monkeypatch.setattr(evaluate, "MockRecommender", Fake)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    evaluate.main(["--config", str(cfg), "--mock"])
    out = next((tmp_path / "outputs" / "eval").glob("*.json"))
    return json.loads(out.read_text(encoding="utf-8"))["summary"]


def test_fallback_answers_are_not_scored_as_model(tmp_path, monkeypatch):
    s = _run_eval(tmp_path, monkeypatch, used_fallback=True)
    assert s["fallback_rate"] == 1.0 and s["scored_n"] == 0
    assert s["precision_at_3"] is None and s["answer_similarity_f1"] is None
    assert s["precision_at_3_with_fallback"] == 0.3333   # 빈 추천 슬롯도 분모 3에 포함


def test_model_answers_are_scored(tmp_path, monkeypatch):
    s = _run_eval(tmp_path, monkeypatch, used_fallback=False)
    assert s["scored_n"] == 1 and s["precision_at_3"] == 0.3333
    assert s["answer_similarity_f1"] == 1.0


# ── compare ──
def _write(d, name, summary):
    (d / name).write_text(json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8")


def test_compare_picks_latest_and_orders_rows(tmp_path):
    _write(tmp_path, "rec_sllm_v1_20261001-1200.json", {"model": "rec_sllm_v1", "n": 201, "precision_at_3": 0.8})
    _write(tmp_path, "rule_baseline_20260926-0700.json", {"model": "rule_baseline", "n": 201, "precision_at_3": 0.5})
    _write(tmp_path, "rule_baseline_20260926-0725.json", {"model": "rule_baseline", "n": 201, "precision_at_3": 0.6883})
    _write(tmp_path, "base_Qwen2.5-7B-Instruct_20260927-0900.json",
           {"model": "base_Qwen2.5-7B-Instruct", "n": 201, "precision_at_3": 0.1})
    s = compare.latest_summaries(tmp_path)
    assert list(s) == ["rule_baseline", "base_Qwen2.5-7B-Instruct", "rec_sllm_v1"]
    assert s["rule_baseline"]["precision_at_3"] == 0.6883   # 최신 파일
    table = compare.build_table(s)
    assert "68.8%" in table and "80.0%" in table


def test_compare_cli_saves_table_in_requested_directory(tmp_path, monkeypatch):
    _write(tmp_path, "rule_20261002-120000.json", {"model": "rule_baseline", "n": 1})
    monkeypatch.setattr("sys.argv", ["compare", "--eval-dir", str(tmp_path)])
    compare.main()
    assert "rule_baseline" in (tmp_path / "comparison.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("v,kind,expected", [
    (None, "pct", "-"), (0.6883, "pct", "68.8%"), (42000, "sec", "42.0s"), (12, "sec", "12ms"), (1534, "int", "1,534"),
])
def test_compare_formatting(v, kind, expected):
    assert compare._fmt(v, kind) == expected


def test_eval_saves_per_item_results(tmp_path, monkeypatch):
    _run_eval(tmp_path, monkeypatch, used_fallback=False)
    out = next((tmp_path / "outputs" / "eval").glob("*.json"))
    items = json.loads(out.read_text(encoding="utf-8"))["items"]
    assert len(items) == 1
    want = {"id": "1", "concern": CONCERN, "gold": ["알로에신"], "pred": ["알로에신"], "fallback": False,
            "attempts": 1, "filtered": False, "similarity": 1.0, "p3_hit": 1, "p3_den": 3}
    assert {k: items[0][k] for k in want} == want and "latency_ms" in items[0]
    assert {k: items[0]["cleaned"][k] for k in ("reasoning", "answer")} == json.loads(TARGET)
    assert "raw_text" in items[0] and items[0]["gold_target"] == json.loads(TARGET)


def test_compare_refuses_incompatible_scoring_or_data():
    a = {"comparison_key": "old", "precision_at_3": 0.8}
    b = {"comparison_key": "new", "precision_at_3": 0.9}
    with pytest.raises(ValueError, match="다른 결과"):
        compare.build_table({"rule": a, "model": b})
    with pytest.raises(ValueError, match="비교할 수"):
        compare.paired_vs_rule({"rule": {"summary": a, "items": [{"id": "1", "p3_den": 3}]},
                               "model": {"summary": b, "items": [{"id": "1", "p3_den": 3}]}})


def _items(d, name, model, hits):
    items = [{"id": str(i), "p3_hit": h, "p3_den": 3} for i, h in enumerate(hits)]
    (d / name).write_text(json.dumps({"summary": {"model": model}, "items": items}), encoding="utf-8")


def test_paired_comparison_uses_common_items_only(tmp_path):
    _items(tmp_path, "rule_baseline_20260927-1000.json", "rule_baseline", [1] * 200)
    _items(tmp_path, "rec_sllm_v1_20260927-1100.json", "rec_sllm_v1", [3] * 60)   # 60건만 평가
    t = compare.paired_vs_rule(compare.latest_results(tmp_path))
    assert "| rec_sllm_v1 | 60 | 33.3% | 100.0% | +66.7%p (±0.0) ✅ |" in t


def test_paired_comparison_flags_insignificant(tmp_path):
    _items(tmp_path, "rule_baseline_20260927-1000.json", "rule_baseline", [2, 1] * 10)
    _items(tmp_path, "rec_sllm_v1_20260927-1100.json", "rec_sllm_v1", [1, 2] * 10)
    assert "유의하지 않음" in compare.paired_vs_rule(compare.latest_results(tmp_path))


def test_paired_comparison_without_items_explains(tmp_path):
    _write(tmp_path, "rule_baseline_20260926-0725.json", {"model": "rule_baseline"})
    assert "--mock" in compare.paired_vs_rule(compare.latest_results(tmp_path))


def test_concern_table_marks_small_samples():
    s = {"rule_baseline": {"per_concern": {
        "모공": {"precision_at_3": 0.65, "reliable": True},
        "민감성(트러블/자극감)": {"precision_at_3": 0.5, "reliable": False}}}}
    t = compare.build_concern_table(s)
    assert "| 모공 | 65% |" in t and "| 민감성(트러블/자극감) | (50%) |" in t
