"""추천 모델 평가.

사용 (PowerShell):
  python -m src.llm.evaluate --mock                                   # 규칙 기반 베이스라인
  python -m src.llm.evaluate --adapter outputs/rec_sllm_v1/adapter    # 학습 모델
  python -m src.llm.evaluate --base Qwen/Qwen2.5-7B-Instruct          # 파인튜닝 전 원본 모델 (비교용)

지표
  - json_ok: 형식 준수율 (재시도 전 1차 생성 기준, 폴백 제외)
  - precision@3: 모델 추천 상위 3개 성분 중 정답 성분 집합에 있는 비율
  - recall@all: 정답 성분 중 모델이 언급한 비율
  - answer_similarity_f1: 정답 추론문(3단계+답변)과의 글자 2-gram F1 (어휘 유사도, 거친 대리지표)
  - filter_rate: 금칙 표현 필터가 개입한 비율 (낮을수록 학습이 잘 된 것)
  - latency p50/p95, 출력 토큰 수
세 가지(원본 모델 / 규칙 베이스라인 / 파인튜닝 모델)를 같이 제시해야 '파인튜닝이 효과가 있었다'를 증명할 수 있다.

주의 1 — 폴백 분리
  precision@3·recall·similarity는 '모델이 직접 생성한 답'만으로 계산한다.
  폴백(규칙 기반) 답까지 섞으면, 형식을 매번 어기는 망가진 모델도 폴백 덕분에
  베이스라인과 비슷한 점수를 받아 실패가 가려진다. 폴백 포함 수치는 *_with_fallback으로 따로 둔다.

주의 2 — precision@3만으로는 규칙 베이스라인과 구분이 안 된다
  이 데이터는 고민별 추천 성분이 거의 고정돼 있어(예: 미백 → 알로에신·헥사펩타이드-2·댕댕이나무열매즙)
  '고민별 빈출 성분' 표만으로 precision@3 ≈ 0.69가 나온다. 파인튜닝의 효과는
  (1) 원본 모델 대비 precision@3 상승, (2) 규칙 템플릿 대비 answer_similarity 상승으로 보인다.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

from .ingredients import INGREDIENT_SCHEMA_VERSION, IngredientExtractor
from .prepare_quality_data import digest, target_text
from .inference import MockRecommender, SLLMRecommender
from .output_filter import OutputFilter
from .schema import RecInput

ROOT = Path(__file__).resolve().parents[2]
SCORING_VERSION = "fixed_top3_v3"


def comparison_key(proc: Path, ext: IngredientExtractor, split: str, with_codes: bool) -> str:
    """표본 수가 달라도 같은 평가 모집단·사전·입력 정책에서만 짝지어 비교."""
    return digest({"corpus": hashlib.sha256((proc / f"{split}.jsonl").read_bytes()).hexdigest(),
                   "lexicon": ext.lexicon, "ingredient_schema": INGREDIENT_SCHEMA_VERSION,
                   "scoring": SCORING_VERSION, "split": split, "with_codes": with_codes,
                   "prompt": hashlib.sha256(Path(__file__).with_name("schema.py").read_bytes()).hexdigest(),
                   "filter": hashlib.sha256((ROOT / "configs/filter_rules.yaml").read_bytes()).hexdigest()
                   if (ROOT / "configs/filter_rules.yaml").exists() else None})


def _bigrams(s: str) -> collections.Counter:
    s = re.sub(r"\s+", "", s)
    return collections.Counter(s[i : i + 2] for i in range(len(s) - 1))


def bigram_f1(pred: str, gold: str) -> float:
    """공백을 뺀 글자 2-gram F1. 형태소 분석기 없이 한국어 유사도를 거칠게 재는 용도."""
    p, g = _bigrams(pred), _bigrams(gold)
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def full_text(parsed: dict) -> str:
    return " ".join(s["content"] for s in parsed["reasoning"]) + " " + parsed["answer"]


def serving_policy(x: RecInput, use_codes: bool) -> RecInput:
    """서빙과 같은 조건으로 평가하기 위해 입력을 맞춘다."""
    if not use_codes:
        x.condition_codes = None
    x.user_note = None
    return x


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/sllm_qlora.yaml"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--adapter")
    g.add_argument("--base")
    ap.add_argument("--n", type=int, default=None,
                    help="평가 샘플 수 (기본: val_eval 전체 201건). 고정 시드 무작위 표본이므로 비교할 모델끼리 같은 값을 쓰세요")
    ap.add_argument("--with-codes", action="store_true", help="상태 코드를 입력에 포함 (기본은 서빙 정책대로 제외)")
    ap.add_argument("--split", choices=("val_eval", "test", "val"), default="val_eval")
    ap.add_argument("--output-dir", type=Path, help="평가 결과 저장 경로 (v3은 기존 결과와 별도 저장)")
    ap.add_argument("--max-new-tokens", type=int, default=2400,
                    help="정답 길이 p95≈2,170자 → 토큰으로 1,450~2,000. 1800이면 긴 답이 JSON 중간에서 잘린다")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    proc = ROOT / cfg["data"]["processed_dir"]
    rows = [json.loads(l) for l in open(proc / f"{args.split}.jsonl", encoding="utf-8") if l.strip()]
    if args.n is not None and args.n <= 0:
        ap.error("--n must be positive")
    if not rows:
        ap.error("평가 split이 비어 있습니다")
    if args.n and args.n < len(rows):
        # val_eval.jsonl은 희소 고민이 앞에 몰려 있고 고민별로 묶여 있어 앞에서 자르면 분포가 왜곡된다.
        # 고정 시드 표본 → 어떤 모델을 평가해도 같은 --n이면 같은 문항 (공정 비교)
        rows = random.Random(0).sample(rows, args.n)
    ext = IngredientExtractor.load(proc / "ingredient_dict.json")
    filt = OutputFilter.from_yaml()
    manifest_path = proc / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest:
        # --n 적용 전 전체 모집단 파일을 검증한다.
        all_rows = [json.loads(l) for l in (proc / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if digest(all_rows) != manifest["split_hashes"].get(args.split) or digest(ext.lexicon) != manifest["lexicon_hash"]:
            raise ValueError("dataset changed after generation; use a new version")
    evaluation_key = comparison_key(proc, ext, args.split, args.with_codes)

    if args.mock:
        model, tag = MockRecommender(proc / "concern_top_ingredients.json"), "rule_baseline"
    else:
        adapter = args.adapter
        base = args.base
        if adapter:
            info = Path(adapter).parent / "train_info.json"
            base = json.loads(info.read_text(encoding="utf-8"))["base_model"] if info.exists() else cfg["model"]["base"]
        print(f"[load] base={base} adapter={adapter}")
        model = SLLMRecommender(base, adapter, cfg["model"]["load_in_4bit"], args.max_new_tokens,
                                top_path=proc / "concern_top_ingredients.json").load()
        tag = Path(adapter).parent.name if adapter else "base_" + base.split("/")[-1]

    per = collections.defaultdict(lambda: collections.Counter())
    lat, toks, samples, sims, items = [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        x = serving_policy(RecInput.from_dict(r["input"]), args.with_codes)
        res = model.recommend(x)
        cleaned, rep = filt.apply(res.parsed)
        step2 = next((s["content"] for s in cleaned["reasoning"] if "성분" in s["title"]),
                     cleaned["reasoning"][min(1, len(cleaned["reasoning"]) - 1)]["content"])
        pred = ext.extract(step2 + "\n" + cleaned["answer"])
        gold = set(ext.extract(target_text(json.loads(r["target"]))))
        if manifest and gold != set(r["gold_ingredients"]):
            raise ValueError(f"stale gold labels: {r['id']}; rebuild dataset")
        c = per[r["target_concern"]]
        item = {"id": r["id"], "concern": r["target_concern"], "gold": sorted(gold), "pred": pred,
                "fallback": res.used_fallback, "attempts": res.attempts, "filtered": rep.changed,
                "latency_ms": res.latency_ms, "raw_text": res.raw_text, "parsed": res.parsed,
                "cleaned": cleaned, "gold_target": json.loads(r["target"]),
                "multi_concern": len(x.concerns) > 1}
        c["n"] += 1
        c["json_ok_first"] += int(res.attempts == 1 and not res.used_fallback)
        c["fallback"] += int(res.used_fallback)
        c["empty_gold"] += int(not gold)
        # 폴백 포함 수치 (사용자가 실제로 받는 결과)
        if gold:
            top3 = pred[:3]
            hit3, den3 = len(set(top3) & gold), 3  # 1개만 추천해도 분모는 항상 3
            hit_all, den_all = len(set(pred) & gold), len(gold)
            c["sys_p3_hit"] += hit3
            c["sys_p3_den"] += den3
        # 모델 단독 수치: 폴백 답은 제외 (주의 1 참고)
        if not res.used_fallback:
            c["gen"] += 1
            c["filtered"] += int(rep.changed)
            sims.append(bigram_f1(full_text(cleaned), full_text(json.loads(r["target"]))))
            item["similarity"] = round(sims[-1], 4)
            if gold:
                c["scored"] += 1
                item["p3_hit"], item["p3_den"] = hit3, den3
                c["p3_hit"] += hit3
                c["p3_den"] += den3
                c["rec_hit"] += hit_all
                c["rec_den"] += den_all
        items.append(item)
        lat.append(res.latency_ms)
        toks.append(res.output_tokens)
        if len(samples) < 5:
            samples.append({"id": r["id"], "concern": r["target_concern"], "gold": sorted(gold),
                            "pred": pred, "answer": cleaned["answer"], "dropped": rep.dropped_sentences})
        print(f"\r{i + 1}/{len(rows)}  마지막 {res.latency_ms}ms", end="", flush=True)
    print()

    tot = sum(per.values(), collections.Counter())
    def rate(a, b):
        return round(a / b, 4) if b else None
    summary = {
        "model": tag, "n": tot["n"], "with_condition_codes": args.with_codes,
        "json_ok_first_try": rate(tot["json_ok_first"], tot["n"]),
        "fallback_rate": rate(tot["fallback"], tot["n"]),
        "generated_n": tot["gen"], "scored_n": tot["scored"], "empty_gold_n": tot["empty_gold"],
        "scoring_version": SCORING_VERSION, "ingredient_schema": INGREDIENT_SCHEMA_VERSION,
        "comparison_key": evaluation_key, "dataset_id": manifest.get("dataset_id", "legacy_relabelled"),
        "split": args.split, "lexicon_hash": digest(ext.lexicon),
        "test_status": manifest.get("test_status"),
        "evaluated_ids_hash": digest(sorted(r["id"] for r in rows)),
        "precision_at_3": rate(tot["p3_hit"], tot["p3_den"]),
        "macro_concern_precision_at_3": round(statistics.mean(v["p3_hit"] / v["p3_den"] for v in per.values() if v["p3_den"]), 4)
        if any(v["p3_den"] for v in per.values()) else None,
        "recall_all": rate(tot["rec_hit"], tot["rec_den"]),
        "answer_similarity_f1": round(statistics.mean(sims), 4) if sims else None,
        "filter_intervention_rate": rate(tot["filtered"], tot["gen"]),
        "precision_at_3_with_fallback": rate(tot["sys_p3_hit"], tot["sys_p3_den"]),
        "latency_ms_p50": int(statistics.median(lat)) if lat else None,
        "latency_ms_p95": int(sorted(lat)[max(0, math.ceil(len(lat) * .95) - 1)]) if lat else None,
        "output_tokens_mean": int(statistics.mean(toks)) if toks and any(toks) else None,
        "wall_minutes": round((time.time() - t0) / 60, 1),
        "per_concern": {
            k: {"n": v["n"], "scored_n": v["scored"], "precision_at_3": rate(v["p3_hit"], v["p3_den"]),
                "reliable": v["scored"] >= 30}
            for k, v in sorted(per.items(), key=lambda kv: -kv[1]["n"])
        },
    }
    if tot["n"] and tot["fallback"] / tot["n"] > 0.1:
        print(f"[경고] 폴백 {tot['fallback']}/{tot['n']}건 — 형식 준수 실패가 많습니다. "
              f"모델 단독 지표는 {tot['gen']}건 기준이며, 폴백 건은 점수에서 제외했습니다.")
    out = args.output_dir or (ROOT / "outputs" / "eval" / proc.name / args.split if manifest else ROOT / "outputs" / "eval")
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # items: 문항별 결과 → compare.py가 모델끼리 '같은 문항'만 골라 짝지어 비교한다
    (out / f"{tag}_{stamp}.json").write_text(json.dumps({"summary": summary, "samples": samples, "items": items},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[saved] {out / f'{tag}_{stamp}.json'}")


if __name__ == "__main__":
    main()
