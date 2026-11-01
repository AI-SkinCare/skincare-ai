"""One response path for live HTTP inference and offline output replay.

No new ingredient or efficacy claim is manufactured by this layer. A null purpose
is an explicit missing explanation, not an empty string filled with canned advice.
"""
from __future__ import annotations

import re

from src.llm.inference import RecResult
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter
from src.llm.schema import RecInput

from .postprocess import contradictory_ingredients, structure_ingredients, structure_routine
from .recommendation_quality import sanitize_recommendation
from .schemas import DiseasePrediction, RecommendResponse

QUALITY_VERSION = "recommendation_quality_v1"
SENSITIVE_CONCERNS = {"민감성(트러블/자극감)", "붉어짐(홍조)"}


def reports_sensitivity(note: str | None) -> bool:
    """Conservative handling of explicit survey symptoms, not disease diagnosis.

    A note cannot disable sensitivity derived from structured inputs. Negation of
    one symptom does not negate a separately reported stinging symptom.
    """
    if not note:
        return False
    for clause in re.split(r"[.,;!?\n]|하지만|그런데|그렇지만", note):
        clause = re.sub(r"민감(?:성)?(?:은|이|가)?\s*(?:없|아니|하지\s*않)[^ ,;.!?]*", "", clause)
        if re.search(r"민감(?:성|해서|해요|한|합니다)|따갑|따가|따끔|자극감|화끈", clause):
            return True
    return False


def prepare_recommendation(
    result: RecResult, x: RecInput, notes: list[str], extractor: IngredientExtractor,
    output_filter: OutputFilter, disease: DiseasePrediction | None = None,
) -> tuple[RecommendResponse, list[dict]]:
    # Check raw content BEFORE generic softening/removal can hide the claim.
    clean, findings = sanitize_recommendation(
        result.parsed, sensitive=bool(SENSITIVE_CONCERNS.intersection(x.concerns)) or reports_sensitivity(x.user_note),
        disease_context=disease is not None,
    )
    clean, report = output_filter.apply(clean)
    ingredients = structure_ingredients(clean, extractor, include_routine=True)
    routine = structure_routine(clean)
    issues = list(dict.fromkeys(f["code"] for f in findings))
    malformed_steps = (len(result.parsed["reasoning"]) != 3
                       or any(not s["content"].strip() for s in result.parsed["reasoning"]))
    if malformed_steps:
        issues.append("invalid_reasoning_structure")
    if any(not item["purpose"] for item in ingredients):
        issues.append("ingredient_purpose_unavailable")
    if result.used_fallback:
        issues.append("rule_fallback")
    # Do not silently truncate a longer list while still recommending omitted
    # ingredients in the answer/routine. Hold such a result for review instead.
    full_ingredients = structure_ingredients(clean, extractor, limit=100, include_routine=True)
    inconsistent = ({i["name"] for i in full_ingredients} != {i["name"] for i in ingredients}
                    or bool(contradictory_ingredients(clean, extractor)))
    if inconsistent:
        issues.append("ingredient_reference_inconsistent")
    if not ingredients:
        issues.append("no_supported_ingredients")
    if not routine:
        issues.append("no_supported_routine")
    if not clean["answer"].strip():
        issues.append("no_supported_answer")
    if any(not s["content"].strip() for s in clean["reasoning"]):
        issues.append("reasoning_part_removed")
    identity_conflict = any(f["code"] == "ingredient_identity_conflict" for f in findings)
    # Removing only the claim would leave the same ingredient in the routine
    # even though the model chose it for a disallowed care action.
    restricted_selection = any(f["code"] == "sensitive_exfoliation" and f["field"] == "reasoning[1].content"
                               for f in findings)
    if restricted_selection:
        issues.append("restricted_ingredient_purpose")
    withheld = (malformed_steps or inconsistent or identity_conflict or restricted_selection
                or not ingredients or not routine or not clean["answer"].strip())
    return RecommendResponse(
        skin_type=x.skin_type, concerns=x.concerns,
        ingredients=[] if withheld else ingredients,
        routine=[] if withheld else routine,
        reasoning=[] if withheld else [s["content"] for s in clean["reasoning"] if s["content"].strip()],
        answer=("현재 입력에 대해 제공할 수 있는 성분 추천이 충분하지 않아 보류했습니다."
                if withheld else clean["answer"]),
        disclaimer=clean["disclaimer"], filtered=report.changed or bool(findings),
        used_fallback=result.used_fallback, input_notes=list(notes),
        model_version=result.model_version, inference_ms=result.latency_ms,
        quality_status="withheld" if withheld else ("limited" if issues else "ok"),
        quality_issues=list(dict.fromkeys(issues)), postprocess_version=QUALITY_VERSION,
    ), findings
