"""Carry existing class care text through generation and final validation.

These are bounded cosmetic support instructions from the existing evidence
store. The checks require a verbatim source sentence and reject a few explicit
contradictions. They do not assess diagnosis, patient suitability or efficacy.
"""
from __future__ import annotations

import re

VERSION = "disease_care_v1"

_DAMP = "보습제는 세정 뒤 수분이 남아 있을 때 사용한다."
_DRY = "보습제는 세정 후 피부가 마른 다음 사용한다."
_ALCOHOL_FREE = "순하게 세정한 뒤 무향·알코올 없는 완제품으로 보습한다."
_ALL_ALCOHOL_FREE = "사용하는 모든 완제품은 알코올 없는 제품을 선택한다."
_RULES = {
    "atopic": ("moisturizer_damp_after_wash", _DAMP, "AAD_AD"),
    "psoriasis": ("moisturizer_damp_after_wash", _DAMP, "AAD_PS"),
    "rosacea": ("moisturizer_dry_after_wash", _DRY, "AAD_ROS"),
    "seborrheic": ("moisturizer_alcohol_free", _ALCOHOL_FREE, "AAD_SD"),
}


def class_care(label):
    """Return the existing class-specific sentence and its actual source ID."""
    rule = _RULES.get(label)
    if rule is None:
        return []
    rule_id, text, source_id = rule
    rules = [dict(id=rule_id, product="moisturizer", text=text, source_id=source_id)]
    if label == "seborrheic":
        rules.append(dict(id="all_products_alcohol_free", product="all", text=_ALL_ALCOHOL_FREE,
                          source_id="AAD_SD"))
    return rules


def applicable_care(bundle, cards):
    """Never add a moisturizer instruction to a sunscreen-only selection.

    Historical bundles have only care_conditions. Require the same source text
    and source entry in those bundles too, rather than inferring new advice
    solely from a label. All live bundles are assembled on the server.
    """
    from .product_conditions import kind

    if not cards:
        return []
    products = {kind(card) for card in cards}
    conditions = bundle.get("care_conditions", [])
    return [rule for rule in class_care(bundle.get("predicted_label"))
            if (rule["product"] == "all" or rule["product"] in products)
            and any(rule["text"] in text for text in conditions)
            and rule["source_id"] in bundle.get("sources", {})]


def care_text(bundle, cards):
    return " ".join(rule["text"] for rule in applicable_care(bundle, cards))


def _compact(text):
    return re.sub(r"\s+", "", text)


def _moisturizer_clauses(text):
    active = False
    for clause in re.split(r"(?<=[.!?。])\s+|[\n;]+|하지만|그러나|반면", text):
        if "보습" in clause:
            active = True
        elif re.search(r"선크림|자외선\s*차단제", clause):
            active = False
        if active:
            yield clause


def review_care(parsed, response, bundle):
    names = {ingredient.name for ingredient in response.ingredients}
    cards = [card for card in bundle["claims"] if card["ingredient"] in names]
    rules = applicable_care(bundle, cards)
    steps = parsed.get("reasoning", [])
    routine = steps[2].get("content", "") if len(steps) > 2 else ""
    delivered_steps = getattr(response, "routine", None)
    delivered_routine = " ".join(step.action for step in delivered_steps) if delivered_steps is not None else routine
    delivered_reasoning = getattr(response, "reasoning", None)
    delivered_usage = delivered_reasoning[2] if delivered_reasoning and len(delivered_reasoning) > 2 else routine
    text = "\n".join(step.get("content", "") for step in steps) + "\n" + parsed.get("answer", "")
    issues, covered, bindings = [], [], []
    for rule in rules:
        rule_id = rule["id"]
        present = _compact(rule["text"]) in _compact(routine)
        if not present:
            issues.append("care_condition_missing_" + rule_id)
        else:
            covered.append(rule_id)
        if present and (_compact(rule["text"]) not in _compact(delivered_routine)
                        or _compact(rule["text"]) not in _compact(delivered_usage)):
            issues.append("care_condition_not_delivered_" + rule_id)
        # Deliberately narrow contradiction detection, not semantic validation.
        if rule_id == "moisturizer_damp_after_wash":
            opposite = r"(?:피부가?\s*마른\s*(?:다음|후|뒤)|(?:물기|피부)[^.!?\n]{0,12}(?:말린|말리고|건조시킨))"
        elif rule_id == "moisturizer_dry_after_wash":
            opposite = r"(?:수분|물기)[^.!?\n]{0,12}(?:남아|남은|있는\s*(?:상태|동안))|젖은\s*피부|마르기\s*전"
        else:
            opposite = r"알코올(?:이|을)?\s*(?:함유|포함|들어|있는)|알코올\s*없는[^.!?\n]{0,20}(?:피하|필요\s*없|쓰지)"
        # Check the whole displayed payload as well as the routine so a correct
        # copied sentence cannot mask contradictory advice in the answer.
        clauses = [text] if rule["product"] == "all" else _moisturizer_clauses(text)
        if any(re.search(opposite, clause) for clause in clauses):
            issues.append("care_condition_contradicted_" + rule_id)
        reversed_text = re.escape(rule["text"].rstrip(".")) + r"[^.!?\n]{0,12}(?:않|아니|필요\s*없)"
        if re.search(reversed_text, text):
            issues.append("care_condition_contradicted_" + rule_id)
        bindings.append({**rule, "source": bundle["sources"][rule["source_id"]]})
    return dict(version=VERSION, issues=list(dict.fromkeys(issues)),
                required=[rule["id"] for rule in rules], covered=covered, bindings=bindings,
                automatic_check_only=True, clinical_entailment_verified=False)
