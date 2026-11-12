"""Bind existing fragrance-free advice to the product it actually describes.

Only the reviewed cosmetic support cards are used. This is a narrow discourse
check, not a clinical safety validator. Generated text is never patched to pass.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy

from .disease_care import care_text, review_care

VERSION='product_conditions_v4'
LABELS={'moisturizer':'보습제','sunscreen':'선크림'}
GUIDANCE="""Minimally correct the draft below, returning the same complete three-step JSON format. Keep its neutral step 1, all valid ingredient effect sentences and their [E_...] ids, and its selected ingredients. Correct step 3 so each selected product has its own fragrance-free instruction. Use separate Korean lines headed 보습제: and 선크림:. A fragrance-free sunscreen clause cannot cover moisturizer. Copy the applicable reviewed product conditions; do not invent efficacy, concentration, timing or frequency. If a requested product kind is missing, add an applicable card of that kind (replace a redundant same-kind selection if needed to keep at most two), copy its full effect and id into step 2, and include its usage line. Keep answer consistent with step 2; never mention unselected ingredients. Remove internal placeholder tags by generating clean Korean text. Do not output JSON field names as advice. Without supplied evidence cards, do not invent ingredient facts or ids."""

_MOIST=re.compile(r'보습(?:제|\s*크림)|moisturi[sz](?:er|ing)',re.I)
_SUN=re.compile(r'선크림|자외선\s*차단제|sunscreen|SPF|광범위\s*자외선\s*차단',re.I)
_FRAGRANCE=re.compile(r'무향|향료(?:가)?\s*없는|향료\s*무첨가',re.I)
_REVERSED=re.compile(r'무향[^,;.!?\n]{0,15}(?:아니|필요\s*없|피하|쓰지)|향료(?:가)?\s*있어도',re.I)
_COMMON=re.compile(r'공통|(?:모든|각|두(?:\s*가지)?)\s*제품|둘\s*다',re.I)
_EXCLUDED=re.compile(r'(?:사용|추천)(?:하지|할\s*필요가)\s*(?:않|마|없)|필요\s*없|쓰지\s*않',re.I)
_INTERNAL=re.compile(r'\[(?:care_conditions|product_usage|usage_line|conditions)\]',re.I)
_LABEL_USE=re.compile(r'표시(?:대로|를\s*따|에\s*따)',re.I)
_STOP=re.compile(r'(?:자극|따가움)[^.!?\n]{0,35}중단',re.I)


def display_parsed(parsed):
    """Remove only nonsemantic field markers; never add care or efficacy text."""
    out=deepcopy(parsed)
    for s in out.get('reasoning',[]):s['content']=_INTERNAL.sub('',s['content']).strip()
    out['answer']=_INTERNAL.sub('',out.get('answer','')).strip()
    return out


def kind(claim):
    return 'sunscreen' if claim['id'].endswith('_SUN') else 'moisturizer'


def requested_products(x):
    text=x.user_note or ''
    if '사용자 추가 요청:' in text:
        text=text.split('사용자 추가 요청:',1)[1].split('\n질환 일반 설명 참고',1)[0]
    wanted=set()
    for clause in re.split(r'[.!?\n;]|하지만|그러나',text):
        if _EXCLUDED.search(clause):continue
        if re.search(r'보습',clause):wanted.add('moisturizer')
        if re.search(r'선크림|자외선\s*차단|자차',clause,re.I):wanted.add('sunscreen')
    return wanted


def prompt_usage(bundle,selected_names=None,required_products=()):
    selected=set(selected_names or [])
    existing={kind(c) for c in bundle['claims'] if c['ingredient'] in selected}
    missing=set(required_products)-existing
    cards=[c for c in bundle['claims'] if not selected or c['ingredient'] in selected or kind(c) in missing]
    lines=[c['ingredient']+' ['+c['id']+']의 검토된 사용 문장 — '+LABELS[kind(c)]+': '+' '.join(c['conditions']) for c in cards]
    if care_text(bundle,cards):lines.append(care_text(bundle,cards))
    return '\n선택 제품별로 적용할 기존 근거의 사용 문장:\n'+'\n'.join(lines)


def routine_lines(bundle,selected_names,required_products=()):
    """A source-derived correction instruction, never an output replacement."""
    selected=set(selected_names)
    cards=[c for c in bundle['claims'] if c['ingredient'] in selected]
    for product in sorted(set(required_products)-{kind(c) for c in cards}):
        extra=next((c for c in bundle['claims'] if kind(c)==product),None)
        if extra:cards.append(extra)
    lines=[]
    for product in ('moisturizer','sunscreen'):
        conditions=list(dict.fromkeys(t for c in cards if kind(c)==product for t in c['conditions']))
        if product=='moisturizer' and care_text(bundle,cards):conditions.append(care_text(bundle,cards))
        if conditions:lines.append(LABELS[product]+': '+' '.join(conditions))
    if not any(kind(c)=='moisturizer' for c in cards) and care_text(bundle,cards):
        lines.append(care_text(bundle,cards))
    return '\n'.join(lines)


def review_usage(parsed,response,bundle,*,required_products=()):
    by_name={c['ingredient']:c for c in bundle['claims']}
    selected={kind(by_name[i.name]) for i in response.ingredients if i.name in by_name}
    available={kind(c) for c in bundle['claims']}
    required=set(required_products)&available
    issues=['product_recommendation_missing_'+p for p in sorted(required-selected)]
    steps=parsed.get('reasoning',[])
    routine=next((s['content'] for s in steps if '사용법' in s.get('title','') or '루틴' in s.get('title','')),steps[-1]['content'] if steps else '')
    # Also inspect products discussed in usage, even if no sunscreen ingredient
    # happened to be named in step 2. Negative-only mentions are not selections.
    routine_kinds=set()
    for sentence in re.split(r'[.!?\n;]',routine):
        if _EXCLUDED.search(sentence):continue
        if _MOIST.search(sentence):routine_kinds.add('moisturizer')
        if _SUN.search(sentence):routine_kinds.add('sunscreen')
    needed=selected|routine_kinds|required
    chosen=[c for c in bundle['claims'] if c['ingredient'] in by_name and c['ingredient'] in {i.name for i in response.ingredients}]
    # Require only conditions actually present in the eligible reviewed cards.
    label_needed={kind(c) for c in chosen if any(_LABEL_USE.search(t) for t in c['conditions'])}
    stop_needed={kind(c) for c in chosen if any(_STOP.search(t) for t in c['conditions'])}
    covered=set();label_covered=set();stop_covered=set();active=set();ambiguities=[]
    for part in re.split(r'(?<=[.!?。])\s+|[\n,;，；]+|하지만|그러나|반면',routine):
        part=part.strip()
        if not part:continue
        mentioned=set()
        if _MOIST.search(part):mentioned.add('moisturizer')
        if _SUN.search(part):mentioned.add('sunscreen')
        if len(mentioned)==1:scope=mentioned
        elif len(mentioned)>1:
            shared=bool(re.search(r'보습(?:제|\s*크림)\s*(?:와|과|및|/)\s*(?:선크림|자외선\s*차단제)|(?:선크림|자외선\s*차단제)\s*(?:와|과|및|/)\s*보습(?:제|\s*크림)',part))
            scope=mentioned if shared or _COMMON.search(part) else set()
            if not shared:ambiguities.append(part)
        elif _COMMON.search(part):scope=set(needed)
        elif mentioned:scope=mentioned
        elif active:scope=active
        elif len(needed)==1:scope=set(needed)
        else:scope=set()
        if mentioned or _COMMON.search(part):active=scope
        if _REVERSED.search(part):
            issues.append('product_fragrance_condition_contradicted')
        elif _FRAGRANCE.search(part) and not _EXCLUDED.search(part):
            covered.update(scope)
        if _LABEL_USE.search(part):label_covered.update(scope)
        if _STOP.search(part):stop_covered.update(scope)
    for p in sorted(needed-covered):issues.append('product_fragrance_missing_'+p)
    for p in sorted(label_needed-label_covered):issues.append('product_labeled_use_missing_'+p)
    for p in sorted(stop_needed-stop_covered):issues.append('product_irritation_stop_missing_'+p)
    text='\n'.join([s.get('content','') for s in steps]+[parsed.get('answer','')])
    advisories=['product_internal_placeholder_removed'] if _INTERNAL.search(text) else []
    care=review_care(parsed,response,bundle)
    return {'version':VERSION,'issues':list(dict.fromkeys(issues+care['issues'])),
            'disease_care':care,
            'advisories':advisories,
            'selected_products':sorted(selected),'required_products':sorted(required),
            'fragrance_covered':sorted(covered),'fragrance_needed':sorted(needed),
            'labeled_use_covered':sorted(label_covered),'labeled_use_needed':sorted(label_needed),
            'irritation_stop_covered':sorted(stop_covered),'irritation_stop_needed':sorted(stop_needed),
            'ambiguous_clauses':ambiguities,'scope':'product-specific fragrance text; not clinical validation'}
