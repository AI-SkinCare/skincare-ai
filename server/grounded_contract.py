"""Korean source-output contract and bounded recovery from eligible cards.

This module never edits a model answer. A recovery specification is prompt data;
the model must actually generate it and every normal serving check still runs.
It adds no medical claims and is not a clinical safety validator.
"""
from __future__ import annotations

import json
import re

from .product_conditions import kind,LABELS
from .disease_care import care_text

VERSION='grounded_contract_v5'
NEUTRAL='사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다.'
_FENCE=re.compile(r'\A```(?:json)?\s*\n?(.*?)\n?```\s*\Z',re.S|re.I)
_HAN=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
_FOREIGN_LATIN=re.compile(r'[\u00c0-\u024f]')
_MIXED=re.compile(r'[가-힣]+[A-Za-z]+[가-힣]*')
_CITATION=re.compile(r'\[E_[A-Z0-9_]+\]')


def review_output(result):
    """Reject malformed outer payloads and script corruption without rewriting."""
    issues=[]
    if result.used_fallback:
        return {'version':VERSION,'issues':['contract_rule_fallback'],'single_json':False}
    raw=result.raw_text.strip();match=_FENCE.fullmatch(raw)
    if match:raw=match.group(1).strip()
    try:obj=json.loads(raw)
    except (ValueError,TypeError):obj=None
    single=isinstance(obj,dict)
    if not single:issues.append('contract_not_single_json')
    else:
        from src.llm.schema import parse_output
        if parse_output(raw)!=result.parsed:issues.append('contract_payload_parse_mismatch')
        if set(obj)!={'reasoning','answer'}:issues.append('contract_payload_extra_fields')
    if len(result.parsed.get('reasoning',[]))!=3:issues.append('contract_three_steps_missing')
    text='\n'.join(s.get('content','')+' '+s.get('title','') for s in result.parsed.get('reasoning',[]))+'\n'+result.parsed.get('answer','')
    text=_CITATION.sub('',text)
    if _HAN.search(text):issues.append('contract_non_korean_script')
    mixed=[m.group() for m in _MIXED.finditer(text)
           if not re.fullmatch(r'[가-힣]+(?:SPF|UVA|UVB|PA)[가-힣]*',m.group(),re.I)]
    if _FOREIGN_LATIN.search(text) or mixed:issues.append('contract_corrupt_mixed_script')
    return {'version':VERSION,'issues':issues,'single_json':single,'mixed_tokens':mixed,
            'scope':'output language and syntax only; not diagnosis or efficacy validation'}


def select_cards(bundle,response,required_products=()):
    """Keep eligible prior choices, replacing duplicate kinds when two are asked.

    Eligibility (disease routing, age, avoidance, acne-related exclusions) comes
    exclusively from the existing retrieval bundle, not from this selector.
    """
    by_name={c['ingredient']:c for c in bundle['claims']}
    previous=[by_name[i.name] for i in response.ingredients if i.name in by_name]
    required=set(required_products)&{kind(c) for c in bundle['claims']}
    if required:
        chosen=[]
        for product in ('moisturizer','sunscreen'):
            if product not in required:continue
            card=next((c for c in previous if kind(c)==product),None)
            if card is None:card=next(c for c in bundle['claims'] if kind(c)==product)
            chosen.append(card)
        # A one-kind request need not erase an already valid other-kind choice.
        for card in previous:
            if len(chosen)<2 and card not in chosen:chosen.append(card)
    else:chosen=previous[:2] or bundle['claims'][:1]
    return chosen


def source_payload(cards,bundle=None):
    """Exact existing card text, explicitly distinguishable from model output."""
    routine='\n'.join(LABELS[kind(c)]+'('+c['ingredient']+'): '+' '.join(c['conditions']) for c in cards)
    care=care_text(bundle or {},cards)
    if care:routine+='\n'+care
    return {'reasoning':[
        dict(step=1,title='문제 분석 및 원인 파악',content=NEUTRAL),
        dict(step=2,title='성분 선택 및 근거 제시',content='\n'.join(c['effect']+' ['+c['id']+']' for c in cards)),
        dict(step=3,title='사용법 및 관리방안',content=routine)],
        'answer':', '.join(c['ingredient']+' 함유 '+('완제품 선크림' if kind(c)=='sunscreen' else '보습 제형') for c in cards)}


def recovery_prompt(cards,bundle=None):
    """A source-derived JSON blueprint to transcribe, never a returned response.

    Rare proper names use ordinary JSON Unicode escapes in the prompt. JSON
    decoding yields the exact original card spelling; no word is patched later.
    The model may still fail, in which case normal validation withholds it.
    """
    payload=source_payload(cards,bundle)
    blueprint=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    # Escapes are a JSON encoding of the same canonical source string, not an
    # ingredient alias, a new fact or post-generation spelling replacement.
    for card in cards:
        if card['ingredient']=='페트롤라툼':
            blueprint=blueprint.replace(card['ingredient'],json.dumps(card['ingredient'],ensure_ascii=True)[1:-1])
    return ('검토된 근거 문구의 제한된 전달 작업입니다. 아래 JSON 초안은 허용된 성분·효능·조건만 담습니다. '
        '내용을 추가·삭제·번역하거나 성분을 다시 선택하지 마세요. 설명·마크다운·다른 JSON 없이 '
        '초안과 동일한 단일 JSON을 출력하세요. 한국어 문장과 근거 ID, 모든 제품의 사용법을 그대로 보존하세요. '
        'JSON의 Unicode 이스케이프는 원래 성분명의 표준 JSON 표현입니다. 역슬래시를 추가하지 마세요. '
        '개인 상태·원인·병변·중증도·추가 효능·횟수·대기시간은 만들지 마세요.\n'
        '복사할 검토 자료(JSON 초안):\n'+blueprint)


def review_recovery(parsed,response,cards):
    """Ensure a focused recovery really retained the specified names and effects."""
    steps=parsed.get('reasoning',[])
    effect=steps[1].get('content','') if len(steps)>1 else ''
    routine=steps[2].get('content','') if len(steps)>2 else ''
    compact=lambda s:re.sub(r'\s+','',_CITATION.sub('',s))
    actual={i.name for i in response.ingredients};wanted={c['ingredient'] for c in cards}
    issues=[]
    if actual!=wanted:issues.append('contract_recovery_selection_mismatch')
    for card in cards:
        if card['id'] not in effect or compact(card['effect']) not in compact(effect):
            issues.append('contract_recovery_effect_missing')
        if any(compact(t) not in compact(routine) for t in card['conditions']):
            issues.append('contract_recovery_conditions_missing')
    return list(dict.fromkeys(issues))
