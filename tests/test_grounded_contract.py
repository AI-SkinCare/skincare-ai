import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.grounded_contract import review_output,select_cards,recovery_prompt,review_recovery
from server.ingredient_evidence import load_evidence,extend_extractor,enforce_evidence
from server.grounded_generation import generate_grounded
from server.schemas import DiseasePrediction
from src.llm.inference import RecResult
from src.llm.schema import RecInput
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/product_conditions_20261110'


def result(parsed,raw=None):
    return RecResult(parsed,'fixture',False,1,json.dumps(parsed,ensure_ascii=False) if raw is None else raw,1,1)


def payload():
    return {'reasoning':[{'step':1,'title':'문제 분석','content':'사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다.'},
        {'step':2,'title':'성분 선택','content':'글리세린 함유 보습 제형 안내'},
        {'step':3,'title':'사용법','content':'광범위SPF30 이상 제품을 선택한다.'}],'answer':'글리세린 함유 보습제 안내'}


def test_clean_korean_and_legitimate_unspaced_spf_are_not_corruption():
    assert not review_output(result(payload()))['issues']


def test_partial_outer_json_and_inner_fence_are_not_a_single_payload():
    p=payload();raw='{"reasoning": [ unfinished\n```json\n'+json.dumps(p,ensure_ascii=False)+'\n```'
    assert 'contract_not_single_json' in review_output(result(p,raw))['issues']
    assert not review_output(result(p,'```json\n'+json.dumps(p,ensure_ascii=False)+'\n```'))['issues']


@pytest.mark.parametrize('bad,issue',[
    ('페트롤라Türkiye','contract_corrupt_mixed_script'),('甘油和凡士林','contract_non_korean_script'),
    ('페트ROL라','contract_corrupt_mixed_script')])
def test_broken_scripts_are_rejected_without_silent_replacement(bad,issue):
    p=payload();p['answer']=bad;res=result(p)
    assert issue in review_output(res)['issues'] and res.parsed['answer']==bad and bad in res.raw_text


def test_source_spec_keeps_one_card_for_each_requested_kind():
    store=load_evidence();bundle={'claims':[c for c in store['claims'] if c['id'] in ('E_GLY_MOIST','E_CER_MOIST','E_ZN_SUN')]}
    response=SimpleNamespace(ingredients=[SimpleNamespace(name=n) for n in ('글리세린','세라마이드')])
    cards=select_cards(bundle,response,{'moisturizer','sunscreen'})
    assert [c['id'] for c in cards]==['E_GLY_MOIST','E_ZN_SUN']
    text=recovery_prompt(cards)
    blueprint=json.loads(text.split('복사할 검토 자료(JSON 초안):\n',1)[1])
    for card in cards:
        assert card['effect'] in text
        for condition in card['conditions']:assert condition in blueprint['reasoning'][2]['content']
    # No omitted or avoided card can be reintroduced by a cached global choice.
    filtered={'claims':[c for c in bundle['claims'] if c['ingredient']!='글리세린']}
    assert [c['id'] for c in select_cards(filtered,response,{'moisturizer','sunscreen'})]==['E_CER_MOIST','E_ZN_SUN']


def test_a_routine_only_sunscreen_is_not_a_complete_recovery():
    cards=[c for c in load_evidence()['claims'] if c['id'] in ('E_GLY_MOIST','E_ZN_SUN')]
    p=payload();p['reasoning'][1]['content']=cards[0]['effect']+' [E_GLY_MOIST]'
    p['reasoning'][2]['content']=' '.join(t for c in cards for t in c['conditions'])
    response=SimpleNamespace(ingredients=[SimpleNamespace(name='글리세린')])
    issues=review_recovery(p,response,cards)
    assert 'contract_recovery_selection_mismatch' in issues and 'contract_recovery_effect_missing' in issues


@pytest.mark.skipif(not (OLD/'novel_v1/responses.jsonl').exists(),reason='local historical outputs unavailable')
def test_invalid_source_draft_can_recover_from_complete_source_spec():
    from scripts.compare_classification_boundary import rows
    row=next(r for r in rows(OLD/'novel_v1/responses.jsonl') if r['arm']=='v3_with' and r['case_id']=='mix_ros_avoid_nia')
    bad=RecResult(**row['result']);cards=[c for c in row['bundle']['claims'] if c['id'] in ('E_GLY_MOIST','E_ZN_SUN')]
    p=payload();p['reasoning'][1]['content']='\n'.join(c['effect']+' ['+c['id']+']' for c in cards)
    p['reasoning'][2]['content']='\n'.join(c['ingredient']+': '+' '.join(c['conditions']) for c in cards)
    # Product labels matter as well as names.
    p['reasoning'][2]['content']='\n'.join(('선크림' if c['id'].endswith('_SUN') else '보습제')+'('+c['ingredient']+'): '+' '.join(c['conditions']) for c in cards)
    p['answer']='글리세린 함유 보습제와 징크옥사이드 함유 완제품 선크림을 선택합니다.'
    good=result(p);calls=[]
    class Model:
        def recommend_with_context(self,x,context,**kwargs):
            calls.append(context)
            return good if '복사할 검토 자료(JSON 초안):' in context else bad
    ext=extend_extractor(IngredientExtractor.load(ROOT/'data/processed/sllm/ingredient_dict.json'),load_evidence())
    res,response,_,review,trace=generate_grounded(Model(),RecInput.from_dict(row['input']),row['notes'],ext,OutputFilter.from_yaml(),DiseasePrediction(**row['prediction']),row['bundle'])
    assert trace[-1]['phase']=='source_contract_recovery' and res.parsed==p and res.raw_text==good.raw_text
    assert not review['issues'] and enforce_evidence(response,review).evidence_status=='source_linked'
    assert '교정할 초안' not in calls[-1] and '나이아신아마이드' not in calls[-1]


def test_payload_mismatch_is_not_accepted_as_real_model_output():
    p=payload();res=result(p);other=payload();other['answer']='서버가 나중에 붙인 답변'
    assert 'contract_payload_parse_mismatch' in review_output(replace(res,parsed=other))['issues']


def test_json_escape_preserves_canonical_pet_without_post_generation_edit():
    card=next(c for c in load_evidence()['claims'] if c['id']=='E_PET_MOIST')
    prompt=recovery_prompt([card]);raw=prompt.split('복사할 검토 자료(JSON 초안):\n',1)[1]
    assert '\\ud398' in raw and '페트롤라툼' not in raw
    decoded=json.loads(raw)
    assert card['effect']+' [E_PET_MOIST]'==decoded['reasoning'][1]['content']
    assert '페트롤라툼' in decoded['answer'] and not review_output(result(decoded,raw))['issues']


@pytest.mark.skipif(not (OLD/'novel_v1/responses.jsonl').exists(),reason='local historical outputs unavailable')
def test_source_card_delivery_is_explicit_and_never_overwrites_model_result():
    from scripts.compare_classification_boundary import rows
    from server.grounded_generation import assess,source_card_response
    row=next(r for r in rows(OLD/'novel_v1/responses.jsonl') if r['arm']=='v3_with' and r['case_id']=='mix_ros_avoid_nia')
    bad=replace(RecResult(**row['result']),attempts=1);calls=[]
    class Model:
        def recommend_with_context(self,x,context,**kwargs):calls.append(context);return bad
    ext=extend_extractor(IngredientExtractor.load(ROOT/'data/processed/sllm/ingredient_dict.json'),load_evidence())
    args=(RecInput.from_dict(row['input']),row['notes'],ext,OutputFilter.from_yaml(),DiseasePrediction(**row['prediction']),row['bundle'])
    res,served,_,review,trace=generate_grounded(Model(),*args)
    assert res.parsed==bad.parsed and res.raw_text==bad.raw_text and len(trace)==len(calls)
    assert served.generation_origin=='source_cards' and served.used_fallback and served.model_version.startswith('source_cards_')
    assert all(i.purpose_source=='source_card' for i in served.ingredients)
    assert enforce_evidence(served,review).evidence_status=='source_linked'
    raw_model,_,raw_review=assess(res,*args,recovery_ids=trace[-1]['recovery_ids'])
    assert enforce_evidence(raw_model,raw_review).evidence_status=='withheld'
    assert review['source_delivery']['payload']!=res.parsed
    with pytest.raises(ValueError):source_card_response(res,*args,['E_INVENTED'],raw_review)
    res2,served2,_,review2,_=generate_grounded(Model(),*args,include_evidence=False)
    assert served2.generation_origin=='model' and 'source_delivery' not in review2
    assert enforce_evidence(served2,review2).evidence_status=='withheld'
