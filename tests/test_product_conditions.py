from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from server.product_conditions import review_usage,requested_products,prompt_usage
from server.ingredient_evidence import load_evidence,retrieve_evidence,extend_extractor
from server.schemas import DiseasePrediction
from server.grounded_generation import assess,generate_grounded
from src.llm.schema import RecInput
from src.llm.inference import RecResult
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/classification_boundary_20261110'


def examples():
    return [json.loads(s) for s in (OLD/'novel_v1/responses.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]


def fixture(routine):
    store=load_evidence();claims=[c for c in store['claims'] if c['id'] in ('E_GLY_MOIST','E_ZN_SUN')]
    bundle={'claims':claims}
    parsed={'reasoning':[{'title':'문제 분석','content':'사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다.'},
        {'title':'성분 선택','content':'글리세린, 징크옥사이드'},{'title':'사용법','content':routine}],'answer':'기본 관리 안내'}
    response=SimpleNamespace(ingredients=[SimpleNamespace(name=c['ingredient']) for c in claims])
    return parsed,response,bundle


def test_sunscreen_fragrance_does_not_cover_moisturizer():
    p,r,b=fixture('보습제는 세정 뒤 사용하세요. 선크림은 광범위 SPF30 이상 내수성 표시가 있는 무향 완제품을 선택하세요.')
    review=review_usage(p,r,b)
    assert review['fragrance_covered']==['sunscreen']
    assert 'product_fragrance_missing_moisturizer' in review['issues']


@pytest.mark.parametrize('routine',[
    '보습제: 무향 완제품을 표시대로 사용하세요. 선크림: SPF30 이상 내수성 무향 완제품을 사용하세요.',
    '보습제와 선크림은 모두 무향 완제품을 선택하세요.',
    '공통: 모든 제품은 무향으로 선택하고 자극이 생기면 중단하세요.',
    '보습제는 무향으로 고르세요. 선크림은 SPF30 이상, 내수성이 있는 무향 완제품을 고르세요.',
])
def test_clear_product_or_explicit_shared_rule_is_accepted(routine):
    assert not any('fragrance' in i for i in review_usage(*fixture(routine))['issues'])


def test_same_clause_does_not_share_an_unrelated_condition():
    p,r,b=fixture('보습제는 일반 제품을 쓰고 선크림은 무향을 쓰세요.')
    assert 'product_fragrance_missing_moisturizer' in review_usage(p,r,b)['issues']


def test_omission_cannot_fix_a_two_product_request():
    p,r,b=fixture('선크림은 무향으로 고르세요.')
    r.ingredients=r.ingredients[1:]
    assert 'product_recommendation_missing_moisturizer' in review_usage(p,r,b,required_products={'moisturizer','sunscreen'})['issues']


def test_reversed_and_internal_conditions_do_not_pass():
    p,r,b=fixture('보습제는 무향이 아니어도 됩니다. 선크림은 무향을 쓰세요. [care_conditions]')
    review=review_usage(p,r,b)
    assert 'product_fragrance_condition_contradicted' in review['issues']
    assert 'product_internal_placeholder_removed' in review['advisories']


def test_one_product_generic_condition_remains_usable():
    p,r,b=fixture('무향 완제품을 표시대로 사용하고 자극 시 중단하세요.')
    r.ingredients=r.ingredients[:1]
    assert not review_usage(p,r,b)['issues']


@pytest.mark.skipif(not (OLD/'novel_v1/responses.jsonl').exists(),reason='local prior outputs unavailable')
def test_known_failure_is_caught_and_prior_nineteen_fragrance_passes_are_preserved():
    rr=examples();dev=[json.loads(s) for s in (OLD/'development_v1/responses.jsonl').read_text(encoding='utf-8').splitlines()]
    passed=dev+[r for r in rr if r['arm']=='v2_with' and r['case_id']!='ps_oily_acne']
    assert len(passed)==19
    for row in passed:
        response=SimpleNamespace(ingredients=[SimpleNamespace(**i) for i in row['common_response']['ingredients']])
        assert not any('fragrance' in s for s in review_usage(row['result']['parsed'],response,row['bundle'])['issues']),row['case_id']
    row=next(r for r in rr if r['arm']=='v2_with' and r['case_id']=='ps_oily_acne')
    response=SimpleNamespace(ingredients=[SimpleNamespace(**i) for i in row['common_response']['ingredients']])
    assert 'product_fragrance_missing_moisturizer' in review_usage(row['result']['parsed'],response,row['bundle'])['issues']


def test_source_usage_lines_are_copies_not_new_medical_facts():
    store=load_evidence();text=prompt_usage({'claims':store['claims'],'care_conditions':[],'excluded':[],'scope':store['scope']})
    for c in store['claims']:
        for condition in c['conditions']:assert condition in text
    x=RecInput(gender='여성',age=30,skin_type='중성',concerns=['민감성(트러블/자극감)'],user_note='보습제와 선크림의 선택 조건을 알려주세요.')
    assert requested_products(x)=={'moisturizer','sunscreen'}


def test_restricted_all_products_clause_does_not_change_product_scope():
    p,r,b=fixture('보습제는 일반 제품을 쓰세요. 선크림은 모든 제품을 무향으로 선택하세요.')
    assert 'product_fragrance_missing_moisturizer' in review_usage(p,r,b)['issues']


def test_reverse_shared_subject_is_unambiguous():
    assert not any('fragrance' in i for i in review_usage(*fixture('선크림과 보습제는 무향 완제품을 사용하세요.'))['issues'])


@pytest.mark.skipif(not (OLD/'novel_v1/responses.jsonl').exists(),reason='local prior outputs unavailable')
def test_only_failed_product_conditions_trigger_a_recorded_model_repair():
    from dataclasses import replace
    from copy import deepcopy
    from server.ingredient_evidence import enforce_evidence
    row=next(r for r in examples() if r['arm']=='v2_with' and r['case_id']=='ps_oily_acne')
    bad=RecResult(**row['result']);p=deepcopy(bad.parsed)
    p['reasoning'][2]['content']='보습제: 무향 완제품을 제품 표시대로 사용하고 자극이 생기면 중단하세요. 선크림: 광범위 SPF30 이상 내수성 표시가 있는 무향 완제품을 선택하고 제품 표시를 따르며 자극이 생기면 중단하세요.'
    good=replace(bad,parsed=p,raw_text=json.dumps(p,ensure_ascii=False),attempts=1)
    calls=[]
    class Model:
        def recommend_with_context(self,x,context,**kw):
            calls.append(context);return replace(bad,attempts=1) if len(calls)==1 else good
    ext=extend_extractor(IngredientExtractor.load(ROOT/'data/processed/sllm/ingredient_dict.json'),load_evidence())
    res,response,_,review,trace=generate_grounded(Model(),RecInput.from_dict(row['input']),[],ext,OutputFilter.from_yaml(),DiseasePrediction(**row['prediction']),row['bundle'])
    assert [t['phase'] for t in trace]==['source','product_repair']
    assert trace[0]['result']['parsed']==bad.parsed and res.parsed==good.parsed
    assert not review['issues'] and enforce_evidence(response,review).evidence_status=='source_linked'
    assert '교정할 초안' in calls[1] and res.attempts==2


@pytest.mark.skipif(not (OLD/'novel_v1/responses.jsonl').exists(),reason='local prior outputs unavailable')
def test_valid_existing_generation_keeps_the_exact_frozen_v2_context():
    from scripts.compare_product_conditions import frozen_v2
    row=next(r for r in examples() if r['arm']=='v2_with' and r['case_id']=='ad_adult_basic')
    result=RecResult(**row['result']);old,_=frozen_v2();contexts=[]
    class Model:
        def recommend_with_context(self,x,context,**kw):contexts.append(context);return result
    ext=extend_extractor(IngredientExtractor.load(ROOT/'data/processed/sllm/ingredient_dict.json'),load_evidence())
    args=(RecInput.from_dict(row['input']),[],ext,OutputFilter.from_yaml(),DiseasePrediction(**row['prediction']),row['bundle'])
    old.generate_grounded(Model(),*args);previous=contexts[:];contexts.clear()
    _,response,_,review,trace=generate_grounded(Model(),*args)
    assert contexts==previous and len(contexts)==1
    assert not review['issues'] and response.ingredients and len(trace)==1


def test_display_marker_cleanup_never_supplies_missing_fragrance():
    from server.product_conditions import display_parsed
    p,r,b=fixture('보습제는 세정 뒤 사용하세요. 선크림은 무향으로 고르세요. [care_conditions]')
    clean=display_parsed(p)
    assert '[care_conditions]' not in clean['reasoning'][2]['content']
    assert '[care_conditions]' in p['reasoning'][2]['content']
    assert clean['reasoning'][2]['content'].count('무향')==1
    assert 'product_fragrance_missing_moisturizer' in review_usage(clean,r,b)['issues']


def test_moisturizer_label_use_cannot_cover_sunscreen_and_shared_stop_is_explicit():
    p,r,b=fixture('보습제는 무향 완제품을 제품 표시대로 사용한다. 자극이 생기면 중단한다. '
        '선크림은 광범위 SPF30 이상 내수성 무향 완제품을 선택한다. 자극이 생기면 중단한다.')
    review=review_usage(p,r,b)
    assert review['issues']==['product_labeled_use_missing_sunscreen']
    assert review['labeled_use_covered']==['moisturizer']
    p['reasoning'][2]['content']+=' 두 제품 공통으로 제품 표시를 따르며 자극이 생기면 중단한다.'
    assert not review_usage(p,r,b)['issues']


def test_correction_routine_is_source_derived_context_not_a_response_patch():
    from server.product_conditions import routine_lines
    p,r,b=fixture('보습제 누락 초안')
    text=routine_lines(b,['징크옥사이드'],{'moisturizer','sunscreen'})
    assert text.startswith('보습제: ') and '\n선크림: ' in text
    for c in b['claims']:
        for condition in c['conditions']:assert condition in text
    assert p['reasoning'][2]['content']=='보습제 누락 초안'
