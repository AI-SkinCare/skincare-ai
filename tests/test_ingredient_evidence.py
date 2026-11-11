import json
from pathlib import Path

import pytest

from server.ingredient_evidence import (load_evidence, retrieve_evidence, extend_extractor,
    evidence_review, enforce_evidence, EVIDENCE_GUIDANCE)
from server.schemas import DiseasePrediction, RecommendResponse
from server.recommendation_response import prepare_recommendation
from src.llm.schema import RecInput
from src.llm.ingredients import IngredientExtractor
from src.llm.inference import RecResult, SLLMRecommender
from src.llm.output_filter import OutputFilter


def prediction(label='atopic', uncertain=False):
    names={'atopic':'아토피','psoriasis':'건선','seborrheic':'지루','rosacea':'주사','acne':'여드름','normal':'정상'}
    return DiseasePrediction(label=names[label],label_en=label,confidence=.8,uncertain=uncertain,
        probabilities={v:.8 if k==label else .04 for k,v in names.items()})


def input_value(**kw):
    return RecInput(gender='여성',age=30,skin_type='건성',concerns=['민감성(트러블/자극감)'],**kw)


def candidate(text='글리세린은 수분 유지와 건조함 관리를 돕는 보습 성분입니다 [E_GLY_MOIST].', routine=None):
    return {'reasoning':[
        {'step':1,'title':'문제 분석','content':'사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다.'},
        {'step':2,'title':'성분 선택 및 근거 제시','content':text},
        {'step':3,'title':'사용법 및 관리방안','content':routine or '매일 무향 글리세린 보습제를 제품 표시대로 사용하고 자극이 생기면 중단하세요.'}],
        'answer':'글리세린 함유 보습제로 기본 보습 관리를 하세요.'}


def response_and_review(parsed=None,label='atopic',avoid=()):
    s=load_evidence();x=input_value();b=retrieve_evidence(s,x,prediction(label),avoid)
    ext=extend_extractor(IngredientExtractor({}),s);p=parsed or candidate()
    r=RecResult(p,'fixture',False,1,json.dumps(p,ensure_ascii=False),100,1)
    res,_=prepare_recommendation(r,x,[],ext,OutputFilter.from_yaml(),prediction(label))
    return res,evidence_review(p,res,b)


def test_valid_source_binding_is_not_clinical_approval():
    res,rev=response_and_review()
    assert not rev['issues'],rev
    out=enforce_evidence(res,rev)
    assert out.evidence_status=='source_linked'
    assert out.ingredient_evidence[0]['sources'][0]['url'].startswith('https://')
    assert not rev['clinical_entailment_verified']
    assert out.ingredients[0].purpose_source=='model'


@pytest.mark.parametrize('text,code',[
    ('글리세린은 보습을 돕습니다.','ingredient_citation_missing'),
    ('글리세린은 피지를 조절합니다 [E_GLY_MOIST].','evidence_claim_outside_scope'),
    ('글리세린은 보습을 돕습니다 [E_NOT_REAL].','evidence_citation_unknown'),
    ('판테놀은 보습을 돕습니다 [E_GLY_MOIST].','ingredient_outside_evidence'),
    ('글리세린은 보습을 돕습니다 [E_GLY_MOIST]. https://example.com/trial','evidence_url_unverified'),
])
def test_citation_does_not_approve_unsupported_claim(text,code):
    # Include a local alias for out-of-store names, as the live corpus does.
    s=load_evidence();ext=extend_extractor(IngredientExtractor({'판테놀':'판테놀'}),s)
    p=candidate(text);x=input_value();b=retrieve_evidence(s,x,prediction())
    r,_=prepare_recommendation(RecResult(p,'fixture',False,1,'',1,1),x,[],ext,OutputFilter.from_yaml(),prediction())
    rev=evidence_review(p,r,b)
    assert code in rev['issues'],rev
    assert enforce_evidence(r,rev).quality_status=='withheld'


def test_conditions_and_spf_are_required_and_not_filled_in():
    p=candidate('징크옥사이드는 자외선 차단 완제품의 선택지입니다 [E_ZN_SUN].',
                '매일 무향 선크림을 사용하고 자극이 생기면 중단하세요.')
    r,rev=response_and_review(p,'rosacea')
    assert 'evidence_sunscreen_condition_missing' in rev['issues']
    out=enforce_evidence(r,rev)
    assert not out.ingredients and not out.routine and not out.reasoning


def test_acne_avoidance_age_and_uncertain_scopes():
    s=load_evidence();x=input_value(user_note='나이아신아마이드를 쓰면 따가워요.')
    b=retrieve_evidence(s,x,prediction('rosacea'))
    assert 'E_NIA_FORM' not in {c['id'] for c in b['claims']}
    assert retrieve_evidence(s,x,prediction(uncertain=True)) is None
    x.concerns.append('여드름/뾰루지')
    b=retrieve_evidence(s,x,prediction('psoriasis'))
    assert 'E_PET_MOIST' not in {c['id'] for c in b['claims']}
    x.age=15
    assert 'E_CER_MOIST' not in {c['id'] for c in retrieve_evidence(s,x,prediction())['claims']}
    assert not retrieve_evidence(s,x,prediction(),avoid=[a for c in s['claims'] for a in c['aliases']])['claims']


def test_study_dose_is_not_a_recommendation():
    r,rev=response_and_review(candidate(routine='매일 무향 글리세린 20%를 사용하고 자극이 생기면 중단하세요.'))
    assert 'evidence_unprescribed_concentration' in rev['issues']


def test_retrieval_uses_prediction_not_file_or_gold_and_preserves_input():
    x=input_value();before=x.to_dict();s=load_evidence()
    ad=retrieve_evidence(s,x,prediction('atopic'));sd=retrieve_evidence(s,x,prediction('seborrheic'))
    assert x.to_dict()==before
    assert ad['predicted_label']=='atopic' and sd['predicted_label']=='seborrheic'
    assert ad['care_conditions']!=sd['care_conditions']
    assert not any(c['id']=='E_CER_MOIST' for c in sd['claims'])


def test_request_context_is_local_and_retry_fallback_is_not_evidence():
    rec=SLLMRecommender('unused',None,serving_guidance='original')
    calls=[]
    rec.generate_raw=lambda x,sample=False,trusted_context=None:(calls.append(trusted_context) or json.dumps(candidate(),ensure_ascii=False),1)
    rec.recommend_with_context(input_value(),EVIDENCE_GUIDANCE)
    rec.recommend(input_value())
    assert calls==[EVIDENCE_GUIDANCE,None] and rec.serving_guidance=='original'


def test_same_source_does_not_merge_different_sunscreen_ingredients():
    s=load_evidence();ext=extend_extractor(IngredientExtractor({}),s)
    assert ext.extract('산화아연, 이산화티타늄')==['징크옥사이드','티타늄디옥사이드']


def test_store_rejects_unreviewed_hosts(tmp_path):
    s=load_evidence();s['sources']['AAD_DRY']['url']='https://example.com/fake'
    p=tmp_path/'bad.json';p.write_text(json.dumps(s),encoding='utf-8')
    with pytest.raises(ValueError):load_evidence(p)


def test_live_factory_propagates_source_binding_and_early_gates(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from server import main
    from server.config import Settings
    calls=[]
    class Recorder:
        version='test_evidence_fixture'
        def __init__(self,*a,**k):pass
        def load(self):return self
        def recommend_with_context(self,x,context,**kw):
            calls.append((x.to_dict(),context))
            return RecResult(candidate(),self.version,False,1,json.dumps(candidate(),ensure_ascii=False),1,1)
        def recommend(self,x):raise AssertionError('evidence path must be used')
    monkeypatch.setattr(main,'SLLMRecommender',Recorder)
    (tmp_path/'ingredient_dict.json').write_text(json.dumps({'lexicon':{'글리세린':'글리세린'}}),encoding='utf-8')
    cfg=Settings(internal_token='test',rec_backend='sllm',rec_evidence_mode='enforce',processed_dir=tmp_path)
    s=dict(gender='female',age=30,skinType='dry',selfReportedConcerns=[])
    headers={'X-Internal-Token':'test'}
    with TestClient(main.create_app(cfg)) as client:
        endpoint='/infer/disease-recommend'
        assert client.post(endpoint,json={'survey':s,'disease':prediction().model_dump()}).status_code==401
        body=client.post(endpoint,headers=headers,json={'survey':s,'disease':prediction().model_dump()}).json()
        assert body['recommendation_status']=='generated',body
        rec=body['recommendation'];assert rec['evidence_status']=='source_linked'
        assert rec['ingredient_evidence'][0]['claim_id']=='E_GLY_MOIST'
        assert calls[0][1].startswith(EVIDENCE_GUIDANCE) and 'E_GLY_MOIST' in calls[0][1]
        assert 'E_GLY_MOIST' not in calls[0][0]['user_note']
        uncertain=client.post(endpoint,headers=headers,json={'survey':s,'disease':prediction(uncertain=True).model_dump()}).json()
        normal=client.post(endpoint,headers=headers,json={'survey':s,'disease':prediction('normal').model_dump()}).json()
        assert uncertain['recommendation_status']=='uncertain' and normal['recommendation_status']=='no_concern'
        s['avoidIngredients']=[c['ingredient'] for c in load_evidence()['claims']]
        excluded=client.post(endpoint,headers=headers,json={'survey':s,'disease':prediction().model_dump()}).json()
        assert excluded['recommendation_status']=='quality_rejected'
        assert excluded['quality_issues']==['evidence_no_eligible_claims']
        assert excluded['generation_origin']=='not_generated'
        assert uncertain['generation_origin']=='not_generated'
        assert normal['generation_origin']=='not_generated'
        assert body['generation_origin']==rec['generation_origin']
        assert len(calls)==1


def test_spurious_source_on_another_ingredient_is_not_sufficient():
    r,rev=response_and_review(candidate('글리세린은 보습을 돕습니다 [E_CER_MOIST].'))
    assert 'ingredient_citation_missing' in rev['issues']
    assert enforce_evidence(r,rev).evidence_status=='withheld'


def test_invented_numeric_citation_and_reversed_condition_are_rejected():
    _,rev=response_and_review(candidate('글리세린은 보습을 돕습니다 [E_100001].',
        '매일 무향 제품을 사용하고 자극이 생겨도 중단하지 말고 계속 사용하세요.'))
    assert 'evidence_citation_unknown' in rev['issues']
    assert 'evidence_condition_contradicted' in rev['issues']


def test_temporary_adapter_disable_restores_after_exception():
    from contextlib import contextmanager
    class Model:
        enabled=True
        @contextmanager
        def disable_adapter(self):
            self.enabled=False
            try:yield
            finally:self.enabled=True
    rec=SLLMRecommender('unused',None);rec.model=Model()
    seen=[]
    def fail(*a,**kw):
        seen.append(rec.model.enabled)
        raise RuntimeError('fixture failure')
    rec.generate_raw=fail
    with pytest.raises(RuntimeError):rec.recommend_with_context(input_value(),'context',disable_adapter=True)
    assert seen==[False] and rec.model.enabled


def test_inline_citation_is_removed_without_inventing_purpose():
    from server.grounded_generation import assess
    from server.ingredient_evidence import without_citation_marks
    s=load_evidence();x=input_value();p=candidate('글리세린 [E_GLY_MOIST]은 수분 유지에 도움을 주는 보습 성분입니다.')
    ext=extend_extractor(IngredientExtractor({}),s);b=retrieve_evidence(s,x,prediction())
    result=RecResult(p,'fixture',False,1,json.dumps(p,ensure_ascii=False),1,1)
    r,_,rev=assess(result,x,[],ext,OutputFilter.from_yaml(),prediction(),b)
    assert r.ingredients[0].purpose and '[E_' not in r.ingredients[0].purpose
    assert not rev['issues'],rev
    assert p['reasoning'][1]['content'].startswith('글리세린 [E_')
    assert without_citation_marks(candidate('글리세린 [E_GLY_MOIST].'))['reasoning'][1]['content']=='글리세린.'


def test_semantic_repair_records_failures_and_keeps_fallback_separate():
    from server.grounded_generation import generate_grounded
    from unittest.mock import Mock
    s=load_evidence();x=input_value();b=retrieve_evidence(s,x,prediction())
    good=candidate('글리세린 [E_GLY_MOIST]은 수분 유지에 도움을 주는 보습 성분입니다.')
    bad=candidate('글리세린은 피지를 조절합니다 [E_GLY_MOIST].')
    core=Mock()
    core.recommend_with_context.side_effect=[RecResult(bad,'fixture',False,1,json.dumps(bad,ensure_ascii=False),30,1),RecResult(good,'fixture',False,1,json.dumps(good,ensure_ascii=False),40,1)]
    r,response,_,review,trace=generate_grounded(core,x,[],extend_extractor(IngredientExtractor({}),s),OutputFilter.from_yaml(),prediction(),b)
    assert len(trace)==2 and trace[0]['review']['issues']
    assert r.attempts==2 and r.output_tokens==70 and not r.used_fallback
    assert not review['issues'] and response.quality_status=='ok'


def test_formulation_purpose_and_declarative_usage_keep_their_scope():
    from server.grounded_generation import assess
    s=load_evidence();x=input_value();b=retrieve_evidence(s,x,prediction())
    p=candidate('세라마이드 함유 보습 제형은 수분 유지와 피부 장벽 보조 관리의 후보가 될 수 있다. [E_CER_MOIST]',
                '무향의 완제품 보습제를 제품 표시대로 사용한다. 자극이 생기면 사용을 중단한다.')
    p['answer']='세라마이드 함유 보습제를 표시대로 사용한다.'
    r,_,review=assess(RecResult(p,'fixture',False,1,json.dumps(p,ensure_ascii=False),1,1),x,[],extend_extractor(IngredientExtractor({}),s),OutputFilter.from_yaml(),prediction(),b)
    assert r.ingredients[0].purpose_source=='model_formulation'
    assert r.ingredients[0].purpose.startswith('함유 보습 제형') and r.routine
    assert not review['issues'],review


def test_default_real_model_uses_evidence_but_mock_remains_legacy(monkeypatch):
    from server.config import Settings
    monkeypatch.delenv('REC_EVIDENCE_MODE',raising=False)
    assert Settings(rec_backend='sllm').rec_evidence_mode=='enforce'
    assert Settings(rec_backend='mock').rec_evidence_mode=='off'
    assert Settings(rec_backend='sllm',rec_evidence_mode='off').rec_evidence_mode=='off'
