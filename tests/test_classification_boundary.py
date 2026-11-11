from dataclasses import replace
import json
from pathlib import Path

import pytest

from server.classification_boundary import DISCLAIMER, model_input, review_boundary
from server.grounded_generation import assess,generate_grounded
from server.ingredient_evidence import load_evidence,extend_extractor
from src.llm.ingredients import IngredientExtractor
from src.llm.output_filter import OutputFilter
from src.llm.inference import RecResult
from src.llm.schema import RecInput
from server.schemas import DiseasePrediction

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/ingredient_evidence_20261110/test_final'


def old_rows():
    return [json.loads(s) for s in (OLD/'responses.jsonl').read_text(encoding='utf-8').splitlines()]


def test_known_missed_case_is_caught_in_raw_text():
    p={'reasoning':[{'content':'피부 타입이 지성이고 주요 고민은 여드름/뾰루지이며, 이미지 분류 참고 점수가 0.800으로 여드름성이 높습니다. 이는 임상적 확진을 의미하지 않습니다.'}],
       'answer':'기본 보습 정보를 안내합니다.'}
    issues=review_boundary(p)
    assert 'classification_score_in_generated_text' in issues
    assert 'classification_as_patient_state' in issues
    # A separate "not a diagnosis" sentence cannot excuse the preceding claim.
    assert 'classification_boundary_instruction_missing' in issues


@pytest.mark.parametrize('text',[
    '분류 신뢰도 99.5%이므로 질환 가능성이 높습니다.',
    '여드름성이 높습니다.',
    '정상으로 분류됐으므로 피부에 이상이 없습니다.',
    '아토피 가능성이 높지만 확진은 아닙니다.',
])
def test_patient_state_is_checked_beyond_the_first_step(text):
    p={'reasoning':[{'content':DISCLAIMER},{'content':'기본 보습 정보를 안내합니다.'},{'content':'제품 표시를 따릅니다.'}],'answer':text}
    assert review_boundary(p)


def test_correct_qualification_and_product_spf_are_preserved():
    p={'reasoning':[{'content':DISCLAIMER},{'content':'징크옥사이드 함유 완제품 선크림은 선택지입니다.'},
        {'content':'광범위 SPF30 이상 내수성 표시 제품을 선택하세요.'}],
       'answer':'분류 점수는 개인의 증상 정도를 알려 주지 않습니다. 질환 가능성이 높다고 판단할 수 없습니다.'}
    assert not review_boundary(p)


@pytest.mark.skipif(not (OLD/'responses.jsonl').exists(), reason='local historical benchmark artifacts unavailable')
def test_all_seven_previous_passes_still_pass_the_shared_guard():
    store=load_evidence();ext=extend_extractor(IngredientExtractor.load(ROOT/'data/processed/sllm/ingredient_dict.json'),store)
    for r in old_rows():
        if r['arm']!='with' or r['case_id']=='acne_oily':continue
        x=RecInput.from_dict(r['input']);d=DiseasePrediction(**r['prediction'])
        response,_,review=assess(RecResult(**r['result']),x,r['common_guard_response']['input_notes'],ext,OutputFilter.from_yaml(),d,r['bundle'])
        # Product coverage is a later policy; this regression protects the
        # original classification/source boundary without grandfathering usage.
        assert not [i for i in review['issues'] if not i.startswith('product_')],(r['case_id'],review)
        assert response.quality_status=='ok'
        assert [i.model_dump() for i in response.ingredients]==r['common_guard_response']['ingredients']


@pytest.mark.skipif(not (OLD/'responses.jsonl').exists(), reason='local historical benchmark artifacts unavailable')
def test_scores_stay_in_metadata_but_not_server_model_prompt():
    r=next(r for r in old_rows() if r['arm']=='with' and r['case_id']=='acne_oily')
    x=RecInput.from_dict(r['input']);d=DiseasePrediction(**r['prediction']);before=x.to_dict();metadata=d.model_dump()
    y=model_input(x,d)
    assert x.to_dict()==before and d.model_dump()==metadata
    assert '분류 점수 0.800' not in y.user_note
    assert '참고 라벨: 여드름' in y.user_note and '자동 변환' in y.user_note
    assert y.gender==x.gender and y.age==x.age and y.skin_type==x.skin_type and y.concerns==x.concerns
    # Do not erase actual user-reported numbers or turn them into trusted facts.
    x.user_note += '\n사용자 추가 요청: 점수 99.5면 증상이 심한가요?'
    assert '점수 99.5면' in model_input(x,d).user_note


@pytest.mark.skipif(not (OLD/'responses.jsonl').exists(), reason='local historical benchmark artifacts unavailable')
def test_unsafe_generation_retries_instead_of_rewriting_or_succeeding_by_withholding():
    r=next(r for r in old_rows() if r['arm']=='with' and r['case_id']=='acne_oily')
    bad=RecResult(**r['result']);good=replace(bad,parsed=json.loads(json.dumps(bad.parsed)))
    good.parsed['reasoning'][0]['content']=DISCLAIMER
    good=replace(good,raw_text=json.dumps(good.parsed,ensure_ascii=False))
    calls=[]
    class Model:
        def recommend_with_context(self,x,context,**kw):
            calls.append((x.to_dict(),context));return bad if len(calls)==1 else good
    store=load_evidence();ext=extend_extractor(IngredientExtractor.load(ROOT/'data/processed/sllm/ingredient_dict.json'),store)
    x=RecInput.from_dict(r['input']);x.user_note='선크림 사용 조건을 알려주세요.'
    result,response,_,review,trace=generate_grounded(Model(),x,[],ext,OutputFilter.from_yaml(),DiseasePrediction(**r['prediction']),r['bundle'])
    assert len(calls)==2 and trace[0]['result']['parsed']==bad.parsed
    assert not review['issues'] and response.ingredients and response.routine
    assert result.parsed==good.parsed
    assert all('분류 점수 0.800' not in x['user_note'] for x,_ in calls)
