"""One shared source-context generation/repair path for HTTP and paired eval."""
from __future__ import annotations

from dataclasses import asdict,replace
import json
import time

from .ingredient_evidence import (EVIDENCE_GUIDANCE,evidence_prompt,evidence_review,
                                 enforce_evidence,without_citation_marks)
from .recommendation_response import prepare_recommendation
from .classification_boundary import VERSION,GUIDANCE,model_input,review_boundary
from .product_conditions import (VERSION as USAGE_VERSION,GUIDANCE as USAGE_GUIDANCE,
                                 requested_products,prompt_usage,routine_lines,review_usage,display_parsed)
from .grounded_contract import (VERSION as CONTRACT_VERSION,review_output,select_cards,
                                recovery_prompt,review_recovery,source_payload)


def _assess_base(result,x,notes,extractor,output_filter,disease,bundle):
    normalized=replace(result,parsed=display_parsed(without_citation_marks(result.parsed,bundle)))
    response,findings=prepare_recommendation(normalized,x,notes,extractor,output_filter,disease)
    review=evidence_review(result.parsed,response,bundle)
    review['issues']=list(dict.fromkeys(review['issues']+review_boundary(result.parsed)))
    review['boundary_version']=VERSION
    response.postprocess_version += '+' + VERSION
    return response,findings,review


def _assess_products(result,x,notes,extractor,output_filter,disease,bundle):
    response,findings,review=_assess_base(result,x,notes,extractor,output_filter,disease,bundle)
    usage=review_usage(result.parsed,response,bundle,required_products=requested_products(x))
    review['issues']=list(dict.fromkeys(review['issues']+usage['issues']))
    review['product_conditions']=usage
    response.postprocess_version += '+' + USAGE_VERSION
    return response,findings,review


def assess(result,x,notes,extractor,output_filter,disease,bundle,*,recovery_ids=None):
    response,findings,review=_assess_products(result,x,notes,extractor,output_filter,disease,bundle)
    contract=review_output(result)
    review['issues']=list(dict.fromkeys(review['issues']+contract['issues']))
    review['output_contract']=contract
    if recovery_ids is not None:
        cards=[c for claim_id in recovery_ids for c in bundle['claims'] if c['id']==claim_id]
        selection_issues=review_recovery(result.parsed,response,cards)
        review['issues']=list(dict.fromkeys(review['issues']+selection_issues))
        review['source_recovery']=dict(version=CONTRACT_VERSION,selected_ids=list(recovery_ids),
            issues=selection_issues,origin='eligible_source_cards_not_new_medical_claims',
            mode='source_blueprint_transcription')
    response.postprocess_version+='+'+CONTRACT_VERSION
    return response,findings,review


def _generate_draft(model,x,notes,extractor,output_filter,disease,bundle,*,include_evidence=True,disable_adapter=True):
    """Preserve valid v3 drafts; route script/payload corruption to recovery early."""
    start=time.perf_counter();trace=[];feedback=''
    actual_input=model_input(x,disease)
    base=EVIDENCE_GUIDANCE+'\n'+GUIDANCE+(evidence_prompt(bundle) if include_evidence else '')
    for round_index in range(2):
        result=model.recommend_with_context(actual_input,base+feedback,disable_adapter=disable_adapter)
        response,findings,review=_assess_base(result,x,notes,extractor,output_filter,disease,bundle)
        issues=list(dict.fromkeys(review['issues']+response.quality_issues))
        trace.append(dict(round=round_index+1,result=asdict(result),review=review,common_quality_issues=response.quality_issues,
                          model_input=actual_input.to_dict(),boundary_version=VERSION,phase='source'))
        contract=review_output(result)
        if contract['issues']:
            trace[-1]['output_contract']=contract
            break
        if not issues and response.quality_status=='ok':break
        feedback=('\n직전 답변의 자동 검사 미통과 항목: '+json.dumps(issues,ensure_ascii=False)+
            '\n분류 점수·가능성·신뢰도를 개인 피부 상태로 설명하지 마세요. 첫 단계는 다음 한 문장만 쓰세요: 사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다. '
            '\n다시 작성하세요. 개인의 원인·장벽 손상·병변은 단정하지 마세요. 근거 자료가 있으면 각 성분의 effect 문장을 그대로 복사하고 끝에 해당 근거 ID를 붙이세요. 추가 효능 문장을 만들지 마세요. '
            '각 성분은 별도 문장으로 쓰고 세라마이드·나이아신아마이드 자료는 함유 제형의 연구 범위를 그대로 유지하세요. '
            '무향·제품 표시대로 사용·자극 시 중단, 선크림 선택 시 광범위 SPF30 이상 내수성 표시 조건을 사용법에 쓰세요. '
            '임의의 농도·횟수·대기시간·고농도·추가 효능은 만들지 마세요. 근거 자료가 없으면 없는 ID를 만들지 말고 추천을 보류하세요.')
    base_ok=not issues and response.quality_status=='ok'
    response,findings,review=_assess_products(result,x,notes,extractor,output_filter,disease,bundle)
    if base_ok and not contract['issues'] and review['product_conditions']['issues']:
        conditions=(prompt_usage(bundle,[i.name for i in response.ingredients],requested_products(x)) if include_evidence else '')
        exact_usage=(routine_lines(bundle,[i.name for i in response.ingredients],requested_products(x)) if include_evidence else '')
        repair=(base+'\n'+USAGE_GUIDANCE+conditions+'\n교정할 초안(JSON):\n'+json.dumps(result.parsed,ensure_ascii=False)+
            '\n누락·모호 항목: '+json.dumps(review['product_conditions']['issues'],ensure_ascii=False)+
            '\n첫 단계와 기존 성분의 효능·인용을 유지하세요. 두 제품이 요청됐으면 보습제를 빼지 마세요. '
            '초안의 step 3 사용법을 그대로 반복하지 마세요. 아래에 제시된 경우 step 3 content는 다음 제품별 문장으로 바꾸어 출력하세요:\n'+exact_usage+
            '\n답변 전체를 동일한 3단계 JSON으로 출력하세요. JSON만 출력하세요.')
        result=model.recommend_with_context(actual_input,repair,disable_adapter=disable_adapter)
        response,findings,review=_assess_products(result,x,notes,extractor,output_filter,disease,bundle)
        trace.append(dict(round=len(trace)+1,result=asdict(result),review=review,common_quality_issues=response.quality_issues,
            model_input=actual_input.to_dict(),boundary_version=VERSION,phase='product_repair'))
    result=replace(result,attempts=sum(t['result']['attempts'] for t in trace),
                   output_tokens=sum(t['result']['output_tokens'] for t in trace),latency_ms=int((time.perf_counter()-start)*1000))
    response.inference_ms=result.latency_ms
    return result,response,findings,review,trace


def generate_grounded(model,x,notes,extractor,output_filter,disease,bundle,*,include_evidence=True,disable_adapter=True):
    """Retain actual model output; explicitly label any source-card delivery.

    A verified source-card response is a fallback, not a native model success.
    RecResult and model trace always remain the actual last model output.
    """
    start=time.perf_counter()
    result,_,_,_,trace=_generate_draft(model,x,notes,extractor,output_filter,disease,bundle,
        include_evidence=include_evidence,disable_adapter=disable_adapter)
    response,findings,review=assess(result,x,notes,extractor,output_filter,disease,bundle)
    if (review['issues'] or response.quality_status!='ok') and include_evidence and bundle['claims']:
        cards=select_cards(bundle,response,requested_products(x))
        actual_input=model_input(x,disease)
        result=model.recommend_with_context(actual_input,recovery_prompt(cards,bundle),disable_adapter=disable_adapter)
        response,findings,review=assess(result,x,notes,extractor,output_filter,disease,bundle,
            recovery_ids=[c['id'] for c in cards])
        trace.append(dict(round=len(trace)+1,result=asdict(result),review=review,common_quality_issues=response.quality_issues,
            model_input=actual_input.to_dict(),boundary_version=VERSION,phase='source_contract_recovery',
            recovery_ids=[c['id'] for c in cards]))
        if review['issues'] or response.quality_status!='ok':
            response,findings,review=source_card_response(result,x,notes,extractor,output_filter,disease,bundle,
                [c['id'] for c in cards],review)
    result=replace(result,attempts=sum(t['result']['attempts'] for t in trace),
        output_tokens=sum(t['result']['output_tokens'] for t in trace),latency_ms=int((time.perf_counter()-start)*1000))
    response.inference_ms=result.latency_ms
    return result,response,findings,review,trace


def source_card_response(model_result,x,notes,extractor,output_filter,disease,bundle,ids,model_review):
    """Reconstruct only eligible source text and mark provenance before serving.

    Never pass an arbitrary model-supplied payload or ingredient into this path.
    A separately validated synthetic carrier feeds the common response builder;
    it is not recorded as a model call, model token or model answer.
    """
    from src.llm.inference import RecResult
    by_id={c['id']:c for c in bundle['claims']}
    if not ids or len(ids)>2 or len(set(ids))!=len(ids) or any(i not in by_id for i in ids):
        raise ValueError('Source delivery requires up to two unique eligible card IDs')
    cards=[by_id[i] for i in ids];payload=source_payload(cards,bundle)
    carrier=RecResult(payload,'source_cards_'+bundle['version'],False,0,json.dumps(payload,ensure_ascii=False),0,model_result.latency_ms)
    response,findings,review=assess(carrier,x,notes,extractor,output_filter,disease,bundle,recovery_ids=ids)
    response.generation_origin='source_cards'
    response.used_fallback=True
    for ingredient in response.ingredients:ingredient.purpose_source='source_card'
    response.source_recovery=dict(version='source_card_delivery_v2',source_ids=list(ids),source_store_sha256=bundle['store_sha256'],
        trigger_issues=list(model_review['issues']),attempted_model_version=model_result.model_version,
        attempted_model_used_rule_fallback=model_result.used_fallback,method='verified_source_card_template')
    review['source_recovery']['mode']='server_source_card_template'
    response.postprocess_version+='+source_card_delivery_v2'
    review['source_delivery']=dict(**response.source_recovery,payload=payload,
        origin='server_source_cards_not_model_text',model_review=model_review)
    # The common checks above evaluated the trusted card payload before the
    # explicit fallback flag was attached. Rule-generated or untrusted model
    # text still fails the unchanged evidence_rule_fallback check.
    return response,findings,review
