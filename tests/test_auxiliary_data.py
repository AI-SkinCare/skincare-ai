import copy
import json

import pytest

from src.llm.prepare_auxiliary_data import (dictionary_tasks, product_tasks,
    select_auxiliary, validate_training_auxiliary)

SOURCE = {'license':'CC-BY-4.0','repository':'fixture','revision':'fixed','attribution':'fixture'}


def test_blank_flags_stay_unknown_and_no_personal_efficacy_is_created():
    rows = dictionary_tasks([{'inci_name':'glycerin','function':'Humectant','fragrance':'0','eu_allergen':'','alcohol_type':''}],SOURCE)
    target = json.loads(rows[0]['target'])
    assert target['source_observations']['fragrance']=='0'
    assert target['source_observations']['eu_allergen'] is None
    assert target['personalized_efficacy']=='not_established'
    assert target['product_safety']=='not_established'
    assert 'input' not in rows[0]
    validate_training_auxiliary(rows)


def test_no_botanical_part_derivative_fuzzy_match_or_dental_sample():
    products = [{'product_id':'1','obf_categories_tags':'en:skin-creams','raw_ingredient_text':'GLYCERIN, Glycerin Oleate, Leaf Oil'},
                {'product_id':'2','obf_categories_tags':'en:dental-creams','raw_ingredient_text':'Glycerin'},
                {'product_id':'3','obf_categories_tags':'en:skin-creams','raw_ingredient_text':'Glycerin Oleate, Leaf Oil'}]
    rows, rejects = product_tasks(products, {'GLYCERIN':'글리세린','LEAFEXTRACT':'잎추출물'},dict(SOURCE,license='ODbL-1.0'))
    assert len(rows)==1
    assert json.loads(rows[0]['target'])['matched_reference_names']==['glycerin']
    assert rejects['not_selected_skincare_category']==1
    assert rejects['no_exact_project_alias']==1


def test_comma_inside_chemical_name_is_preserved():
    rows, _ = product_tasks([{'product_id':'1','obf_categories_tags':'en:skin-creams',
        'raw_ingredient_text':'Aqua, 1,2-Hexanediol, Glycerin'}],
        {'1,2-HEXANEDIOL':'1,2-HEXANEDIOL','GLYCERIN':'glycerin'},dict(SOURCE,license='ODbL-1.0'))
    assert json.loads(rows[0]['target'])['matched_reference_names']==['1,2-hexanediol','glycerin']


def test_sampling_is_bounded_reproducible_and_does_not_replicate():
    products = [{'id':f'p{i}'} for i in range(2)]
    dictionary = [{'id':f'd{i}'} for i in range(50)]
    a = select_auxiliary(products,dictionary,100,.05)
    assert a==select_auxiliary(products[::-1],dictionary[::-1],100,.05)
    assert len(a)==5 and len({r['id'] for r in a})==5
    assert len(select_auxiliary([],dictionary,1000,.1))==50


@pytest.mark.parametrize('change', [
    {'expert_efficacy_verified':True}, {'label_strength':'expert'}, {'split':'val_eval'},
    {'source':dict(SOURCE,license='unspecified')}, {'task_type':'recommendation_fabrication'}])
def test_auxiliary_cannot_be_silently_promoted_to_gold_or_unlicensed(change):
    row = dictionary_tasks([{'inci_name':'glycerin'}],SOURCE)[0]
    row.update(copy.deepcopy(change))
    with pytest.raises(ValueError):
        validate_training_auxiliary([row])
