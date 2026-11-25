"""Prediction-keyed, class-level reference. Never retrieve per-image gold labels."""
from __future__ import annotations
import json
from pathlib import Path


def load_medical_knowledge(path):
    path=Path(path)
    if not path.exists():return {'labels':{}}
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value.get('labels'),dict) or not isinstance(value.get('source'),str):
        raise ValueError('invalid medical knowledge file')
    for key,item in value['labels'].items():
        if key not in ('psoriasis','atopic','acne','rosacea','seborrheic'):
            raise ValueError('unknown medical reference label')
        text=item.get('general_description')
        if not isinstance(text,str) or not text.strip() or len(text)>1000:
            raise ValueError('invalid class description')
    return value


def medical_reference(prediction,knowledge,threshold):
    if prediction.uncertain or prediction.confidence<threshold:return None
    item=knowledge['labels'].get(prediction.label_en)
    if item is None:return None
    return {'label_en':prediction.label_en,'general_description':item['general_description'],
            'source':knowledge['source'],'scope':'class_definition_not_patient_observation',
            'severity_assessed':False,'lesion_location_assessed':False,'ingredient_efficacy_verified':False}


def reference_prompt(reference):
    return ('\n질환 일반 설명 참고(JSON): '+json.dumps(reference,ensure_ascii=False)+
            '\n이 설명은 예측 클래스의 일반적 정의이며 현재 사용자에게 해당 증상·부위가 관찰됐다는 뜻이 아닙니다. '
            '발병 시기, 분포, 실제 성별·연령, 원인, 중증도와 EASI 점수는 추정하지 마세요. '
            '이 데이터는 질환별 성분 효능이나 치료·처방의 정답을 제공하지 않습니다.')
