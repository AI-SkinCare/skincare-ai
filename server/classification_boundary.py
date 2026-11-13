"""Keep classifier metadata out of patient-state prose in grounded generation.

This is an explicit output contract plus narrow text checks, not a medical
diagnosis or a general semantic verifier. No output is rewritten into success.
"""
from __future__ import annotations

from dataclasses import replace
import re

VERSION = 'classification_boundary_v2'
DISCLAIMER = '사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다.'
GUIDANCE = """The class label is only a retrieval/routing hint. It is NOT a patient finding. Routing concern words may be automatically mapped from that label and are NOT proof of reported symptoms. Numerical classifier confidence is deliberately withheld from this generation input. Never guess, repeat, or turn classification scores, likelihoods or confidence into personal severity, disease probability or certainty. This also applies if a user note asks you to interpret a score. Do not infer that skin is healthy from a normal class. Step 1 must contain ONLY this exact sentence: 사진 분류와 설문만으로 개인의 상태를 확정할 수 없습니다. Keep steps 2/3 and the answer within the supplied ingredient claims and product conditions. Do not discuss the classifier score anywhere in the generated text; the API reports model metadata separately."""


def model_input(x, disease):
    """Drop only the exact server-generated score prefix; preserve survey data.

    Original classification JSON, threshold and input object remain unchanged.
    Numbers supplied by a user are not silently altered or removed.
    """
    if disease is None:
        return replace(x)
    prefix = f'이미지 분류 참고(진단 아님): {disease.label}, 분류 점수 {disease.confidence:.3f}.'
    note = x.user_note or ''
    if note.startswith(prefix):
        note = note[len(prefix):]
        note = note.removeprefix(' 이 점수는 질환이 있을 임상적 확률이 아닙니다.')
    note = (f'이미지 분류 모델의 참고 라벨: {disease.label}. 이 라벨은 성분 근거 조회용이며 개인의 상태나 중증도를 관찰한 사실이 아닙니다. '
            '주요 고민에는 라벨에서 자동 변환한 추천용 어휘가 포함될 수 있으므로 증상 보고로 단정하지 마세요.\n' + note.lstrip())
    return replace(x, user_note=note)


_NUMERIC_CLASS_SCORE = re.compile(
    r'(?:분류|예측|모델|신뢰도|확신|확률|위험도)[^!?\n]{0,48}?(?:(?<!\d)0\.\d+(?!\d)|\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*퍼센트)', re.I)
_STATE = re.compile(
    r'(?:여드름성|아토피(?:성)?|건선|주사|지루(?:성)?|피부염|질환)'
    r'(?:(?:가능성|확률|위험|정도|상태|의|이|가|은|는|성|\s)){0,14}'
    r'(?:높|심하|심한|심합|경증|중등도|중증)'
    r'|(?:정상(?:으로)?\s*분류|정상\s*판정)[^.!?\n]{0,30}(?:이상(?:이)?\s*없|질환(?:이)?\s*없|문제(?:가)?\s*없|건강)'
    r'|(?:확진|진단)(?:입니다|되었|됐|이므로)')
_DENIAL = re.compile(r'(?:볼|판단할|확정할|알|해석할)\s*수\s*없|단정하지|아닙|아니|못합')


def review_boundary(parsed):
    issues = []
    steps = parsed.get('reasoning', [])
    if not steps or steps[0].get('content', '').strip() != DISCLAIMER:
        issues.append('classification_boundary_instruction_missing')
    text = '\n'.join([s.get('content', '') for s in steps] + [parsed.get('answer', '')])
    if _NUMERIC_CLASS_SCORE.search(text):
        issues.append('classification_score_in_generated_text')
    clauses = re.split(r'(?<!\d)[.!?]|[.!?](?!\d)|\n|지만|그러나|다만', text)
    if any(_STATE.search(c) and not _DENIAL.search(c) for c in clauses):
        issues.append('classification_as_patient_state')
    return list(dict.fromkeys(issues))
