"""Small, attributed auxiliary tasks; never fabricate personal efficacy labels."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from .prepare_quality_data import ROOT, digest, read_rows

INCIDB_REPO = 'Ichlibitiche/incidb-skincare-free-sample'
INCIDB_REV = 'b866a0f3fa3355a360ee6f6597af717655b40acd'
CURESKIN_REV = 'e34203b35153fd1cd9de6de7c938165c722c003f'
DEFAULT_OUT = ROOT / 'data/auxiliary_20261008_v2'


def name_key(name):
    return re.sub(r'\s+', '', name).casefold().strip(' .')


def make_row(task, key, messages, target, provenance):
    return {'id': 'AUX_' + task + '_' + digest(key)[:20], 'task_type': task,
            'messages': messages, 'target': json.dumps(target, ensure_ascii=False),
            'label_strength': 'source_observation_only', 'source': provenance,
            'expert_efficacy_verified': False, 'split': 'train_auxiliary'}


def product_tasks(products, lexicon, provenance):
    # Exact English identity only; an alias must map to exactly one representative.
    aliases = {name_key(a): canonical for a, canonical in lexicon.items() if re.fullmatch(r'[A-Za-z0-9 /().,+\-]+', a)}
    candidates = sorted(set(aliases))
    rows, rejected, seen = [], collections.Counter(), set()
    for product in products:
        categories = product.get('obf_categories_tags', '').casefold()
        if (re.search(r'dent|tooth|hair|shampoo|nail|ongle|parfum|make.?up|maquillage', categories)
                or not re.search(r'skin|face|visage|cream|cr[eè]me|lotion|s[eé]rum|moist|solaire|body|corps|hand|mains|mask|masque|hydrat|savon|soap', categories)):
            rejected['not_selected_skincare_category'] += 1
            continue
        text = product.get('raw_ingredient_text', '').strip()
        key = name_key(text)
        if not text or len(text) > 2400 or key in seen:
            rejected['empty_long_or_duplicate_declaration'] += 1
            continue
        # Whole declaration entries only, preserving commas inside names such as 1,2-hexanediol.
        compact = re.sub(r'\s+', '', text).casefold()
        present = sorted(a for a in candidates if re.search(r'(?:^|[,;])' + re.escape(a) + r'\.?\s*(?=[,;]|$)', compact))
        if not present:
            rejected['no_exact_project_alias'] += 1
            continue
        seen.add(key)
        # Include deterministic absent candidates; avoid a huge inventory in every prompt.
        absent = [a for a in candidates if a not in present]
        random.Random(int(digest(key)[:16],16)).shuffle(absent)
        references = sorted(present + absent[:8])
        messages = [{'role': 'system', 'content': '당신은 화장품 전성분 원문을 읽는 도우미입니다. 제공한 목록의 영문 이름을 공백·대소문자 차이만 무시하고 정확히 식별하세요. 비슷한 식물·부위·유도체를 합치지 마세요. 목록 외 성분은 무시하세요. 성분 표기는 효능·농도·사용 안전성 근거가 아닙니다. JSON으로 답하세요.'},
                    {'role': 'user', 'content': '참조 목록: ' + ', '.join(references) + '\n전성분 원문: ' + text}]
        target = {'matched_reference_names': present,
                  'efficacy_evidence': 'not_provided', 'concentrations': 'not_provided'}
        rows.append(make_row('label_identity', key, messages, target,
                             {**provenance, 'source_record_id': product['product_id']}))
    return rows, dict(rejected)


def dictionary_tasks(entries, provenance):
    rows, seen = [], set()
    fields = ('function', 'fragrance', 'eu_allergen', 'alcohol_type')
    for entry in entries:
        key = name_key(entry['inci_name'])
        if not key or key in seen:
            continue
        seen.add(key)
        observations = {f: entry.get(f, '').strip() or None for f in fields}
        # 0 is the source's literal negative flag; NULL is unassessed, neither means safe.
        target = {'inci_name': entry['inci_name'], 'source_observations': observations,
                  'source': 'CureSkin v2026.07', 'personalized_efficacy': 'not_established',
                  'product_safety': 'not_established'}
        messages = [{'role': 'system', 'content': '제공된 성분 사전 행의 관측값만 JSON으로 정리하세요. 비어 있는 값은 null(미평가)로 쓰세요. 0과 미평가를 구분하고 제품의 안전성이나 개인의 고민에 대한 효능을 단정하지 마세요. 사전 기능 설명은 출처의 표현으로만 반환하세요.'},
                    {'role': 'user', 'content': '출처: CureSkin Indian Skincare Ingredient Flags v2026.07\n' + json.dumps({'inci_name': entry['inci_name'], **{f: entry.get(f, '') for f in fields}}, ensure_ascii=False)}]
        rows.append(make_row('dictionary_observation', key, messages, target,
                             {**provenance, 'source_record_id': entry['inci_name']}))
    return rows


def select_auxiliary(products, dictionary, main_count, ratio=.05, seed=42):
    if not 0 < ratio <= .1:
        raise ValueError('auxiliary count/main count ratio must be within (0, .1]')
    cap = int(main_count * ratio)
    rng = random.Random(seed)
    products, dictionary = sorted(products, key=lambda r:r['id']), sorted(dictionary, key=lambda r:r['id'])
    rng.shuffle(products)
    rng.shuffle(dictionary)
    # Product diversity is finite; never replicate rows to fill the quota.
    chosen_products = products[:cap // 2]
    chosen = chosen_products + dictionary[:cap-len(chosen_products)]
    rng.shuffle(chosen)
    return chosen


def validate_training_auxiliary(rows):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate auxiliary IDs')
    seen = set()
    for row in rows:
        if row.get('task_type') not in ('label_identity', 'dictionary_observation'):
            raise ValueError('unsupported auxiliary task')
        if row.get('expert_efficacy_verified') is not False or row.get('label_strength') != 'source_observation_only':
            raise ValueError('auxiliary observations must not become expert efficacy labels')
        if row.get('split') != 'train_auxiliary' or row['source']['license'] not in ('ODbL-1.0', 'CC-BY-4.0'):
            raise ValueError('unapproved split or missing license')
        if [m['role'] for m in row['messages']] != ['system', 'user']:
            raise ValueError('auxiliary must contain only system and user prompts')
        json.loads(row['target'])
        key = digest(row['messages'])
        if key in seen:
            raise ValueError('duplicate auxiliary prompt')
        seen.add(key)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    base = ROOT / 'data/processed/sllm_v3'
    inc = ROOT / 'data/external_20261008/incidb' / INCIDB_REV
    cure = ROOT / 'data/review_20261006/external/cureskin' / CURESKIN_REV
    if 'Open Database License' not in (inc/'README.md').read_text(encoding='utf-8'):
        raise ValueError('INCIDB attribution/license declaration missing')
    if 'CC BY 4.0' not in (cure/'LICENSE').read_text(encoding='utf-8'):
        raise ValueError('CureSkin license declaration missing')
    lex = json.loads((base/'ingredient_dict.json').read_text(encoding='utf-8'))['lexicon']
    def csv_rows(path, delimiter=','):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    inc_source = {'repository': INCIDB_REPO, 'revision': INCIDB_REV, 'license': 'ODbL-1.0',
                  'attribution': 'Open Beauty Facts contributors; INCIDB by Ichlibitiche',
                  'url': f'https://huggingface.co/datasets/{INCIDB_REPO}/tree/{INCIDB_REV}',
                  'license_url': 'https://opendatacommons.org/licenses/odbl/1-0/',
                  'changes': 'skincare category selection; exact name lookup; task conversion'}
    cure_source = {'repository': 'cureskin/indian-skincare-ingredient-flags', 'revision': CURESKIN_REV,
                   'license': 'CC-BY-4.0', 'attribution': 'CureSkin, Indian Skincare Ingredient Flags v2026.07',
                   'url': f'https://huggingface.co/datasets/cureskin/indian-skincare-ingredient-flags/tree/{CURESKIN_REV}',
                   'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                   'changes': 'source-observation task conversion; blank values to null'}
    # Extend identity coverage from source-declared exact matches, never fuzzy/synonym guesses.
    identity_lex = {r['canonical_name']: r['canonical_name'] for r in csv_rows(inc/'ingredient_name_map.csv','|')
                    if r['method']=='exact' and float(r['confidence'])==1.0}
    products, rejects = product_tasks(csv_rows(inc/'products.csv', '|'), identity_lex, inc_source)
    relevant = {name_key(a) for a in lex if a.isascii()}
    relevant.update(name for row in products for name in json.loads(row['target'])['matched_reference_names'])
    entries = csv_rows(cure/'ingredient-flags.csv')
    dictionary = dictionary_tasks([r for r in entries if name_key(r['inci_name']) in relevant], cure_source)
    main_rows = read_rows(base/'train.jsonl')
    chosen = select_auxiliary(products, dictionary, len(main_rows))
    validate_training_auxiliary(chosen)
    source_paths = [base/'manifest.json', base/'train.jsonl', base/'ingredient_dict.json',
                    inc/'products.csv', inc/'ingredient_name_map.csv', inc/'README.md', cure/'ingredient-flags.csv', cure/'LICENSE', cure/'README.md']
    manifest = {'version': 'auxiliary_observations_v2', 'recommendation_dataset_id': json.loads((base/'manifest.json').read_text(encoding='utf-8'))['dataset_id'],
                'sources': [inc_source, cure_source], 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'rows_hash': digest(chosen), 'rows': len(chosen), 'seed':42, 'maximum_count_ratio_to_main':.05,
                'task_counts': dict(collections.Counter(r['task_type'] for r in chosen)),
                'candidate_counts': {'products':len(products),'dictionary':len(dictionary)}, 'rejected_products':rejects,
                'dictionary_relevance_excluded':len(entries)-len(dictionary),
                'label_policy':'observations only; no synthesized profiles or personalized efficacy',
                'evaluation_policy':'unchanged v3 development split; auxiliary rows never enter recommendation evaluation',
                'distribution':'local private training; keep ODbL-derived auxiliary database separately attributed; AI Hub records not redistributed',
                'limitations':['finite biased product sample', 'dictionary flags are not clinical efficacy', 'no new rare/multiple-concern gold labels', 'expert efficacy review pending']}
    manifest['dataset_id'] = digest(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('existing auxiliary dataset differs; use new --out')
    (args.out/'train_auxiliary.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen),encoding='utf-8')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'ATTRIBUTION.md').write_text('\n\n'.join(f"{s['attribution']}\n{s['url']}\nLicense: {s['license']} {s['license_url']}\nChanges: {s['changes']}" for s in manifest['sources'])+'\n\nOriginal recommendation data remains a separate, non-redistributed AI Hub dataset. Auxiliary source observations are not expert efficacy labels.\n',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('dataset_id','rows','task_counts','candidate_counts','rejected_products')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
