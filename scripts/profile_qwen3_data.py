"""CPU-only audit and token accounting for the two Qwen3 training cohorts."""
from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.audit_quality_data import audit
from src.llm.prepare_auxiliary_data import DEFAULT_OUT, validate_training_auxiliary
from src.llm.prepare_quality_data import digest, read_rows
from src.llm.train_qlora import encode


def main():
    from transformers import AutoTokenizer
    proc = ROOT / 'data/processed/sllm_v3'
    report = {'integrity': audit(proc, ROOT/'data/processed/sllm'), 'cohorts': {}}
    rows = read_rows(proc/'train.jsonl')
    auxiliary = read_rows(DEFAULT_OUT/'train_auxiliary.jsonl')
    validate_training_auxiliary(auxiliary)
    for base in ('Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-8B'):
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        included, excluded, lengths = [], [], []
        for row in rows:
            encoded = encode(row, tok, 2048)
            (included if encoded else excluded).append(row['id'])
            if encoded:
                lengths.append(len(encoded['input_ids']))
        kept_aux = [r['id'] for r in auxiliary if encode(r, tok, 2048)]
        lengths.sort()
        report['cohorts'][base] = {'included': len(included), 'excluded':len(excluded),
                                 'included_ids_hash':digest(sorted(included)), 'excluded_ids':excluded,
                                 'excluded_by_concern':dict(collections.Counter(r['target_concern'] for r in rows if r['id'] in set(excluded))),
                                 'p50':lengths[len(lengths)//2], 'p95':lengths[int(len(lengths)*.95)], 'max':max(lengths),
                                 'auxiliary_included':len(kept_aux), 'auxiliary_excluded_ids':sorted(set(r['id'] for r in auxiliary)-set(kept_aux))}
    Path(sys.argv[1]).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
