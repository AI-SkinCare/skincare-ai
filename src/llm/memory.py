"""Opt-in storage reduction for frozen embeddings; normalization stays FP32."""
from __future__ import annotations


def compact_frozen_embeddings(model):
    import torch
    seen, saved = set(), 0
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is None:
            continue
        weight = module.weight
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        if weight.requires_grad:
            raise ValueError('only frozen embeddings may use compact storage')
        if weight.__class__.__name__=='Params4bit':
            raise ValueError('do not reinterpret quantized weights')
        if weight.dtype==torch.float32:
            saved += weight.numel()*2
            weight.data = weight.data.to(torch.bfloat16)
    return {'frozen_embedding_dtype':'bfloat16', 'estimated_storage_saved_mib':round(saved/1024**2,1)}
