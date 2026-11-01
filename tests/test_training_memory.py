import pytest
torch = pytest.importorskip('torch')

from src.llm.memory import compact_frozen_embeddings


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32,8)
        self.head = torch.nn.Linear(8,32,bias=False)
        self.norm = torch.nn.LayerNorm(8)
        for p in self.parameters():
            p.requires_grad=False
        self.embed.weight.data = self.embed.weight.data.to(torch.bfloat16).float()
        self.head.weight.data = self.head.weight.data.to(torch.bfloat16).float()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def test_bf16_source_values_are_preserved_and_norm_stays_fp32():
    model = FrozenModel()
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    info = compact_frozen_embeddings(model)
    for n,p in model.named_parameters():
        assert not p.requires_grad
        assert torch.equal(p.float(),before[n])
    assert model.embed.weight.dtype==model.head.weight.dtype==torch.bfloat16
    assert model.norm.weight.dtype==torch.float32
    assert info['frozen_embedding_dtype']=='bfloat16'


def test_trainable_head_is_rejected():
    model=FrozenModel()
    model.head.weight.requires_grad=True
    with pytest.raises(ValueError,match='frozen'):
        compact_frozen_embeddings(model)
