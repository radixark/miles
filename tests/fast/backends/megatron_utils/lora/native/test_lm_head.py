"""Native lm_head adapters on every converter-named family, tied embeddings included — no GPU."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.fast.fixtures.args_fixtures import make_trainer_config

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.naming import ConverterNaming
from miles.backends.megatron_utils.lora.native.registry import resolve_adapter_targets
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.backends.megatron_utils.lora.native.spec.lm_head import LMHeadSpec
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

HIDDEN, VOCAB, PADDED_VOCAB = 8, 10, 12


class _OutputLayer(nn.Module):
    """MCore's output ColumnParallelLinear; a tied one owns no weight and is handed the embedding's."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x, weight=None, runtime_gather_output=None):
        return x @ (self.weight if weight is None else weight).t(), None


class _Model(nn.Module):
    def __init__(self, *, tied):
        super().__init__()
        self.post_process = True
        self.share_embeddings_and_output_weights = tied
        self.embedding = nn.Embedding(PADDED_VOCAB, HIDDEN)
        self.embedding.weight.requires_grad_(False)
        head = None if tied else nn.Parameter(torch.randn(PADDED_VOCAB, HIDDEN), requires_grad=False)
        self.output_layer = _OutputLayer(head)

    def shared_embedding_or_output_weight(self):
        return self.embedding.weight

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        logits, _ = self.output_layer(hidden, weight=weight)
        return logits


class _HeadNaming:
    def output_head_prefix(self, weight):
        return "lm_head."


def _context(targets):
    return AttachContext(
        lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=targets),
        transformer_config=SimpleNamespace(hidden_size=HIDDEN, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=_HeadNaming(),
    )


_ARGS = SimpleNamespace(backend=SimpleNamespace(vocab_size=VOCAB))


def _attached(*, tied, targets=("lm_head",)):
    model = _Model(tied=tied)
    assert LMHeadSpec().attach(model, _ARGS, _context(targets)) == 1
    nn.init.normal_(model.lora_lm_head_adapter.head_B)
    return model


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_the_adapter_changes_only_the_output_side(tied):
    """Tinker's unembed: the logits get s * B @ A, the input embedding stays the base one."""
    model = _attached(tied=tied)
    adapter = model.lora_lm_head_adapter
    tokens = torch.tensor([1, 4, 7])
    embedding_before = model.embedding.weight.clone()
    head = model.embedding.weight if tied else model.output_layer.weight
    hidden = model.embedding.weight[tokens]

    logits = model(tokens)
    logits.sum().backward()

    expected = hidden @ head.t() + 2.0 * hidden @ adapter.head_A.t() @ adapter.head_B.t()
    torch.testing.assert_close(logits, expected)
    assert torch.equal(model.embedding.weight, embedding_before) and model.embedding.weight.grad is None
    assert adapter.head_A.grad is not None and adapter.head_B.grad is not None


def test_the_vocab_sharded_b_counts_once_per_tp_rank():
    """head_B holds this rank's vocab rows, so Megatron's grad norm must count every rank's shard."""
    head_b = _attached(tied=True).lora_lm_head_adapter.head_B
    assert head_b.tensor_model_parallel and head_b.partition_dim == 0


def test_a_converter_naming_the_head_otherwise_fails():
    """Targets are checked against the spec's leaf; a differently named head would silently get no adapter."""
    context = _context(("lm_head",))
    context = dataclasses.replace(context, naming=SimpleNamespace(output_head_prefix=lambda weight: "head."))
    with pytest.raises(AssertionError, match="names the output head 'head'"):
        LMHeadSpec().attach(_Model(tied=True), _ARGS, context)


def test_an_unselected_head_gets_no_adapter():
    model = _Model(tied=True)
    assert LMHeadSpec().attach(model, _ARGS, _context(("q_proj",))) == 0
    assert not hasattr(model, "lora_lm_head_adapter")


def test_merging_a_tied_head_is_refused():
    """Merged into the shared weight, the delta would also move the input embedding."""
    model = _attached(tied=True)
    with pytest.raises(AssertionError, match="shares the input embedding's weight"):
        merge_lora_into_weights([model], {"embedding.word_embeddings.weight": model.embedding.weight})


def test_loading_the_served_vocab_zero_fills_megatrons_padding():
    """The export trims B to the HF vocab; loading it back pads Megatron's extra rows with zeros."""
    adapter = _attached(tied=True).lora_lm_head_adapter
    tensors = {"lm_head.lora_A.weight": torch.randn(2, HIDDEN), "lm_head.lora_B.weight": torch.randn(VOCAB, 2)}

    plan = dict((id(parameter), tensor) for parameter, tensor in adapter.load_plan(tensors.__getitem__))

    torch.testing.assert_close(plan[id(adapter.head_A)], tensors["lm_head.lora_A.weight"])
    head_b = plan[id(adapter.head_B)]
    assert head_b.shape == (PADDED_VOCAB, 2)
    torch.testing.assert_close(head_b[:VOCAB], tensors["lm_head.lora_B.weight"])
    assert not head_b[VOCAB:].any()
    with pytest.raises(AssertionError, match="has 9 rows, not 10"):
        adapter.load_plan({**tensors, "lm_head.lora_B.weight": torch.randn(VOCAB - 1, 2)}.__getitem__)


@pytest.mark.parametrize(
    "model_name,head",
    [
        ("qwen3_5config", "lm_head."),
        ("qwen3config", "lm_head."),
        ("kimi_k3config", "language_model.lm_head."),
        ("deepseekv41config", "head."),
    ],
)
def test_the_converter_names_the_head(model_name, head):
    args = make_trainer_config(hidden_size=HIDDEN, num_attention_heads=2, hf_checkpoint="/ckpt")
    naming = ConverterNaming(args, model_name, tp_size=2, is_hf_name=lambda name: True)
    weight = nn.Parameter(torch.empty(PADDED_VOCAB // 2, HIDDEN))
    weight.tensor_model_parallel, weight.partition_dim = True, 0
    assert naming.output_head_prefix(weight) == head


_QWEN3_5 = {
    "model_type": "qwen3_5",
    "text_config": {"layer_types": ["linear_attention", "full_attention"], "tie_word_embeddings": True},
}
_DEEPSEEK_V3 = {
    "model_type": "deepseek_v3",
    "q_lora_rank": 4,
    "first_k_dense_replace": 2,
    "num_hidden_layers": 2,
    "n_shared_experts": 1,
}


@pytest.mark.parametrize(
    "hf_config",
    [{"model_type": "qwen3"}, _QWEN3_5, _DEEPSEEK_V3],
    ids=["qwen3", "qwen3_5_tied", "deepseek_v3"],
)
def test_the_unembed_group_is_attachable(hf_config):
    hf_targets = resolve_hf_lora_targets(hf_config, target_modules=["attn", "mlp", "unembed"])
    served = resolve_adapter_targets(
        hf_config, hf_targets, hf_modules=[], lora_type="canonical_lora", experts_shared_outer_loras=True
    )
    assert "lm_head" in served


@pytest.mark.parametrize(
    "model_type,head", [("glm5_next", "lm_head"), ("deepseek_v41", "lm_head"), ("deepseek_v41", "head")]
)
def test_specs_without_a_head_adapter_still_refuse_it(model_type, head):
    with pytest.raises(AssertionError, match=rf"does not implement adapters for \['{head}'\]"):
        resolve_adapter_targets(
            {"model_type": model_type},
            [head],
            hf_modules=[],
            lora_type="canonical_lora",
            experts_shared_outer_loras=True,
        )
