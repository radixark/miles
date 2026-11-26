"""The shared-prefix adapter selects capabilities, not architecture names."""

import copy

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
)

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.transformers import TreeSFTBatch, enable_transformers_prefix_tree


def _model(family, *, linear_rope=False):
    options = dict(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        sliding_window=None,
    )
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
        "gpt_neox": (GPTNeoXConfig, GPTNeoXForCausalLM),
    }[family]
    if linear_rope:
        options["rope_parameters"] = {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    config = config_type(**options)
    config._attn_implementation = "eager"
    return model_type(config).train()


@pytest.mark.parametrize("family", ["llama", "mistral", "gpt_neox"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_loss_every_gradient_and_adamw_step_across_architectures(family, checkpoint):
    torch.manual_seed(19)
    expanded = _model(family)
    compact = copy.deepcopy(expanded)
    if checkpoint:
        for model in (expanded, compact):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enable_transformers_prefix_tree(compact)
    # Includes nested forks, a duplicate trajectory and a sequence ending at a fork.
    sequences = [[1, 2, 3, 4, 5], [1, 2, 3, 7], [1, 2], [1, 2, 3, 4, 5], [8, 9, 10]]
    targets = [[2, 3, 4, 5, 6], [-100, -100, 7, 11], [2, 15], [2, -100, 4, 5, 12], [9, 10, -100]]
    weights = [[1.0 + sample / 4] * len(seq) for sample, seq in enumerate(sequences)]
    batch = TreeSFTBatch.from_plan(
        build_prefix_tree(sequences), targets, weights=weights, device="cpu", deterministic_reduction=True
    )
    actual = batch.loss(compact)
    terms, denominator = [], 0.0
    for sequence, labels, weight in zip(sequences, targets, weights, strict=True):
        logits = expanded(torch.tensor([sequence]), use_cache=False).logits[0]
        terms.append((F.cross_entropy(logits, torch.tensor(labels), reduction="none") * torch.tensor(weight)).sum())
        denominator += sum(w for target, w in zip(labels, weight, strict=True) if target != -100)
    expected = torch.stack(terms).sum() / denominator
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    actual.backward()
    expected.backward()
    for (name, a), (_, b) in zip(compact.named_parameters(), expanded.named_parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=3e-5, msg=lambda m, n=name: f"{n}: {m}")
    for model in (compact, expanded):
        torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-6).step()
    for a, b in zip(compact.parameters(), expanded.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=3e-5)


def test_logical_positions_with_static_scaled_rope():
    torch.manual_seed(19)
    expanded = _model("llama", linear_rope=True)
    compact = copy.deepcopy(expanded)
    enable_transformers_prefix_tree(compact)
    sequences = [[1, 2, 3], [1, 2, 4, 5]]
    targets = [[2, 3, -100], [2, 4, 5, -100]]
    actual = TreeSFTBatch.from_plan(build_prefix_tree(sequences), targets, device="cpu").loss(compact)
    expected = (
        sum(
            F.cross_entropy(expanded(torch.tensor([s]), use_cache=False).logits[0], torch.tensor(t), reduction="sum")
            for s, t in zip(sequences, targets, strict=True)
        )
        / 5
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    for a, b in zip(compact.parameters(), expanded.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=3e-5)


@pytest.mark.parametrize("capability", ["backend", "dropout", "sliding", "hybrid", "dynamic_rope"])
def test_unsupported_semantics_fail_before_execution(capability):
    model = _model("llama")
    if capability == "backend":
        model._supports_attention_backend = False
    elif capability == "dropout":
        model.config.attention_dropout = 0.1
    elif capability == "sliding":
        model.config.sliding_window = 32
    elif capability == "hybrid":
        model.config.layer_types = ["full_attention", "linear_attention"]
    else:
        model.config.rope_parameters = {"rope_type": "dynamic", "factor": 2.0}
    with pytest.raises(ValueError):
        enable_transformers_prefix_tree(model)


@pytest.mark.parametrize("modifier", ["softcap", "alibi", "head_mask", "sinks", "position_bias"])
def test_attention_modifiers_are_not_silently_ignored(modifier):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = _model("llama")
    enable_transformers_prefix_tree(model)
    module = model.model.layers[0].self_attn
    function = ALL_ATTENTION_FUNCTIONS["miles_prefix_tree"]
    tensor = torch.zeros(1, 2, 3, 8)
    layout = TreeSFTBatch.from_plan(build_prefix_tree([[1, 2, 3]]), [[2, 3, -100]], device="cpu").attention
    with pytest.raises(ValueError, match="score modifiers"):
        function(module, tensor, tensor, tensor, None, scaling=1.0, prefix_tree_layout=layout, **{modifier: 1.0})
