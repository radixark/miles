"""Independent expanded Qwen oracle: ordinary causal masks, never tree attention."""

import copy

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.qwen import TreeSFTBatch, enable_qwen_prefix_tree


def _model(dtype, *, fp64_norm=False):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    # Eager Qwen also explicitly casts softmax to FP32; SDPA preserves FP64.
    config._attn_implementation = "sdpa" if fp64_norm else "eager"
    model = Qwen3ForCausalLM(config).to(dtype=dtype).train()
    if fp64_norm:
        _replace_norms_for_fp64_oracle(model)
    return model


def _replace_norms_for_fp64_oracle(module):
    """Test-only oracle: keep RMSNorm algebra in FP64 instead of native Qwen's FP32."""
    for name, child in module.named_children():
        if child.__class__.__name__ == "Qwen3RMSNorm":
            replacement = torch.nn.RMSNorm(
                child.weight.numel(),
                eps=child.variance_epsilon,
                dtype=torch.float64,
            )
            replacement.weight.data.copy_(child.weight.data)
            setattr(module, name, replacement)
        else:
            _replace_norms_for_fp64_oracle(child)


# Native Qwen RMSNorm computes internally in FP32 even for FP64 parameters.
# Its backward rounds before/after prefix fan-in differently; test the true
# FP64 attention derivative separately at 1e-12, rather than claiming bit equality.
@pytest.mark.parametrize(
    "dtype,fp64_norm,atol,rtol",
    [(torch.float64, True, 1e-12, 1e-12), (torch.float64, False, 1e-7, 2e-6), (torch.float32, False, 2e-7, 2e-5)],
)
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("deterministic_reduction", [False, True])
def test_qwen_sft_loss_every_gradient_and_optimizer_step(
    dtype, fp64_norm, atol, rtol, checkpoint, deterministic_reduction
):
    sequences = [[1, 2, 3, 4, 5], [1, 2, 3, 7], [1, 2], [1, 2, 3, 4, 5], [8, 9, 10]]
    targets = [[2, 3, 4, 5, 6], [-100, -100, 7, 11], [2, 15], [2, -100, 4, 5, 12], [9, 10, -100]]
    weights = [[1.0 + sample / 4] * len(seq) for sample, seq in enumerate(sequences)]
    expanded = _model(dtype, fp64_norm=fp64_norm)
    compact = copy.deepcopy(expanded)
    if checkpoint:
        compact.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        expanded.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enable_qwen_prefix_tree(compact)
    batch = TreeSFTBatch.from_plan(
        build_prefix_tree(sequences),
        targets,
        weights=weights,
        device="cpu",
        deterministic_reduction=deterministic_reduction,
    )
    compact_loss = batch.loss(compact)
    terms, total_weight = [], 0.0
    for seq, labels, weight in zip(sequences, targets, weights, strict=True):
        logits = expanded(torch.tensor([seq]), use_cache=False).logits[0]
        labels_tensor = torch.tensor(labels)
        weight_tensor = torch.tensor(weight, dtype=dtype)
        terms.append((F.cross_entropy(logits, labels_tensor, reduction="none") * weight_tensor).sum())
        total_weight += sum(w for label, w in zip(labels, weight, strict=True) if label != -100)
    expanded_loss = torch.stack(terms).sum() / total_weight
    torch.testing.assert_close(compact_loss, expanded_loss, atol=atol, rtol=rtol)
    compact_loss.backward()
    expanded_loss.backward()
    for (name, actual), (_, expected) in zip(compact.named_parameters(), expanded.named_parameters(), strict=True):
        assert actual.grad is not None, name
        torch.testing.assert_close(
            actual.grad, expected.grad, atol=atol, rtol=rtol, msg=lambda msg, name=name: f"{name}: {msg}"
        )
    for model in (compact, expanded):
        torch.optim.SGD(model.parameters(), lr=0.05).step()
    for actual, expected in zip(compact.parameters(), expanded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_reject_dropout_before_execution():
    model = _model(torch.float32)
    model.config.attention_dropout = 0.1
    with pytest.raises(ValueError, match="dropout"):
        enable_qwen_prefix_tree(model)


@pytest.mark.parametrize("return_statistics", [False, True])
def test_fp32_master_weights_cast_qk_norm_outputs_for_fused_attention(return_statistics):
    model = _model(torch.float32)
    enable_qwen_prefix_tree(model)
    batch = TreeSFTBatch.from_plan(
        build_prefix_tree([[1, 2, 3], [1, 2, 4]]), [[2, 3, -100], [2, 4, -100]], device="cpu"
    )
    calls = []

    def fake_varlen(q, k, v, **kwargs):
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        calls.append(q.dtype)
        output = q + k.repeat_interleave(2, 1)[: q.shape[0]] * 0 + v.repeat_interleave(2, 1)[: q.shape[0]] * 0
        return (output, None) if return_statistics else output

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = batch.loss(model, varlen_func=fake_varlen)
    loss.backward()
    assert len(calls) == 2


def test_long_non_aligned_prefix_fanout_and_adamw_update_in_fp64():
    generator = torch.Generator().manual_seed(29)
    prefix = torch.randint(0, 37, (129,), generator=generator).tolist()
    sequences = [prefix + torch.randint(0, 37, (7 + branch,), generator=generator).tolist() for branch in range(8)]
    targets = [[-100] * 128 + sequence[129:] + [-100] for sequence in sequences]
    expanded = _model(torch.float64, fp64_norm=True)
    compact = copy.deepcopy(expanded)
    enable_qwen_prefix_tree(compact)
    plan = build_prefix_tree(sequences)
    assert len(plan.tokens) <= 129 + sum(len(sequence) - 129 for sequence in sequences)
    batch = TreeSFTBatch.from_plan(plan, targets, device="cpu")
    compact_loss = batch.loss(compact)
    losses = []
    for sequence, labels in zip(sequences, targets, strict=True):
        logits = expanded(torch.tensor([sequence]), use_cache=False).logits[0]
        losses.append(F.cross_entropy(logits, torch.tensor(labels), reduction="sum"))
    denominator = sum(sum(label != -100 for label in labels) for labels in targets)
    expanded_loss = torch.stack(losses).sum() / denominator
    torch.testing.assert_close(compact_loss, expanded_loss, atol=1e-12, rtol=1e-12)
    compact_loss.backward()
    expanded_loss.backward()
    for actual, expected in zip(compact.parameters(), expanded.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-12)
    for model in (compact, expanded):
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    for actual, expected in zip(compact.parameters(), expanded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
