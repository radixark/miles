"""Integration coverage for true-on-policy scoring in Megatron BI mode."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])


import megatron.core.transformer.custom_layers.batch_invariant_kernels as batch_invariant_kernels
import torch

from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy


def test_true_on_policy_scorer_is_batch_invariant(monkeypatch):
    """The same BF16 row must score identically under different row groupings."""
    calls = 0
    original_log_softmax = batch_invariant_kernels.log_softmax

    def tracked_log_softmax(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_log_softmax(*args, **kwargs)

    monkeypatch.setattr(batch_invariant_kernels, "log_softmax", tracked_log_softmax)

    generator = torch.Generator(device="cuda").manual_seed(42)
    vocab_size = 129280
    batched_logits = torch.randn(
        17,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    token_ids = torch.randint(
        vocab_size,
        (17,),
        generator=generator,
        device="cuda",
    )

    with batch_invariant_kernels.set_batch_invariant_mode():
        assert batch_invariant_kernels.is_batch_invariant_mode_enabled()
        single_logprob, _ = calculate_log_probs_and_entropy(
            batched_logits[:1].clone(),
            token_ids[:1],
            None,
            true_on_policy=True,
        )
        batched_logprobs, _ = calculate_log_probs_and_entropy(
            batched_logits,
            token_ids,
            None,
            true_on_policy=True,
        )

    assert calls == 2
    assert single_logprob.dtype == torch.bfloat16
    assert torch.equal(single_logprob, batched_logprobs[:1])
