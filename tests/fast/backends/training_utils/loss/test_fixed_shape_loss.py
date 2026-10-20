"""The fixed-shape (TPU) log-prob and policy-loss path agrees with the per-sample path.

`loss/hub/fixed_shape.py` scores the padded microbatch instead of per-sample slices. On CPU both paths
run directly, so these tests feed them the same packed samples (the fixed-shape side gets extra padding
rows of garbage logits) and compare log-probs, entropy, loss, metrics and gradients.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.fixed_shape import (
    fixed_shape_policy_loss_function,
    get_fixed_shape_log_probs_and_entropy,
)
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.losses import policy_loss_function

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state

PROMPT_LENS = [7, 30, 3]
RESPONSE_LENS = [25, 2, 40]
VOCAB = 64
PADDED_ROWS = 128  # > sum(PROMPT_LENS + RESPONSE_LENS) = 107


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("fixed-shape-loss") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _args(**overrides):
    defaults = dict(
        true_on_policy_mode=False,
        observe_training_entropy=False,
        log_probs_chunk_size=32,
        eps_clip_c=None,
    )
    return make_args(**{**defaults, **overrides})


def _padded(logits: torch.Tensor) -> torch.Tensor:
    """The packed `[1, N, V]` logits followed by garbage rows up to PADDED_ROWS."""
    pad = torch.full((1, PADDED_ROWS - logits.size(1), logits.size(2)), 1e4)
    return torch.cat([logits, pad], dim=1)


def _inputs(args, seed: int = 0):
    make_parallel_state()
    inputs = make_inputs(
        seed=seed,
        batch_size=len(PROMPT_LENS),
        prompt_lens=PROMPT_LENS,
        response_lens=RESPONSE_LENS,
        vocab_size=VOCAB,
        args=args,
    )
    # Partial loss masks exercise the per-sample denominators.
    inputs["loss_masks"][0][::3] = 0
    # rows outside the responses score token 0 at log-prob ~ -200, which overflows an unmasked k3 KL
    response_rows = set()
    start = 0
    for total, response in zip(inputs["total_lens"], inputs["response_lens"], strict=True):
        response_rows.update(range(start + total - response - 1, start + total - 1))
        start += total
    for row in set(range(start)) - response_rows:
        inputs["policy_logits"][0, row, 0] = -200.0
    return inputs


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_forward_log_probs_and_entropy_match(temperature):
    args = _args(rollout_temperature=temperature)
    inputs = _inputs(args)
    common = dict(
        args=args,
        unconcat_tokens=inputs["unconcat_tokens"],
        total_lengths=inputs["total_lens"],
        response_lengths=inputs["response_lens"],
        with_entropy=True,
    )
    expected = get_log_probs_and_entropy(inputs["policy_logits"].clone(), **common)
    actual = get_fixed_shape_log_probs_and_entropy(_padded(inputs["policy_logits"]), **common)
    for key in ("log_probs", "entropy"):
        assert len(actual[key]) == len(expected[key])
        for got, want in zip(actual[key], expected[key], strict=True):
            torch.testing.assert_close(got.reshape(-1), want.reshape(-1).cpu(), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(entropy_coef=0.0),
        dict(entropy_coef=0.0, observe_training_entropy=True),
        dict(use_kl_loss=True, kl_loss_coef=0.05, kl_loss_type="low_var_kl"),
        dict(use_kl_loss=True, kl_loss_coef=0.05, use_unbiased_kl=True),
        dict(use_kl_loss=True, kl_loss_coef=0.05, kl_loss_type="k3"),
        dict(use_rollout_logprobs=True),
        dict(calculate_per_token_loss=True),
        dict(rollout_temperature=0.7),
    ],
    ids=[
        "default",
        "no-entropy",
        "observe-entropy",
        "kl",
        "unbiased-kl",
        "k3-kl",
        "rollout-logprobs",
        "per-token",
        "temp",
    ],
)
def test_policy_loss_matches(overrides):
    args = _args(**overrides)
    inputs = _inputs(args)

    logits = inputs["policy_logits"].clone().requires_grad_(True)
    batch = make_batch(inputs, "policy_loss")
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        None,
    )
    want_loss, want_metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    want_loss.backward()

    padded = _padded(inputs["policy_logits"]).requires_grad_(True)
    got_loss, got_metrics = fixed_shape_policy_loss_function(
        args, make_batch(inputs, "policy_loss"), padded, sum_of_sample_mean=None
    )
    got_loss.backward()

    torch.testing.assert_close(got_loss, want_loss, rtol=1e-6, atol=1e-7)
    assert set(want_metrics) <= set(got_metrics)
    for key, want in want_metrics.items():
        torch.testing.assert_close(got_metrics[key], want, rtol=1e-5, atol=1e-6, msg=key)
    num_packed = sum(inputs["total_lens"])
    torch.testing.assert_close(padded.grad[:, :num_packed], logits.grad, rtol=1e-5, atol=1e-8)
    assert torch.count_nonzero(padded.grad[:, num_packed:]) == 0
