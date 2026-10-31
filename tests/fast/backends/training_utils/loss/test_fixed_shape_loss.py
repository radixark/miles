"""The fixed-shape (TPU) log-prob, policy-loss and score-centering path agrees with the per-sample path.

`loss/hub/fixed_shape.py` scores the padded microbatch instead of per-sample slices. On CPU both paths
run directly, so these tests feed them the same packed samples (the fixed-shape side gets extra padding
rows of garbage logits) and compare log-probs, entropy, loss, metrics and gradients.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import fixed_shape
from miles.backends.training_utils.loss.hub.fixed_shape import (
    _FusedLogProb,
    fixed_shape_loss_enabled,
    fixed_shape_policy_loss_function,
    fixed_shape_score_centering_loss_function,
    get_fixed_shape_log_probs_and_entropy,
)
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function

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


def _inputs(args, seed: int = 0, *, response_lens: list[int] = RESPONSE_LENS):
    make_parallel_state()
    inputs = make_inputs(
        seed=seed,
        batch_size=len(PROMPT_LENS),
        prompt_lens=PROMPT_LENS,
        response_lens=response_lens,
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
    _assert_policy_loss_matches(_args(**overrides))


def test_the_fused_log_prob_backend_matches_the_per_sample_policy_loss():
    """`--log-probs-backend fused` on TPU only swaps the row scorer; loss, metrics and gradients stay the same."""
    # with Megatron installed the per-sample side's cross-entropy rounds the logits gradient to bf16
    _assert_policy_loss_matches(
        _args(entropy_coef=0.0),
        fixed_shape_args=_args(entropy_coef=0.0, log_probs_backend="fused"),
        grad_tolerance=dict(rtol=2**-7, atol=1e-6),
    )


def _assert_policy_loss_matches(args, *, fixed_shape_args=None, grad_tolerance=None):
    inputs = _inputs(args)

    logits = inputs["policy_logits"].clone().requires_grad_(True)
    batch = make_batch(inputs, "policy_loss")
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.backend.calculate_per_token_loss,
        args.qkv_format,
        None,
    )
    want_loss, want_metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    want_loss.backward()

    padded = _padded(inputs["policy_logits"]).requires_grad_(True)
    got_loss, got_metrics = fixed_shape_policy_loss_function(
        fixed_shape_args or args, make_batch(inputs, "policy_loss"), padded, sum_of_sample_mean=None
    )
    got_loss.backward()

    torch.testing.assert_close(got_loss, want_loss, rtol=1e-6, atol=1e-7)
    assert set(want_metrics) <= set(got_metrics)
    for key, want in want_metrics.items():
        torch.testing.assert_close(got_metrics[key], want, rtol=1e-5, atol=1e-6, msg=key)
    num_packed = sum(inputs["total_lens"])
    torch.testing.assert_close(
        padded.grad[:, :num_packed], logits.grad, **(grad_tolerance or dict(rtol=1e-5, atol=1e-8))
    )
    assert torch.count_nonzero(padded.grad[:, num_packed:]) == 0


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_fused_log_prob_matches_log_softmax(temperature):
    """Log-probs, entropy and the logits gradient match the fp32 log-softmax path."""
    torch.manual_seed(0)
    logits = (torch.randn(PADDED_ROWS, VOCAB) * 3).to(torch.bfloat16).requires_grad_(True)
    targets = torch.randint(0, VOCAB, (PADDED_ROWS,))
    weights = torch.randn(PADDED_ROWS)

    log_probs, entropy = _FusedLogProb.apply(logits, targets, temperature)
    (log_probs * weights).sum().backward()
    fused_grad, logits.grad = logits.grad, None

    scaled = logits.float() / temperature
    ref = torch.log_softmax(scaled, -1).gather(-1, targets[:, None]).squeeze(-1)
    ref_entropy = -(torch.softmax(scaled, -1) * torch.log_softmax(scaled, -1)).sum(-1)
    (ref * weights).sum().backward()

    torch.testing.assert_close(log_probs, ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(entropy, ref_entropy.detach(), atol=1e-4, rtol=0)
    torch.testing.assert_close(fused_grad.float(), logits.grad.float(), atol=1e-6, rtol=0)


TOP_K = 5


def _score_centering_args(**overrides):
    defaults = dict(
        loss_type="score_centering",
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        rollout_top_logprobs_num=TOP_K,
    )
    return _args(**{**defaults, **overrides})


def _score_centering_batch(inputs: dict) -> dict:
    """`make_batch` plus rollout top-k candidates: distinct ids, short rows padded with -1 / -inf."""
    generator = torch.Generator().manual_seed(1)
    ids, log_probs = [], []
    for length in inputs["response_lens"]:
        sample_ids = torch.rand((length, VOCAB), generator=generator).argsort(dim=-1)[:, :TOP_K]
        sample_log_probs = -torch.rand((length, TOP_K), generator=generator) * 4
        sample_ids[1::4, 3:] = -1
        sample_log_probs[1::4, 3:] = -torch.inf
        ids.append(sample_ids.numpy().astype(np.int32))
        log_probs.append(sample_log_probs.numpy())
    batch = make_batch(inputs, "score_centering")
    batch["rollout_topk_token_ids"] = ids
    batch["rollout_topk_log_probs"] = log_probs
    # rows outside the loss mask may carry NaN; both paths must mask them before use
    batch["rollout_log_probs"][0][::3] = torch.nan
    batch["advantages"][0][::3] = torch.nan
    return batch


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(score_centering_is="tis"),
        dict(score_centering_is="mis"),
        dict(entropy_coef=0.01),
        dict(entropy_coef=0.0, observe_training_entropy=True),
        dict(use_kl_loss=True, kl_loss_coef=0.05, kl_loss_type="low_var_kl"),
        dict(use_kl_loss=True, kl_loss_coef=0.05, use_unbiased_kl=True),
        dict(calculate_per_token_loss=True),
        dict(rollout_temperature=0.7),
        dict(log_probs_chunk_size=-1),
    ],
    ids=["none", "tis", "mis", "entropy", "observe-entropy", "kl", "unbiased-kl", "per-token", "temp", "unchunked"],
)
def test_score_centering_loss_matches(overrides):
    _assert_score_centering_loss_matches(_score_centering_args(**{"entropy_coef": 0.0, **overrides}))


def test_score_centering_loss_matches_with_an_empty_response():
    """A sample with no response tokens (e.g. aborted before its first token) scores no rows."""
    _assert_score_centering_loss_matches(_score_centering_args(entropy_coef=0.0), response_lens=[25, 0, 40])


def _assert_score_centering_loss_matches(args, *, response_lens: list[int] = RESPONSE_LENS):
    inputs = _inputs(args, response_lens=response_lens)

    logits = inputs["policy_logits"].clone().requires_grad_(True)
    batch = _score_centering_batch(inputs)
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.backend.calculate_per_token_loss,
        args.qkv_format,
        None,
    )
    want_loss, want_metrics = score_centering_loss_function(args, batch, logits, sum_of_sample_mean)
    want_loss.backward()

    padded = _padded(inputs["policy_logits"]).requires_grad_(True)
    got_loss, got_metrics = fixed_shape_score_centering_loss_function(
        args, _score_centering_batch(inputs), padded, sum_of_sample_mean=None
    )
    got_loss.backward()

    assert torch.isfinite(want_loss)
    torch.testing.assert_close(got_loss, want_loss, rtol=1e-6, atol=1e-7)
    assert set(got_metrics) == set(want_metrics)
    for key, want in want_metrics.items():
        torch.testing.assert_close(got_metrics[key], want, rtol=1e-5, atol=1e-6, msg=key)
    num_packed = sum(inputs["total_lens"])
    torch.testing.assert_close(padded.grad[:, :num_packed], logits.grad, rtol=1e-5, atol=1e-8)
    assert torch.count_nonzero(padded.grad[:, num_packed:]) == 0


def _gate_args(**overrides) -> Namespace:
    values = dict(
        train_backend="torchtitan",
        qkv_format="thd",
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        loss_type="score_centering",
        advantage_estimator="grpo",
        true_on_policy_mode=False,
        allgather_cp=False,
        use_sampling_support_replay=False,
        use_rollout_routing_replay=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_opd=False,
        custom_pg_loss_reducer_function_path=None,
        dump_details=None,
        multi_lora=False,
    )
    return Namespace(**{**values, **overrides})


@pytest.mark.parametrize(
    "overrides, enabled",
    [
        (dict(), True),
        (dict(loss_type="policy_loss"), True),
        (dict(loss_type="sft_loss"), False),
        # the forward-only pass would need the dense sampling mask, which has data-dependent shapes
        (dict(use_sampling_support_replay=True), False),
    ],
    ids=["score-centering", "policy-loss", "sft-loss", "score-centering-replay"],
)
def test_fixed_shape_loss_gate(monkeypatch, overrides, enabled):
    # TpuPlatform's values; importing it would import torch_tpu
    monkeypatch.setattr(fixed_shape.platforms, "IS_TPU", True)
    monkeypatch.setattr(fixed_shape.platforms.current_platform, "dynamic_shapes", False)
    assert fixed_shape_loss_enabled(_gate_args(**overrides)) is enabled
