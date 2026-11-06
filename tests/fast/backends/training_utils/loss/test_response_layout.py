"""Each loss gives the same log-probs, entropy, loss, metrics and gradients on `PackedSlices` and `PaddedRows`.

The padded side gets the same packed samples followed by rows of garbage logits.
"""

from __future__ import annotations

import itertools
import logging
import re
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import advantages, response_layout
from miles.backends.training_utils.loss.hub.advantages import normalize_advantages
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.hub.response_layout import (
    PackedSlices,
    PaddedRows,
    padded_rows_enabled,
    validate_response_layout_args,
)
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state

PROMPT_LENS = [7, 30, 3]
RESPONSE_LENS = [25, 2, 40]
VOCAB = 64
PADDED_ROWS = 128  # > sum(PROMPT_LENS + RESPONSE_LENS) = 107
TOP_K = 5


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("response-layout") / "process-group"
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


def _batch(inputs: dict, loss_type: str) -> dict:
    """`make_batch` plus the boundaries `get_batch` records: samples back to back, then padding to PADDED_ROWS."""
    batch = make_batch(inputs, loss_type)
    batch["cu_seqlens_host"] = (*itertools.accumulate(inputs["total_lens"], initial=0), PADDED_ROWS)
    return batch


def _reducer(args, batch):
    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.backend.calculate_per_token_loss,
        args.qkv_format,
        None,
        denominators=batch.get("rollout_mask_sums"),
    )


def _assert_both_layouts_match(monkeypatch, loss_function, args, inputs, batch_of):
    """`loss_function` on PackedSlices (the CPU default) and on PaddedRows agrees on loss, metrics and gradients."""
    logits = inputs["policy_logits"].clone().requires_grad_(True)
    batch = batch_of(inputs)
    want_loss, want_metrics = loss_function(args, batch, logits, _reducer(args, batch))
    want_loss.backward()

    monkeypatch.setattr(response_layout, "padded_rows_enabled", lambda args: True)
    padded = _padded(inputs["policy_logits"]).requires_grad_(True)
    batch = batch_of(inputs)
    got_loss, got_metrics = loss_function(args, batch, padded, _reducer(args, batch))
    got_loss.backward()

    assert torch.isfinite(want_loss)
    torch.testing.assert_close(got_loss, want_loss, rtol=1e-6, atol=1e-7)
    assert set(got_metrics) == set(want_metrics)
    for key, want in want_metrics.items():
        torch.testing.assert_close(got_metrics[key], want, rtol=1e-5, atol=1e-6, msg=key)
    num_packed = sum(inputs["total_lens"])
    torch.testing.assert_close(padded.grad[:, :num_packed], logits.grad, rtol=1e-5, atol=1e-8)
    assert torch.count_nonzero(padded.grad[:, num_packed:]) == 0


@pytest.mark.parametrize("log_probs_chunk_size", [-1, 32, 4096])
def test_every_bucket_is_scored_in_the_same_number_of_chunks(log_probs_chunk_size):
    """Each chunk all-reduces over TP; data-parallel ranks on different buckets must issue as many collectives."""
    args = _args(log_probs_chunk_size=log_probs_chunk_size)
    batch = _batch(_inputs(args), "policy_loss")
    num_chunks = set()
    for num_rows in (PADDED_ROWS, 1024, 8192):
        chunk_size = PaddedRows.build(args, batch, torch.empty(1, num_rows, 1)).chunk_size
        num_chunks.add((num_rows - 1) // chunk_size + 1)  # how calculate_log_probs_and_entropy splits the rows
    assert num_chunks == {response_layout.ROW_CHUNKS}


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_forward_log_probs_and_entropy_match(temperature):
    args = _args(rollout_temperature=temperature)
    inputs = _inputs(args)
    batch = _batch(inputs, "policy_loss")
    common = dict(with_entropy=True, use_sampling_mask=False)
    expected = PackedSlices.build(args, batch, inputs["policy_logits"]).forward_log_probs(
        inputs["policy_logits"].clone(), **common
    )
    padded = _padded(inputs["policy_logits"])
    (actual,) = PaddedRows.finish_forward_pass(
        [PaddedRows.build(args, batch, padded).forward_log_probs(padded, **common)]
    )
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
        dict(rollout_mask_sums=True),
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
        "rollout-mask-sums",
    ],
)
def test_policy_loss_matches(monkeypatch, overrides):
    use_rollout_mask_sums = overrides.pop("rollout_mask_sums", False)
    args = _args(**overrides)

    def batch_of(inputs):
        batch = _batch(inputs, "policy_loss")
        if use_rollout_mask_sums:
            # the denominators the rollout fixed before any rejection, here larger than the masks' own sums
            batch["rollout_mask_sums"] = torch.tensor([float(mask.numel()) for mask in batch["loss_masks"]])
        return batch

    _assert_both_layouts_match(monkeypatch, policy_loss_function, args, _inputs(args), batch_of)


def test_unbiased_kl_keeps_tokens_outside_the_loss_mask_out_of_the_gradient():
    """A token outside the loss mask may carry any old log-prob; where exp(log_probs - old_log_probs) overflows
    there, the masked-out KL must not turn the logits' gradient into 0 * inf = NaN."""
    args = _args(use_kl_loss=True, kl_loss_coef=0.05, use_unbiased_kl=True)
    inputs = _inputs(args)
    inputs["log_probs"][0][::3] = -1e4  # the tokens _inputs masks out
    logits = inputs["policy_logits"].clone().requires_grad_(True)
    batch = make_batch(inputs, "policy_loss")
    loss, _ = policy_loss_function(args, batch, logits, _reducer(args, batch))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_host_advantages_whiten_over_the_given_group(monkeypatch):
    """PaddedRows keeps advantages on the host, so the statistics must reduce over the group the actor passes in."""
    whiten_groups = []

    def fake_whiten(values, masks, *, process_group, shift_mean):
        whiten_groups.append(process_group)
        return values

    monkeypatch.setattr(advantages, "distributed_masked_whiten", fake_whiten)
    args = _args(normalize_advantages=True)
    batch = _batch(_inputs(args), "policy_loss")
    gloo_group = object()
    normalize_advantages(
        args,
        batch["advantages"],
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        whiten_group=gloo_group,
    )
    assert whiten_groups == [gloo_group]


def _score_centering_args(**overrides):
    defaults = dict(
        loss_type="score_centering",
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        rollout_top_logprobs_num=TOP_K,
        entropy_coef=0.0,
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
    batch = _batch(inputs, "score_centering")
    batch["rollout_topk_token_ids"] = ids
    batch["rollout_topk_log_probs"] = log_probs
    # rows outside the loss mask may carry NaN; both layouts must mask them before use
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
        dict(observe_training_entropy=True),
        dict(use_kl_loss=True, kl_loss_coef=0.05, kl_loss_type="low_var_kl"),
        dict(use_kl_loss=True, kl_loss_coef=0.05, use_unbiased_kl=True),
        dict(calculate_per_token_loss=True),
        dict(rollout_temperature=0.7),
        dict(log_probs_chunk_size=-1),
    ],
    ids=["none", "tis", "mis", "entropy", "observe-entropy", "kl", "unbiased-kl", "per-token", "temp", "unchunked"],
)
def test_score_centering_loss_matches(monkeypatch, overrides):
    args = _score_centering_args(**overrides)
    _assert_both_layouts_match(monkeypatch, score_centering_loss_function, args, _inputs(args), _score_centering_batch)


def test_score_centering_loss_matches_with_an_empty_response(monkeypatch):
    """A sample with no response tokens (e.g. aborted before its first token) scores no rows."""
    args = _score_centering_args()
    inputs = _inputs(args, response_lens=[25, 0, 40])
    _assert_both_layouts_match(monkeypatch, score_centering_loss_function, args, inputs, _score_centering_batch)


def _startup_args(**overrides) -> Namespace:
    values = dict(
        train_backend="torchtitan",
        loss_type="policy_loss",
        qkv_format="thd",
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        advantage_estimator="grpo",
        true_on_policy_mode=False,
        log_probs_backend="torch",
        allgather_cp=False,
        use_sampling_support_replay=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_opd=False,
        custom_pg_loss_reducer_function_path=None,
        dump_details=None,
        multi_lora=False,
    )
    return Namespace(**{**values, **overrides})


@pytest.fixture
def static_shapes(monkeypatch):
    # TpuPlatform's value; importing it would import torch_tpu
    monkeypatch.setattr(response_layout.platforms.current_platform, "dynamic_shapes", False)


@pytest.mark.parametrize(
    "overrides,enabled",
    [
        (dict(), True),
        (dict(loss_type="score_centering"), True),
        (dict(loss_type="sft_loss"), False),
        (dict(train_backend="fsdp"), False),
    ],
    ids=["policy-loss", "score-centering", "sft-loss", "fsdp"],
)
def test_a_static_shape_platform_pads_the_losses_written_on_the_layout(static_shapes, overrides, enabled):
    assert padded_rows_enabled(_startup_args(**overrides)) is enabled


@pytest.mark.parametrize(
    "overrides,refused",
    [
        (dict(use_tis=True), "--use-tis"),
        (dict(loss_type="score_centering", use_sampling_support_replay=True), "--use-sampling-support-replay"),
        (dict(context_parallel_size=2), "--context-parallel-size 1"),
        (dict(advantage_estimator="gspo"), "--advantage-estimator"),
        (dict(log_probs_backend="fused"), "--log-probs-backend fused"),
    ],
)
def test_a_static_shape_platform_refuses_what_padded_rows_cannot_score(static_shapes, overrides, refused):
    """It would otherwise take the per-sample path, which compiles a program per response length."""
    with pytest.raises(ValueError, match=re.escape(refused)):
        validate_response_layout_args(_startup_args(**overrides))


def test_a_supported_run_and_every_dynamic_shape_run_pass_startup(static_shapes, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=response_layout.__name__):
        validate_response_layout_args(_startup_args())
        assert not caplog.records
        validate_response_layout_args(_startup_args(loss_type="sft_loss", use_tis=True))  # not on the layout
        assert "per-sample loss path" in caplog.text
        caplog.clear()
        monkeypatch.setattr(response_layout.platforms.current_platform, "dynamic_shapes", True)
        validate_response_layout_args(_startup_args(loss_type="sft_loss", use_tis=True))
        assert not caplog.records
