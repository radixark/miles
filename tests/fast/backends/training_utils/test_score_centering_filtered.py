"""Filtered rollout probabilities through candidate transport and the real loss."""

import math

import numpy as np
import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.generate_utils.score_centering import (
    append_score_centering_topk,
    configure_score_centering_request,
    validate_score_centering_sample,
)
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


@pytest.fixture
def single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    singleton = GroupInfo(rank=0, size=1, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        ParallelState(
            **{name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        ),
    )


def _filtered_sample() -> Sample:
    # SGLang support mode returns the actual post-filter behavior probabilities.
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        response="2",
        rollout_log_probs=[math.log(4 / 7)],
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[3, 2]]),
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )
    append_score_centering_topk(
        sample,
        {
            "output_token_logprobs": [(math.log(0.4), 2, None)],
            "output_token_sampling_mask": [[3, 2]],
            "output_token_sampling_logprobs": [[math.log(3 / 7), math.log(4 / 7)]],
        },
        3,
        sampling_logprobs_mode="support",
    )
    return sample


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_filtered_loss_matches_dense_support_gradient(single_rank: None, mode: str) -> None:
    sample = _filtered_sample()
    validate_score_centering_sample(sample, 3)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, [[3, 2, -1]])
    np.testing.assert_allclose(np.exp(sample.rollout_topk_log_probs[0, :2]), [3 / 7, 4 / 7])
    args = _args(
        score_centering_is=mode,
        rollout_top_p=0.6,
        rollout_top_k=3,
        use_sampling_support_replay=True,
        entropy_coef=0.03,
    )
    validate_score_centering_args(args)
    data = convert_samples_to_train_data(args, [sample], {}, None, None)
    assert data["rollout_sampling_mask_ids"][0].tolist() == [3, 2]
    batch = {
        "unconcat_tokens": [torch.tensor(sample.tokens)],
        "total_lengths": [len(sample.tokens)],
        "response_lengths": [sample.response_length],
        "loss_masks": [torch.tensor(sample.loss_mask)],
        "rollout_log_probs": [torch.tensor(sample.rollout_log_probs)],
        "rollout_topk_token_ids": data["rollout_topk_token_ids"],
        "rollout_topk_log_probs": data["rollout_topk_log_probs"],
        "advantages": [torch.tensor([0.7])],
    }
    logits = torch.tensor(
        [[[0.0] * 8, [0.0, 0.0, 1.4, 0.2, 2.1, -0.3, 0.1, 0.5], [0.0] * 8]],
        requires_grad=True,
    )
    loss, metrics = score_centering_loss_function(args, batch, logits, torch.mean)
    actual = torch.autograd.grad(loss, logits)[0]

    reference_logits = logits.detach().clone().requires_grad_()
    logp = torch.log_softmax(reference_logits[0, 1, [2, 3]] / 0.7, dim=0)
    q = torch.tensor([4 / 7, 3 / 7])
    ratio = logp.detach().exp() / q
    if mode == "none":
        weight = torch.ones_like(ratio)
    elif mode == "tis":
        weight = ratio.clamp(max=2)
    else:
        weight = torch.where((ratio >= 0.5) & (ratio <= 5), ratio, 0)
    support_entropy = -(logp.exp() * logp).sum()
    reference = -0.7 * (weight[0] * logp[0] - (q * weight * logp).sum()) - 0.03 * support_entropy
    expected = torch.autograd.grad(reference, reference_logits)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual[0, 1, [0, 1, 4, 5, 6, 7]], torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["sc_rollout_head_mass"], torch.ones_like(metrics["sc_rollout_head_mass"]))
    torch.testing.assert_close(metrics["entropy_loss"], support_entropy.detach())


def test_missing_support_candidate_fails_before_training() -> None:
    sample = _filtered_sample()
    sample.rollout_topk_token_ids[0, 1] = -1
    sample.rollout_topk_log_probs[0, 1] = -np.inf
    with pytest.raises(ValueError, match="must equal the sampling support"):
        validate_score_centering_sample(sample, 3)


def test_missing_support_candidate_fails_at_generation() -> None:
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[2, 5, 7, 8]]),
    )
    meta = {
        "output_token_logprobs": [(math.log(0.4), 2, None)],
        "output_token_sampling_mask": [[2, 5, 7, 8]],
        "output_token_sampling_logprobs": [[math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)]],
    }
    with pytest.raises(ValueError, match="support exceeds"):
        append_score_centering_topk(sample, meta, 3, sampling_logprobs_mode="support")


def test_missing_support_probabilities_fail_closed() -> None:
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[2, 3]]),
    )
    meta = {"output_token_logprobs": [(math.log(0.4), 2, None)], "output_token_sampling_mask": [[2, 3]]}
    with pytest.raises(ValueError, match="output_token_sampling_logprobs"):
        append_score_centering_topk(sample, meta, 3, sampling_logprobs_mode="support")


def test_unnormalized_support_probabilities_fail_closed() -> None:
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[2, 3]]),
    )
    meta = {
        "output_token_logprobs": [(math.log(0.4), 2, None)],
        "output_token_sampling_mask": [[2, 3]],
        "output_token_sampling_logprobs": [[math.log(0.4), math.log(0.3)]],
    }
    with pytest.raises(ValueError, match="must sum to one"):
        append_score_centering_topk(sample, meta, 3, sampling_logprobs_mode="support")


@pytest.mark.parametrize("top_p,top_k", [(0.8, -1), (0.8, 4), (1.0, 4)])
def test_uncovered_or_unbounded_support_rejected(top_p: float, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        validate_score_centering_args(_args(rollout_top_p=top_p, rollout_top_k=top_k))


def test_request_override_cannot_exceed_candidate_count() -> None:
    request = {"sampling_params": {"temperature": 0.7, "top_p": 0.6, "top_k": 4}}
    with pytest.raises(ValueError, match="top_k"):
        configure_score_centering_request(_args(rollout_top_p=0.6, rollout_top_k=3), request)
