from argparse import Namespace
from copy import deepcopy

import pytest
import torch
from tests.fast.backends.training_utils.test_true_on_policy_loss_metrics import (
    _make_args,
    _make_batch,
    _patch_single_rank_loss_helpers,
)

from miles.backends.training_utils import loss
from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub import losses as loss_utils
from miles.backends.training_utils.loss_hub.credit_assignment import constrain_positive_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _set_parallel(rank: int = 0, cp_size: int = 1) -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=GroupInfo(rank=rank, size=cp_size, group=None),
            tp=group,
            pp=group,
            ep=group,
            etp=group,
            indep_dp=group,
        )
    )


@pytest.fixture(autouse=True)
def _single_rank():
    _set_parallel()
    yield
    _set_parallel()


def test_only_positive_credit_inside_failed_spans_changes() -> None:
    original = torch.tensor([2.0, -3.0, 0.0, 4.0, 5.0, -6.0, 7.0])
    snapshot = original.clone()
    result = constrain_positive_advantages([original], [[[0, 3], [5, 7]]], [10], [7])[0]
    torch.testing.assert_close(result, torch.tensor([0.0, -3.0, 0.0, 4.0, 5.0, -6.0, 0.0]))
    torch.testing.assert_close(original, snapshot)  # may also be a value target


def test_missing_field_is_noop_and_empty_spans_keep_values() -> None:
    original = [torch.tensor([2.0, -1.0])]
    assert constrain_positive_advantages(original, None, [4], [2]) is original
    result = constrain_positive_advantages(original, [[]], [4], [2])
    torch.testing.assert_close(result[0], original[0])


@pytest.mark.parametrize("span", [[-1, 2], [0, 4], [2, 2], [2, 1], [True, 2], [0.0, 2], [0], None, "01"])
def test_malformed_spans_fail(span: object) -> None:
    with pytest.raises(ValueError, match="Invalid non-positive-advantage span"):
        constrain_positive_advantages([torch.ones(3)], [[span]], [5], [3])


def test_batch_and_local_shape_mismatch_fail() -> None:
    with pytest.raises(ValueError, match="align"):
        constrain_positive_advantages([torch.ones(3)], [], [5], [3])
    with pytest.raises(ValueError, match="shape"):
        constrain_positive_advantages([torch.ones(2)], [[]], [5], [3])


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("qkv_format,max_seq_len", [("thd", None), ("bshd", 12)])
@pytest.mark.parametrize("response_length", [2, 6])
def test_context_parallel_alignment(rank: int, qkv_format: str, max_seq_len: int | None, response_length: int) -> None:
    _set_parallel(rank, cp_size=2)
    full = torch.arange(1, response_length + 1, dtype=torch.float32)
    total_length = 7
    expected = full.clone()
    expected[-2:] = 0  # includes the final stop token, possibly on only one rank
    local = slice_log_prob_with_cp(full, total_length, response_length, qkv_format, max_seq_len)
    local_expected = slice_log_prob_with_cp(expected, total_length, response_length, qkv_format, max_seq_len)
    actual = constrain_positive_advantages(
        [local],
        [[[response_length - 2, response_length]]],
        [total_length],
        [response_length],
        qkv_format,
        [max_seq_len] if max_seq_len is not None else None,
    )[0]
    torch.testing.assert_close(actual, local_expected)


def _args(**overrides) -> Namespace:
    return Namespace(
        **{
            "skip_actor_forward_only": False,
            "use_rollout_logprobs": False,
            "kl_coef": 0.0,
            "advantage_estimator": "grpo",
            "qkv_format": "thd",
            "use_opd": False,
            "normalize_advantages": False,
            **overrides,
        }
    )


def test_actual_advantage_pipeline_keeps_returns_masks_and_negative_credit() -> None:
    data = {
        "log_probs": [torch.zeros(7), torch.zeros(7)],
        "rewards": [2.0, -1.0],
        "response_lengths": [7, 7],
        "total_lengths": [9, 9],
        "loss_masks": [torch.tensor([1, 1, 1, 0, 0, 1, 1])] * 2,
        "non_positive_advantage_spans": [[[0, 3]], [[0, 3]]],
    }
    old_masks = deepcopy(data["loss_masks"])
    loss.compute_advantages_and_returns(_args(), data)
    torch.testing.assert_close(data["advantages"][0], torch.tensor([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]))
    torch.testing.assert_close(data["advantages"][1], torch.full((7,), -1.0))
    torch.testing.assert_close(data["returns"][0], torch.full((7,), 2.0))
    for mask, old_mask in zip(data["loss_masks"], old_masks, strict=True):
        torch.testing.assert_close(mask, old_mask)
    assert data["rewards"] == [2.0, -1.0]


def test_constraint_is_after_whitening(monkeypatch: pytest.MonkeyPatch) -> None:
    def whiten(*args) -> list[torch.Tensor]:
        # Even a negative pre-normalization reward can become positive here.
        return [torch.tensor([3.0, 2.0, -1.0])]

    monkeypatch.setattr(loss, "normalize_advantages", whiten)
    data = {
        "log_probs": [torch.zeros(3)],
        "rewards": [-1.0],
        "response_lengths": [3],
        "total_lengths": [5],
        "loss_masks": [torch.ones(3)],
        "non_positive_advantage_spans": [[[0, 2]]],
    }
    loss.compute_advantages_and_returns(_args(normalize_advantages=True), data)
    torch.testing.assert_close(data["advantages"][0], torch.tensor([0.0, 0.0, -1.0]))


def test_zero_positive_attempt_credit_preserves_kl_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.01
    args.kl_loss_type = "k3"
    batch = _make_batch(old_log_probs=torch.tensor([-0.5, -0.5]), rollout_log_probs=torch.tensor([-0.5, -0.5]))
    batch["ref_log_probs"] = [torch.tensor([-1.0, -1.0])]
    batch["advantages"] = constrain_positive_advantages([torch.ones(2)], [[[0, 2]]], [3], [2])
    logits = torch.tensor([[[-0.5], [-0.5], [0.0]]], requires_grad=True)
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda logits, *args, **kwargs: {"log_probs": [logits.flatten()[:2]]},
    )
    actual, _ = loss_utils.policy_loss_function(args, batch, logits, sum_of_sample_mean=lambda tensor: tensor.mean())
    actual.backward()
    # Both tokens have zero PG credit, but still get the separate KL gradient.
    assert actual.item() > 0
    assert torch.all(logits.grad.flatten()[:2] != 0)
    torch.testing.assert_close(batch["loss_masks"][0], torch.ones(2))
