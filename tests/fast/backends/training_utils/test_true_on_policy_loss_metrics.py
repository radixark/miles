from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import loss as loss_utils


def _make_args(*, use_rollout_logprobs: bool) -> Namespace:
    return Namespace(
        use_rollout_logprobs=use_rollout_logprobs,
        use_opsm=False,
        advantage_estimator="ppo",
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=False,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        kl_loss_coef=0.0,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        allgather_cp=False,
    )


def _make_batch(*, old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> dict:
    return {
        "advantages": [torch.tensor([1.0, -0.5], dtype=torch.float32)],
        "log_probs": [old_log_probs],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.tensor([7, 8, 9], dtype=torch.long)],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 1.0], dtype=torch.float32)],
    }


def _patch_single_rank_masks(monkeypatch):
    monkeypatch.setattr(
        loss_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )


def test_log_probs_remain_1d_for_single_token_true_on_policy_chunks(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.true_on_policy_mode = True
    args.bf16 = True
    args.fp16 = False
    args.vocab_size = 8

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    monkeypatch.setattr(
        loss_utils,
        "get_responses",
        lambda *args, **kwargs: iter([(torch.zeros((1, 8), dtype=torch.bfloat16), torch.tensor([3]))]),
    )
    monkeypatch.setattr(
        loss_utils,
        "calculate_log_probs_and_entropy",
        lambda *args, **kwargs: (torch.tensor([-2.0], dtype=torch.bfloat16), None),
    )

    result = loss_utils.get_log_probs_and_entropy(
        torch.zeros((1, 1, 8), dtype=torch.bfloat16),
        args=args,
        unconcat_tokens=[torch.tensor([3])],
        total_lengths=[1],
        response_lengths=[1],
    )

    assert result["log_probs"][0].shape == (1,)


@pytest.mark.parametrize(
    ("use_rollout_logprobs", "train_log_probs", "old_log_probs", "rollout_log_probs", "expected_abs_diff"),
    [
        (
            False,
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.0,
        ),
        (
            True,
            torch.tensor([0.50, 1.00], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.15,
        ),
    ],
)
def test_train_rollout_logprob_abs_diff_uses_recomputed_train_logprobs(
    monkeypatch,
    use_rollout_logprobs: bool,
    train_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    expected_abs_diff: float,
):
    args = _make_args(use_rollout_logprobs=use_rollout_logprobs)
    batch = _make_batch(old_log_probs=old_log_probs, rollout_log_probs=rollout_log_probs)

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_masks(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [train_log_probs.clone()],
            "entropy": [torch.zeros_like(train_log_probs)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(expected_abs_diff))


def test_zero_weighted_entropy_nan_does_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_masks(monkeypatch)

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        assert kwargs["with_entropy"] is False
        return {"log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)]}

    monkeypatch.setattr(loss_utils, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["entropy_loss"], torch.tensor(0.0))


def test_zero_weighted_kl_nan_does_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.0
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )
    batch["ref_log_probs"] = [torch.tensor([float("nan"), float("nan")], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_masks(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])


def test_masked_nonfinite_ppo_terms_do_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
    )
    batch["loss_masks"] = [torch.tensor([1.0, 0.0], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_masks(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, float("nan")], dtype=torch.float32)],
        },
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: (tensor * batch["loss_masks"][0]).sum(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["ppo_kl"])
