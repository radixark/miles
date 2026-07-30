import pytest
import torch

import miles.backends.training_utils.loss_hub.math_utils as math_utils
from miles.backends.training_utils.loss_hub.math_utils import (
    _calculate_log_probs_and_entropy_true_on_policy,
    _prepare_true_on_policy_full_logits,
    _split_replicated_loss_gather_grad,
)


def test_true_on_policy_logprobs_tp1_truncate_after_real_vocab():
    logits = torch.tensor(
        [
            [1.0, 0.0, -1.0, 3.0, 40.0, 50.0],
            [2.0, 1.0, 0.5, -0.5, 60.0, 70.0],
        ],
        dtype=torch.float16,
    )
    tokens = torch.tensor([3, 0], dtype=torch.long)

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        vocab_size=4,
    )

    expected_log_probs_full = torch.log_softmax(logits[:, :4], dim=-1)
    expected_log_probs = expected_log_probs_full.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(expected_log_probs_full.exp() * expected_log_probs_full).sum(dim=-1)

    assert torch.equal(log_probs, expected_log_probs)
    assert torch.equal(entropy, expected_entropy)


def test_true_on_policy_fake_tp_vocab_gather_truncates_before_log_softmax():
    shard_0 = torch.tensor(
        [
            [5.0, 1.0, -2.0, 0.0],
            [0.0, 3.0, -4.0, 1.0],
        ],
        dtype=torch.float16,
    )
    shard_1 = torch.tensor(
        [
            [2.0, 4.0, 30.0, 40.0],
            [-1.0, 2.0, 50.0, 60.0],
        ],
        dtype=torch.float16,
    )
    tokens = torch.tensor([5, 4], dtype=torch.long)

    gathered_logits = _prepare_true_on_policy_full_logits((shard_0, shard_1), vocab_size=6)
    log_probs, _ = _calculate_log_probs_and_entropy_true_on_policy(
        gathered_logits,
        tokens,
        None,
        vocab_size=6,
    )

    expected_full_logits = torch.cat([shard_0, shard_1], dim=-1)[:, :6]
    expected_log_probs = torch.log_softmax(expected_full_logits, dim=-1)
    expected_selected = expected_log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(gathered_logits, expected_full_logits)
    torch.testing.assert_close(log_probs, expected_selected)


@pytest.mark.parametrize("transport_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scoring_dtype", [torch.bfloat16, torch.float32])
def test_batch_invariant_backend_truncates_before_softmax_and_casts_selected_after_gather(
    monkeypatch,
    transport_dtype,
    scoring_dtype,
):
    seen = {}

    def fake_batch_invariant_log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        seen["shape"] = input.shape
        seen["dtype"] = input.dtype
        return torch.log_softmax(input, dim=dim)

    monkeypatch.setattr(
        math_utils,
        "_load_batch_invariant_log_softmax",
        lambda: fake_batch_invariant_log_softmax,
    )

    logits = torch.tensor(
        [
            [5.0, 1.0, -2.0, 0.0, 50.0, 60.0],
            [0.0, 3.0, -4.0, 1.0, 70.0, 80.0],
        ],
        dtype=scoring_dtype,
        requires_grad=True,
    )
    tokens = torch.tensor([0, 3], dtype=torch.long)

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        vocab_size=4,
        logsoftmax_backend="sglang_batch_invariant",
        logprob_output_dtype=transport_dtype,
    )

    expected_full = torch.log_softmax(logits[:, :4], dim=-1)
    expected_selected = expected_full.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(expected_full.exp() * expected_full).sum(dim=-1)

    assert seen == {"shape": torch.Size([2, 4]), "dtype": scoring_dtype}
    assert log_probs.dtype == transport_dtype
    assert entropy.dtype == scoring_dtype
    torch.testing.assert_close(log_probs.float(), expected_selected.float(), atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(entropy, expected_entropy)

    (log_probs.float().sum() + entropy.sum()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_batch_invariant_backend_fails_closed_for_fp16_logits():
    logits = torch.randn(2, 4, dtype=torch.float16)
    tokens = torch.tensor([0, 1], dtype=torch.long)

    try:
        _calculate_log_probs_and_entropy_true_on_policy(
            logits,
            tokens,
            None,
            logsoftmax_backend="sglang_batch_invariant",
        )
    except ValueError as exc:
        assert "requires BF16 or FP32 logits" in str(exc)
    else:
        raise AssertionError("Expected FP16 batch-invariant scoring to fail")


@pytest.mark.parametrize("scoring_dtype", [torch.bfloat16, torch.float32])
def test_empty_scoring_preserves_entropy_dtype_and_selected_transport_dtype(scoring_dtype):
    logits = torch.empty(0, 4, dtype=scoring_dtype)
    tokens = torch.empty(0, dtype=torch.long)

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        logsoftmax_backend="sglang_batch_invariant",
        logprob_output_dtype=torch.float16,
    )

    assert log_probs.shape == (0,)
    assert log_probs.dtype == torch.float16
    assert entropy is not None
    assert entropy.shape == (0,)
    assert entropy.dtype == scoring_dtype


def test_true_on_policy_replicated_loss_gather_backward_splits_without_tp_sum():
    grad_output = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    rank0_grad = _split_replicated_loss_gather_grad(
        grad_output,
        rank=0,
        world_size=2,
        local_last_dim=3,
    )
    rank1_grad = _split_replicated_loss_gather_grad(
        grad_output,
        rank=1,
        world_size=2,
        local_last_dim=3,
    )

    torch.testing.assert_close(rank0_grad, grad_output[:, :3])
    torch.testing.assert_close(rank1_grad, grad_output[:, 3:])
