import torch

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

    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)


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
