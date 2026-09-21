"""target_tokens are explicit gather labels; shifted targets reproduce next-token scoring."""

import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss_hub import math_utils, tinker_losses

VOCAB = 32


def _naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None):
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def _target_logprobs(batch, logits):
    args = make_args(true_on_policy_mode=False, rollout_temperature=1.0, log_probs_chunk_size=64)
    return tinker_losses._target_logprobs(args, batch, logits)


def _batch(sequences: list[list[int]], targets: list[list[int]]) -> dict:
    return {
        "unconcat_tokens": [torch.tensor(sequence) for sequence in sequences],
        "target_tokens": targets,
        "total_lengths": [len(sequence) for sequence in sequences],
        "response_lengths": [len(target) for target in targets],
    }


def test_targets_replace_the_sequence_as_gather_labels(monkeypatch):
    monkeypatch.setattr(math_utils, "compute_log_probs", _naive_compute_log_probs)
    make_parallel_state()
    generator = torch.Generator().manual_seed(0)
    sequences = [[1, 2, 3, 4, 5], [6, 7, 8]]
    targets = [[9, 10, 11], [12, 13]]  # rl_loop-style: labels differ from the sequence
    logits = torch.randn((1, 8, VOCAB), generator=generator)

    log_probs = _target_logprobs(_batch(sequences, targets), logits)

    flat = torch.log_softmax(logits.squeeze(0), dim=-1)
    offset = 0
    for sequence, target, log_prob in zip(sequences, targets, log_probs, strict=True):
        start = offset + len(sequence) - len(target) - 1
        expected = [flat[start + i, label] for i, label in enumerate(target)]
        torch.testing.assert_close(log_prob, torch.stack(expected))
        offset += len(sequence)


def test_shifted_targets_degenerate_to_next_token_scoring(monkeypatch):
    monkeypatch.setattr(math_utils, "compute_log_probs", _naive_compute_log_probs)
    make_parallel_state()
    generator = torch.Generator().manual_seed(1)
    sequence = torch.randint(0, VOCAB, (6,), generator=generator).tolist()
    logits = torch.randn((1, 6, VOCAB), generator=generator)

    (log_probs,) = _target_logprobs(_batch([sequence], [sequence[1:]]), logits)

    expected = torch.log_softmax(logits.squeeze(0)[:-1], dim=-1).gather(-1, torch.tensor(sequence[1:]).unsqueeze(-1))
    torch.testing.assert_close(log_probs, expected.squeeze(-1))
