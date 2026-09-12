from argparse import Namespace
from dataclasses import replace

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.log_utils import aggregate_forward_results
from miles.backends.training_utils.loss_hub import logit_processors
from miles.backends.training_utils.parallel import GroupInfo


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_old_candidate_scores_align_with_packed_responses_and_stay_fixed(monkeypatch, qkv_format):
    make_parallel_state()
    args = make_args(qkv_format=qkv_format, true_on_policy_mode=False)
    # Sampled-token scoring is independently covered by the existing loss tests.
    monkeypatch.setattr(
        logit_processors,
        "calculate_log_probs_and_entropy",
        lambda logits, tokens, *a, **kw: (logits.log_softmax(-1).gather(-1, tokens[:, None]), None),
    )
    torch.manual_seed(81)
    lengths, responses = [5, 3, 4], [2, 0, 3]
    tokens = [torch.randint(0, 11, (n,)) for n in lengths]
    ids = [
        torch.stack([torch.randperm(11)[:3] for _ in range(n)]) if n else torch.empty(0, 3, dtype=torch.long)
        for n in responses
    ]
    padded = [5] * 3 if qkv_format == "bshd" else None
    logits = torch.randn((3, 5, 11) if padded else (1, 12, 11), requires_grad=True)
    result = logit_processors.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=tokens,
        total_lengths=lengths,
        response_lengths=responses,
        max_seq_lens=padded,
        opd_candidate_ids=ids,
    )
    old = result["opd_candidate_old_log_probs"]
    offset = 0
    for i, (length, response) in enumerate(zip(lengths, responses, strict=True)):
        sample_logits = logits[i, :length] if padded else logits[0, offset : offset + length]
        expected = sample_logits[length - response - 1 : length - 1].log_softmax(-1).gather(-1, ids[i])
        torch.testing.assert_close(old[i], expected)
        assert not old[i].requires_grad
        offset += length
    frozen = [value.clone() for value in old]
    with torch.no_grad():
        logits.add_(torch.randn_like(logits))
    for value, expected in zip(old, frozen, strict=True):
        torch.testing.assert_close(value, expected)


def test_old_candidate_cache_restores_dynamic_microbatch_order():
    values = [torch.full((n, 3), float(n)) for n in [2, 4, 1]]
    iterator = DataIterator({}, micro_batch_indices=[[2, 0], [1]])
    output = aggregate_forward_results(
        [{"opd_candidate_old_log_probs": [values[2], values[0]]}, {"opd_candidate_old_log_probs": [values[1]]}],
        iterator,
        Namespace(use_dynamic_batch_size=True),
    )
    for result, expected in zip(output["opd_candidate_old_log_probs"], values, strict=True):
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(("rank", "indices"), [(0, [6]), (1, [0, 1, 2, 3, 4, 5])])
def test_old_candidate_scores_follow_zigzag_cp(monkeypatch, rank, indices):
    state = replace(make_parallel_state(), cp=GroupInfo(rank=rank, size=2, group=None))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(
        logit_processors,
        "calculate_log_probs_and_entropy",
        lambda logits, tokens, *a, **kw: (logits.log_softmax(-1).gather(-1, tokens[:, None]), None),
    )
    torch.manual_seed(98)
    global_logits = torch.randn(12, 23)
    positions = [0, 1, 2, 9, 10, 11] if rank == 0 else [3, 4, 5, 6, 7, 8]
    ids = torch.arange(21).reshape(7, 3)
    result = logit_processors.get_log_probs_and_entropy(
        global_logits[positions].unsqueeze(0),
        args=make_args(true_on_policy_mode=False),
        unconcat_tokens=[torch.arange(11)],
        total_lengths=[11],
        response_lengths=[7],
        opd_candidate_ids=[ids[indices]],
    )
    expected = global_logits[3:10].log_softmax(-1).gather(-1, ids)[indices]
    torch.testing.assert_close(result["opd_candidate_old_log_probs"][0], expected)
