"""Unit tests for `get_opd_full_kl`, the disaggregated-OPD-teacher KL callback.

Verifies the exact reverse-KL(student || teacher) math against a hand-computed
reference, and that hidden-state chunks are drawn from the iterator in the same
per-sample order `get_responses` yields student logits chunks.
"""

from argparse import Namespace

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_opd_full_kl
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _make_trivial_parallel_state() -> None:
    def _trivial_group() -> GroupInfo:
        return GroupInfo(rank=0, size=1, group=None)

    set_parallel_state(
        ParallelState(
            intra_dp=_trivial_group(),
            intra_dp_cp=_trivial_group(),
            cp=_trivial_group(),
            tp=_trivial_group(),
            pp=_trivial_group(),
            ep=_trivial_group(),
            etp=_trivial_group(),
            indep_dp=_trivial_group(),
            is_pp_last_stage=True,
        )
    )


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        qkv_format="thd",
        rollout_temperature=1.0,
        allgather_cp=False,
        true_on_policy_mode=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _reference_reverse_kl(student_logits: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    return (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)


def _identity_teacher_unembedding(hidden_states: torch.Tensor) -> torch.Tensor:
    """Treats the received "hidden states" as if they were already teacher logits,
    just to keep the test's fixtures simple -- the real TeacherUnembedding is tested
    separately in test_opd_teacher_unembedding.py."""
    return torch.log_softmax(hidden_states.float(), dim=-1)


def test_get_opd_full_kl_matches_reference_for_two_samples():
    _make_trivial_parallel_state()
    vocab_size = 4
    total_lengths = [5, 4]
    response_lengths = [2, 1]
    unconcat_tokens = [torch.arange(length) for length in total_lengths]

    torch.manual_seed(0)
    student_logits = torch.randn(1, sum(total_lengths), vocab_size)
    # "teacher hidden states" (here, logits, per _identity_teacher_unembedding) for the
    # 3 total response tokens (2 + 1), drawn from the iterator in per-sample order.
    teacher_hidden_states = [torch.randn(2, vocab_size), torch.randn(1, vocab_size)]

    result = get_opd_full_kl(
        student_logits,
        args=_make_args(),
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        teacher_unembedding=_identity_teacher_unembedding,
        teacher_hidden_states_iter=iter(teacher_hidden_states),
    )

    assert list(result.keys()) == ["opd_reverse_kl"]
    sample_0_kl, sample_1_kl = result["opd_reverse_kl"]

    # Response-aligned student logits chunk for sample 0: positions [2:4) (see
    # get_responses' thd/cp_size==1 slicing, tested directly in test_hidden_states.py).
    expected_kl_0 = _reference_reverse_kl(
        student_logits.squeeze(0)[2:4], _identity_teacher_unembedding(teacher_hidden_states[0])
    )
    expected_kl_1 = _reference_reverse_kl(
        student_logits.squeeze(0)[7:8], _identity_teacher_unembedding(teacher_hidden_states[1])
    )

    assert torch.allclose(sample_0_kl, expected_kl_0, atol=1e-5)
    assert torch.allclose(sample_1_kl, expected_kl_1, atol=1e-5)


def test_get_opd_full_kl_is_zero_when_student_matches_teacher():
    _make_trivial_parallel_state()
    vocab_size = 5
    # 1 prompt token + 3 response tokens; get_responses' next-token shift means the
    # response-aligned logits chunk is positions [0:3) of this 4-position sequence.
    total_lengths = [4]
    response_lengths = [3]
    unconcat_tokens = [torch.arange(4)]

    torch.manual_seed(1)
    full_sequence_logits = torch.randn(4, vocab_size)
    student_logits = full_sequence_logits.unsqueeze(0)
    expected_response_chunk = full_sequence_logits[0:3]

    result = get_opd_full_kl(
        student_logits,
        args=_make_args(),
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        teacher_unembedding=_identity_teacher_unembedding,
        teacher_hidden_states_iter=iter([expected_response_chunk]),
    )

    assert torch.allclose(result["opd_reverse_kl"][0], torch.zeros(3), atol=1e-5)
