"""Unit tests for the decoupled on-policy-distillation (OPD) loss path.

`apply_opd_kl_to_advantages` is orthogonal to the advantage estimator: it adds a
reverse-KL penalty (student_logp - teacher_logp) to per-token advantages. These
tests cover the math and the guard rails without needing the external loss
snapshot artifacts.
"""

import math
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import loss as loss_utils
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages

from .loss_test_utils import make_parallel_state

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


def _args(opd_kl_coef: float = 1.0) -> Namespace:
    return Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=opd_kl_coef)


def test_subtracts_weighted_reverse_kl_and_stores_metric():
    args = _args(opd_kl_coef=0.5)
    student = [torch.tensor([0.0, 1.0], requires_grad=True)]
    teacher = [torch.tensor([0.0, 0.0], requires_grad=True)]
    advantages = [torch.tensor([2.0, 2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # reverse_kl = student - teacher = [0, 1]; adv - 0.5 * reverse_kl = [2.0, 1.5]
    assert torch.allclose(advantages[0], torch.tensor([2.0, 1.5]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 1.0]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.2, -0.3], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert student[0].grad is None
    assert teacher[0].grad is None


def test_precomputed_reverse_kl_is_detached_before_weighting_advantages():
    args = _args(opd_kl_coef=0.25)
    precomputed = torch.tensor([0.4, -0.2], requires_grad=True)
    advantages = [torch.tensor([0.0, 0.0])]
    rollout_data = {"opd_reverse_kl": [precomputed]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs=[torch.zeros(2)])

    torch.testing.assert_close(advantages[0], torch.tensor([-0.1, 0.05]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.3, -0.1], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert precomputed.grad is None


def test_fixed_opd_inputs_are_detached_in_persistent_rollout_data(monkeypatch):
    make_parallel_state()
    old_source = torch.tensor([0.2, 0.4], requires_grad=True)
    rollout_source = torch.tensor([0.3, 0.5], requires_grad=True)
    reference_source = torch.tensor([0.4, 0.6], requires_grad=True)
    teacher_source = torch.tensor([0.1, 0.2], requires_grad=True)
    rollout_data = {
        "log_probs": [old_source.sin()],
        "rollout_log_probs": [rollout_source.cos()],
        "ref_log_probs": [reference_source.exp()],
        "teacher_log_probs": [teacher_source.square()],
        "rewards": [0.0],
        "values": None,
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
    }
    args = Namespace(
        skip_actor_forward_only=False,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=0.5,
        normalize_advantages=False,
    )

    def fake_compute_advantages(**kwargs):
        assert kwargs["log_probs"][0].grad_fn is None
        zeros = torch.zeros_like(kwargs["log_probs"][0])
        return [zeros], [zeros.clone()]

    monkeypatch.setattr(loss_utils, "compute_advantages", fake_compute_advantages)

    loss_utils.compute_advantages_and_returns(args, rollout_data)

    for key in (
        "log_probs",
        "rollout_log_probs",
        "ref_log_probs",
        "teacher_log_probs",
        "opd_reverse_kl",
        "advantages",
    ):
        assert rollout_data[key][0].grad_fn is None
        assert rollout_data[key][0].requires_grad is False

    assert old_source.grad is None
    assert rollout_source.grad is None
    assert reference_source.grad is None
    assert teacher_source.grad is None


def test_noop_when_student_log_probs_none():
    args = _args()
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0, 0.0])]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, None)

    assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0]))
    assert "opd_reverse_kl" not in rollout_data


def test_raises_when_teacher_log_probs_missing():
    args = _args()
    with pytest.raises(ValueError, match="requires teacher_log_probs"):
        apply_opd_kl_to_advantages(args, {}, [torch.tensor([1.0])], [torch.tensor([1.0])])


def test_raises_on_length_mismatch():
    args = _args()
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0])]}  # 1 sample
    advantages = [torch.tensor([1.0]), torch.tensor([1.0])]  # 2 samples
    student = [torch.tensor([1.0]), torch.tensor([1.0])]

    with pytest.raises(ValueError, match="OPD length mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


def test_raises_on_scalar_advantage_broadcast_trap():
    # GRPO-style per-sample scalar advantage must be expanded to per-token first.
    args = _args()
    student = [torch.tensor([0.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0])]
    advantages = [torch.tensor([2.0])]  # shape (1,) != student shape (2,)
    rollout_data = {"teacher_log_probs": teacher}

    with pytest.raises(ValueError, match="OPD shape mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


def test_clip_bounds_each_per_token_contribution():
    """--opd-kl-clip caps the divergence a single token can contribute."""
    args = _args()
    args.opd_kl_clip = 1.0
    student = [torch.tensor([0.0, 5.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0, 0.0])]
    advantages = [torch.tensor([0.0, 0.0, 0.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # raw reverse_kl is [0, 5, 1]; the 5 is the heavy tail this exists to bound.
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.allclose(advantages[0], torch.tensor([0.0, -1.0, -1.0]))


def test_clip_records_how_often_it_binds():
    """A tau that never binds is inert and one that always binds flattens the signal."""
    args = _args()
    args.opd_kl_clip = 2.0
    student = [torch.tensor([0.0, 3.0, 9.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0, 0.0, 0.0])]
    advantages = [torch.zeros(4)]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # only the 3 and the 9 exceed tau=2
    assert torch.allclose(rollout_data["opd_kl_clipfrac"][0], torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_unset_clip_changes_nothing_and_reports_no_clipfrac():
    args = _args()
    args.opd_kl_clip = None
    student = [torch.tensor([0.0, 5.0])]
    teacher = [torch.tensor([0.0, 0.0])]
    advantages = [torch.zeros(2)]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 5.0]))
    assert "opd_kl_clipfrac" not in rollout_data


def _forward_kl_args(coef=1.0, clip=None):
    return Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=coef, opd_divergence="forward_kl", opd_kl_clip=clip)


def _forward_kl_batch(logits_row, ids, t_logprobs):
    """One sample, one response token; get_responses is stubbed so no megatron env is needed."""
    return {
        "teacher_top_ids": [torch.tensor(ids)],
        "teacher_top_logprobs": [torch.tensor(t_logprobs)],
        "unconcat_tokens": [torch.tensor([0, 1])],
        "total_lengths": [2],
        "response_lengths": [1],
    }, torch.tensor(logits_row)


def test_forward_kl_matches_the_closed_form(monkeypatch):
    """sum_v p_T(v) * (log p_T(v) - log p_S(v)) over the teacher's support."""
    from miles.backends.training_utils.loss_hub import opd as opd_mod

    # student logits over a 4-token vocab; teacher supports ids 0 and 1 with p=[0.75, 0.25]
    logits_row = [[0.0, 0.0, 0.0, 0.0]]  # uniform student -> log p_S = log(0.25) for every id
    t_logprobs = [[math.log(0.75), math.log(0.25)]]
    batch, logits_chunk = _forward_kl_batch(logits_row, [[0, 1]], t_logprobs)

    monkeypatch.setattr(opd_mod, "get_responses", lambda *a, **k: iter([(logits_chunk, None)]), raising=False)
    monkeypatch.setattr(
        "miles.backends.training_utils.loss_hub.logit_processors.get_responses",
        lambda *a, **k: iter([(logits_chunk, None)]),
    )

    loss, _, _ = opd_mod.forward_kl_loss(_forward_kl_args(), batch, logits_chunk, lambda x: x.mean())
    expected = 0.75 * (math.log(0.75) - math.log(0.25)) + 0.25 * (math.log(0.25) - math.log(0.25))
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_forward_kl_ignores_padded_support_entries(monkeypatch):
    """Padding carries -inf, so p_T is 0 and the entry must contribute nothing."""
    from miles.backends.training_utils.loss_hub import opd as opd_mod

    logits_row = [[0.0, 0.0, 0.0, 0.0]]
    t_logprobs = [[math.log(1.0), float("-inf")]]  # second slot is padding
    batch, logits_chunk = _forward_kl_batch(logits_row, [[0, 0]], t_logprobs)
    monkeypatch.setattr(
        "miles.backends.training_utils.loss_hub.logit_processors.get_responses",
        lambda *a, **k: iter([(logits_chunk, None)]),
    )
    loss, _, _ = opd_mod.forward_kl_loss(_forward_kl_args(), batch, logits_chunk, lambda x: x.mean())
    assert loss.item() == pytest.approx(math.log(1.0) - math.log(0.25), abs=1e-5)


def test_forward_kl_is_zero_when_student_matches_teacher(monkeypatch):
    from miles.backends.training_utils.loss_hub import opd as opd_mod

    logits_row = [[0.0, 0.0, 0.0, 0.0]]
    t_logprobs = [[math.log(0.25), math.log(0.25)]]
    batch, logits_chunk = _forward_kl_batch(logits_row, [[0, 1]], t_logprobs)
    monkeypatch.setattr(
        "miles.backends.training_utils.loss_hub.logit_processors.get_responses",
        lambda *a, **k: iter([(logits_chunk, None)]),
    )
    loss, _, _ = opd_mod.forward_kl_loss(_forward_kl_args(), batch, logits_chunk, lambda x: x.mean())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_forward_kl_raises_rather_than_silently_training_on_nothing():
    """Missing teacher support must fail loudly.

    Returning None here yielded loss=0, grad_norm=0 and no opd_forward_kl metric: a run
    that looks healthy and learns nothing.
    """
    from miles.backends.training_utils.loss_hub import opd as opd_mod

    with pytest.raises(ValueError, match="teacher_top_ids"):
        opd_mod.forward_kl_loss(_forward_kl_args(), {}, torch.zeros(1), lambda x: x)


def test_forward_kl_keeps_an_infinite_contribution_instead_of_zeroing_it(monkeypatch):
    """A student assigning ~zero mass where the teacher has some is the strongest signal.

    Masking padding with isfinite (rather than nan_to_num) is what preserves it; the clip
    is what bounds it.
    """
    from miles.backends.training_utils.loss_hub import opd as opd_mod

    # student puts almost nothing on id 0
    logits_row = [[-1e30, 0.0, 0.0, 0.0]]
    t_logprobs = [[math.log(1.0), float("-inf")]]  # teacher is certain about id 0
    batch, logits_chunk = _forward_kl_batch(logits_row, [[0, 0]], t_logprobs)
    monkeypatch.setattr(
        "miles.backends.training_utils.loss_hub.logit_processors.get_responses",
        lambda *a, **k: iter([(logits_chunk, None)]),
    )
    unclipped, _, _ = opd_mod.forward_kl_loss(_forward_kl_args(), batch, logits_chunk, lambda x: x.mean())
    assert unclipped.item() > 10.0, "a near-zero student probability must produce a large penalty"

    clipped, frac, _ = opd_mod.forward_kl_loss(_forward_kl_args(clip=2.0), batch, logits_chunk, lambda x: x.mean())
    assert clipped.item() == pytest.approx(2.0, abs=1e-5)
    assert frac.item() == pytest.approx(0.5, abs=1e-6), "one of the two slots clipped"

def test_forward_kl_reports_teacher_mass_inside_the_truncated_support(monkeypatch):
    """Coverage is what makes a too-small top-k, or a server-side cap, visible."""
    from miles.backends.training_utils.loss_hub import opd as opd_mod

    logits_row = [[0.0, 0.0, 0.0, 0.0]]
    # The support carries 0.90 of the teacher's mass; the padded slot must not count
    # toward it, or a short position would silently read as full coverage.
    t_logprobs = [[math.log(0.75), math.log(0.15), float("-inf")]]
    batch, logits_chunk = _forward_kl_batch(logits_row, [[0, 1, 0]], t_logprobs)
    monkeypatch.setattr(
        "miles.backends.training_utils.loss_hub.logit_processors.get_responses",
        lambda *a, **k: iter([(logits_chunk, None)]),
    )

    _, _, coverage = opd_mod.forward_kl_loss(_forward_kl_args(), batch, logits_chunk, lambda x: x.mean())
    assert coverage.item() == pytest.approx(0.90, abs=1e-6)
