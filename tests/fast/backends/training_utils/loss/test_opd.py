"""Unit tests for the decoupled on-policy-distillation (OPD) loss path.

`apply_opd_kl_to_advantages` is orthogonal to the advantage estimator: it adds a
reverse-KL penalty (student_logp - teacher_logp) to per-token advantages. These
tests cover the math and the guard rails without needing the external loss
snapshot artifacts.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


def _args(opd_kl_coef: float = 1.0) -> Namespace:
    return Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=opd_kl_coef)


def test_subtracts_weighted_reverse_kl_and_stores_metric():
    args = _args(opd_kl_coef=0.5)
    student = [torch.tensor([0.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0])]
    advantages = [torch.tensor([2.0, 2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # reverse_kl = student - teacher = [0, 1]; adv - 0.5 * reverse_kl = [2.0, 1.5]
    assert torch.allclose(advantages[0], torch.tensor([2.0, 1.5]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 1.0]))


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


# --------------------------------------------------------------------------- #
# Precomputed reverse-KL branch (cross-tokenizer OPD path).
#
# Cross-tokenizer OPD (miles/rollout/cross_tokenizer_opd.py) DPCA-aligns the
# student and teacher tokenizations at rollout time and stores a per-token
# reverse-KL directly in rollout_data["opd_reverse_kl"]. The loss must consume
# that tensor and ignore teacher_log_probs. The value reduces to
# (student_logp - teacher_logp) when the tokenizers agree, so these tests pin the
# generalized branch the same way the tests above pin the shared-tokenizer one.
# --------------------------------------------------------------------------- #


def test_precomputed_reverse_kl_is_used_and_ignores_teacher():
    args = _args(opd_kl_coef=0.5)
    advantages = [torch.tensor([2.0, 2.0, 2.0])]
    student = [torch.tensor([-1.0, -1.0, -1.0])]  # must be non-None to apply the penalty
    rollout_data = {
        "opd_reverse_kl": [torch.tensor([0.2, -0.5, 1.0])],
        "teacher_log_probs": [torch.tensor([99.0, 99.0, 99.0])],  # must be ignored
    }

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # adv - 0.5 * reverse_kl
    assert torch.allclose(advantages[0], torch.tensor([1.9, 2.25, 1.5]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.2, -0.5, 1.0]))


def test_precomputed_reverse_kl_accepts_python_lists():
    # post_process_rewards stores opd_reverse_kl as a plain python list per sample.
    args = _args(opd_kl_coef=1.0)
    advantages = [torch.tensor([1.0, 1.0])]
    student = [torch.tensor([0.0, 0.0])]
    rollout_data = {"opd_reverse_kl": [[0.25, -0.25]]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    assert torch.allclose(advantages[0], torch.tensor([0.75, 1.25]))
    assert torch.is_tensor(rollout_data["opd_reverse_kl"][0])


def test_precomputed_reverse_kl_noop_when_student_log_probs_none():
    # The penalty is only applied when student log-probs are available.
    args = _args()
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"opd_reverse_kl": [torch.tensor([1.0, 1.0])]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, None)

    assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0]))


def test_precomputed_reverse_kl_length_mismatch_raises():
    args = _args()
    advantages = [torch.tensor([1.0]), torch.tensor([1.0])]  # 2 samples
    student = [torch.tensor([1.0]), torch.tensor([1.0])]
    rollout_data = {"opd_reverse_kl": [torch.tensor([1.0])]}  # 1 sample

    with pytest.raises(ValueError, match="OPD length mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


def test_precomputed_reverse_kl_shape_mismatch_raises():
    args = _args()
    advantages = [torch.tensor([1.0, 2.0])]  # (2,)
    student = [torch.tensor([0.0, 0.0])]
    rollout_data = {"opd_reverse_kl": [torch.tensor([1.0])]}  # (1,)

    with pytest.raises(ValueError, match="OPD shape mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)
