"""Tests for the in-trainer OPD top-k reverse KL
with pointwise clipping and fail-loud validation.

rkl[r] = sum_k exp(s_k) * (s_k - t_k)   -- s = student full-vocab-normalized
logprobs at the student's own top-k ids, t = teacher logprobs gathered at the
SAME ids; NOT renormalized within the top-k.
"""

import math

import pytest
import torch

from miles.backends.training_utils.loss_hub.opd import topk_reverse_kl


def test_identical_scores_give_zero_kl():
    s = torch.log(torch.tensor([[0.5, 0.3, 0.1], [0.6, 0.2, 0.1]]))
    rkl, clipfrac = topk_reverse_kl(student_vals=s, teacher_vals=s.clone(), pointwise_clip=0.0)
    assert torch.allclose(rkl, torch.zeros(2), atol=1e-6)
    assert torch.equal(clipfrac, torch.zeros(2))


def test_hand_computed_value():
    s = torch.log(torch.tensor([[0.5, 0.25]]))
    t = torch.log(torch.tensor([[0.25, 0.5]]))
    expected = 0.5 * (math.log(0.5) - math.log(0.25)) + 0.25 * (math.log(0.25) - math.log(0.5))
    rkl, _ = topk_reverse_kl(student_vals=s, teacher_vals=t, pointwise_clip=0.0)
    assert torch.allclose(rkl, torch.tensor([expected]), atol=1e-6)


def test_pointwise_clip_bounds_contributions_and_reports_fraction():
    s = torch.log(torch.tensor([[0.9, 0.05]]))
    t = torch.tensor([[-30.0, math.log(0.05)]])  # teacher hates the student's top choice
    unclipped, _ = topk_reverse_kl(student_vals=s, teacher_vals=t, pointwise_clip=0.0)
    clipped, clipfrac = topk_reverse_kl(student_vals=s, teacher_vals=t, pointwise_clip=1.0)
    assert clipped.item() < unclipped.item()
    assert clipped.item() <= 1.0 * s.shape[-1] + 1e-6  # each contribution capped at 1.0
    assert clipfrac.item() == pytest.approx(0.5)  # 1 of 2 entries clipped


def test_nonfinite_teacher_fails_loud():
    s = torch.log(torch.tensor([[0.5, 0.5]]))
    t = torch.tensor([[float("nan"), -1.0]])
    with pytest.raises(ValueError, match="finite"):
        topk_reverse_kl(student_vals=s, teacher_vals=t, pointwise_clip=0.0)


def test_shape_mismatch_fails_loud():
    s = torch.zeros(2, 3)
    t = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="shape"):
        topk_reverse_kl(student_vals=s, teacher_vals=t, pointwise_clip=0.0)
