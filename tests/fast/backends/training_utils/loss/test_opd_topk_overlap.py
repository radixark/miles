"""Tests for the teacher/student top-k id-set overlap fraction.

overlap[r] = |student_topk_ids[r] ∩ teacher_topk_ids[r]| / K -- the distillation
health metric: distillation transfers through the student's top-k tokens, so
the teacher's own top-k must increasingly coincide with the student's as
training progresses (a flat/falling curve means the teacher view is shifting
the distribution instead of sharpening it).
"""

import pytest
import torch

from miles.backends.training_utils.loss_hub.opd import topk_overlap


def test_identical_ids_give_full_overlap():
    ids = torch.tensor([[1, 2, 3, 4], [9, 8, 7, 6]])
    out = topk_overlap(student_ids=ids, teacher_ids=ids.clone())
    assert torch.allclose(out, torch.ones(2))


def test_disjoint_ids_give_zero_overlap():
    s = torch.tensor([[1, 2, 3, 4]])
    t = torch.tensor([[5, 6, 7, 8]])
    assert torch.allclose(topk_overlap(student_ids=s, teacher_ids=t), torch.zeros(1))


def test_partial_overlap_is_order_invariant():
    s = torch.tensor([[1, 2, 3, 4]])
    t = torch.tensor([[4, 3, 99, 98]])  # {3, 4} shared, order scrambled
    assert torch.allclose(topk_overlap(student_ids=s, teacher_ids=t), torch.tensor([0.5]))


def test_per_position_rows_are_independent():
    s = torch.tensor([[1, 2], [1, 2], [1, 2]])
    t = torch.tensor([[1, 2], [2, 9], [8, 9]])
    out = topk_overlap(student_ids=s, teacher_ids=t)
    assert torch.allclose(out, torch.tensor([1.0, 0.5, 0.0]))


def test_chunking_matches_unchunked():
    g = torch.Generator().manual_seed(7)
    s = torch.randint(0, 50, (33, 8), generator=g)
    t = torch.randint(0, 50, (33, 8), generator=g)
    full = topk_overlap(student_ids=s, teacher_ids=t)
    chunked = topk_overlap(student_ids=s, teacher_ids=t, chunk_size=5)
    assert torch.allclose(full, chunked)


def test_empty_positions_give_empty_result():
    s = torch.zeros((0, 4), dtype=torch.long)
    out = topk_overlap(student_ids=s, teacher_ids=s.clone())
    assert out.shape == (0,)


def test_shape_mismatch_fails_loud():
    s = torch.tensor([[1, 2, 3]])
    t = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="shape"):
        topk_overlap(student_ids=s, teacher_ids=t)
