from types import SimpleNamespace
from unittest.mock import patch

import torch

from miles_plugins.top.kernels import slot_combine


def _slots(*, tokens=9, topk=8, hidden=32, live=0.6, seed=0):
    """Routed slots with inactive ones (id -1); live slots index a permutation of the delivered rows."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 16, (tokens, topk), generator=g, dtype=torch.int32)
    ids[torch.rand(tokens, topk, generator=g) >= live] = -1
    active = ids >= 0
    num_rows = int(active.sum())
    index = torch.zeros(tokens, topk, dtype=torch.int32)
    index[active] = torch.randperm(num_rows, generator=g).to(torch.int32)
    rows = torch.randn(num_rows, hidden, generator=g).to(torch.bfloat16)
    weights = torch.rand(tokens, topk, generator=g)
    grad = torch.randn(tokens, hidden, generator=g).to(torch.bfloat16)
    return rows, weights, ids, index, grad


def test_backward_writes_each_live_row_once_and_keeps_one_scratch_row():
    """Inactive slots share one scratch row; the live rows' gradients are the slot products, unchanged."""
    rows, weights, ids, index, grad = _slots()
    ctx = SimpleNamespace(saved_tensors=(rows, weights, ids, index))
    drows, dweights, _, _ = slot_combine._SlotOrderCombine.backward(ctx, grad)

    active = ids >= 0
    expected = torch.zeros_like(rows)
    tokens, slots = active.nonzero(as_tuple=True)
    expected[index[active].long()] = (grad[tokens].float() * weights[tokens, slots, None]).to(rows.dtype)
    assert torch.equal(drows, expected)
    dots = (rows[index.long().clamp(min=0)].float() * grad[:, None, :].float()).sum(-1)
    torch.testing.assert_close(dweights, torch.where(active, dots, 0.0))
    # The gradient buffer holds the delivered rows plus one scratch row, not one per routed slot.
    assert drows.untyped_storage().nbytes() == (rows.shape[0] + 1) * rows.shape[1] * rows.element_size()


def _gather(rows, ids, weights, index, out):
    """ep_gather's FP32 slot-order sum, on CPU."""
    acc = torch.zeros(out.shape, dtype=torch.float32)
    for k in range(ids.shape[1]):
        live = ids[:, k] >= 0
        acc[live] += rows[index[live, k].long()].float() * weights[live, k, None]
    out.copy_(acc.to(out.dtype))


def test_forward_saves_a_dense_copy_of_the_padded_payload():
    """The transport returns rows as a view of its padded payload; saving the view kept the padding alive."""
    rows, weights, ids, index, _ = _slots()
    padded = torch.cat([rows, torch.full((rows.shape[0], 16), float("nan"), dtype=rows.dtype)], dim=1)
    padded.requires_grad_()
    with patch.object(slot_combine, "ep_gather", _gather):
        out = slot_combine._SlotOrderCombine.apply(padded[:, : rows.shape[1]], weights, ids, index)
    saved = out.grad_fn.saved_tensors[0]
    assert saved.is_contiguous() and torch.equal(saved, rows)
    assert saved.untyped_storage().data_ptr() != padded.untyped_storage().data_ptr()
