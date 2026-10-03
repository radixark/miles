"""`select_indexer_topk` and `freeze_indexer_parameters` are what keep every
indexer frozen and on one selection width."""

import pytest
import torch

from miles_plugins.models.indexer import (
    EMPTY_SLOT,
    freeze_indexer_parameters,
    get_indexer_topk_fn,
    select_indexer_topk,
)


def test_selection_pads_to_a_fixed_width_when_keys_are_scarce():
    """A short sequence returning a narrower tensor would break the sparse
    attention kernels, which are built for one width."""
    indices = select_indexer_topk(torch.randn(4, 3), 8)

    assert indices.shape == (4, 8)
    assert indices.dtype == torch.int32
    assert (indices == EMPTY_SLOT).sum().item() == 4 * 5


def test_selection_marks_masked_keys_as_empty():
    """A key the query must not see has to leave as -1, not as its index."""
    logits = torch.randn(2, 6)
    logits[:, 4:] = -torch.inf

    indices = select_indexer_topk(logits, 6)

    assert (indices == EMPTY_SLOT).sum().item() == 4
    assert (indices[indices != EMPTY_SLOT] < 4).all()


def test_selection_keeps_leading_dimensions():
    """The flashinfer path reshapes to 2D internally and has to restore shape."""
    assert select_indexer_topk(torch.randn(3, 5, 64), 16).shape == (3, 5, 16)


def test_selection_builds_no_autograd_graph():
    """The indexer is frozen everywhere; a graph here would train it by accident."""
    hidden = torch.randn(4, 32, requires_grad=True)

    indices = select_indexer_topk(hidden * 2, 8)

    assert not indices.requires_grad
    assert indices.grad_fn is None


def test_selection_rejects_an_unknown_backend():
    """A misspelled backend must fail, not silently fall back to torch."""
    with pytest.raises(ValueError, match="backend"):
        select_indexer_topk(torch.randn(2, 8), 4, backend="cudnn")
    with pytest.raises(ValueError, match="backend"):
        get_indexer_topk_fn("cudnn")


def test_selection_can_return_the_scores_it_selected_on():
    """GLM-5.3's expand kernel reads these back, so they must line up with the
    indices they came from."""
    logits = torch.randn(4, 32)

    scores, indices = select_indexer_topk(logits, 8, return_scores=True)

    assert scores.shape == indices.shape == (4, 8)
    torch.testing.assert_close(scores, logits.gather(1, indices.long()))


def test_returned_scores_are_padded_with_negative_infinity():
    """A padded slot scoring anything finite would read as a real pick."""
    scores, indices = select_indexer_topk(torch.randn(2, 3), 5, return_scores=True)

    assert torch.isneginf(scores[:, 3:]).all()
    assert (indices[:, 3:] == EMPTY_SLOT).all()


def _module_with_indexer() -> torch.nn.Module:
    module = torch.nn.Module()
    module.indexer = torch.nn.Linear(4, 4)
    module.backbone = torch.nn.Linear(4, 4)
    return module


def test_freezing_clears_requires_grad_on_indexer_parameters_only():
    """Freezing the backbone too would silently stop the model training."""
    module = _module_with_indexer()

    frozen = freeze_indexer_parameters(module, ("indexer.*",))

    assert set(frozen) == {"indexer.weight", "indexer.bias"}
    assert not module.indexer.weight.requires_grad
    assert module.backbone.weight.requires_grad


def test_freezing_raises_when_the_globs_match_nothing():
    """A renamed parameter must fail loudly; otherwise the indexer keeps training."""
    with pytest.raises(ValueError, match="matched no parameter"):
        freeze_indexer_parameters(_module_with_indexer(), ("self_attention.indexer.*",))


def test_freezing_tolerates_an_empty_match_when_asked():
    """A pipeline stage holding no indexer layer legitimately matches nothing."""
    assert freeze_indexer_parameters(_module_with_indexer(), ("absent.*",), expect_match=False) == ()
