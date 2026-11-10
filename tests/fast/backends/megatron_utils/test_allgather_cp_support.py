from types import SimpleNamespace

import pytest
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch import nn

from miles.backends.megatron_utils.parallel import verify_allgather_cp_support


class _Layer(TransformerLayer):
    """A ``TransformerLayer`` holding only what the check reads."""

    def __init__(self, attention: nn.Module, *, cp_size: int):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(context_parallel_size=cp_size)
        self.self_attention = attention


class _ZigzagAttention(nn.Module):
    pass


class _ContiguousAttention(nn.Module):
    supports_allgather_cp = True


class _ConfiguredAttention(nn.Module):
    def __init__(self, supports_allgather_cp: bool):
        super().__init__()
        self.supports_allgather_cp = supports_allgather_cp


def _model(*layers: nn.Module) -> list[nn.Module]:
    return [nn.ModuleDict({"decoder": nn.ModuleList(layers)})]


def test_rejects_zigzag_attention_and_names_the_layer() -> None:
    model = _model(_Layer(_ContiguousAttention(), cp_size=2), _Layer(_ZigzagAttention(), cp_size=2))

    with pytest.raises(ValueError, match=r"layer decoder\.1 uses _ZigzagAttention"):
        verify_allgather_cp_support(model, allgather_cp=True)


def test_rejects_layer_that_opts_out_per_instance() -> None:
    model = _model(_Layer(_ConfiguredAttention(supports_allgather_cp=False), cp_size=2))

    with pytest.raises(ValueError, match="_ConfiguredAttention"):
        verify_allgather_cp_support(model, allgather_cp=True)


def test_accepts_contiguous_attention_and_skips_layers_outside_the_cp_split() -> None:
    model = _model(
        _Layer(_ContiguousAttention(), cp_size=2),
        _Layer(_ConfiguredAttention(supports_allgather_cp=True), cp_size=2),
        _Layer(IdentityOp(), cp_size=2),
        # e.g. a vision tower, which runs with its own CP size of 1
        _Layer(_ZigzagAttention(), cp_size=1),
    )

    verify_allgather_cp_support(model, allgather_cp=True)


def test_zigzag_attention_is_fine_without_allgather_cp() -> None:
    verify_allgather_cp_support(_model(_Layer(_ZigzagAttention(), cp_size=2)), allgather_cp=False)


def test_empty_model_passes() -> None:
    verify_allgather_cp_support([], allgather_cp=True)
