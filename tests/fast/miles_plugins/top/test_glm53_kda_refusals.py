"""The GLM-5.3-Flash TOP KDA layer refuses trainer layouts its whole-sample program does not cover."""

from types import SimpleNamespace

import pytest

from miles_plugins.top.glm53_kda import TopGlm5NextKDAAttention


def _config(**overrides):
    fields = dict(tensor_model_parallel_size=1, sequence_parallel=False, context_parallel_size=1)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _hf():
    lin = dict(num_heads=2, head_dim=128, gate_lower_bound=-5.0, short_conv_kernel_size=4)
    return SimpleNamespace(linear_attn_config=lin, rms_norm_eps=1e-5, hidden_size=256)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(tensor_model_parallel_size=2), "TP1"),
        (dict(sequence_parallel=True), "sequence parallelism"),
        # The KDA core prefills whole samples on one rank: a CP piece would start from a state no rank holds.
        (dict(context_parallel_size=2), "context parallelism"),
    ],
)
def test_refuses_split_layouts(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        TopGlm5NextKDAAttention(_config(**overrides), layer_number=1, hf_config=_hf())


def test_refuses_unpacked_input():
    import torch

    layer = TopGlm5NextKDAAttention(_config(), layer_number=1, hf_config=_hf())
    with pytest.raises(ValueError, match="packed THD"):
        layer(torch.zeros(4, 2, 256))
