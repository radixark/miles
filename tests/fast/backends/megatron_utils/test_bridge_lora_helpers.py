import torch

from miles.backends.megatron_utils.bridge_lora_helpers import (
    _add_parameterless_output_anchors,
    _remove_parameterless_output_anchors,
)


class _ParameterlessOutput(torch.nn.Module):
    pass


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.output_layer = _ParameterlessOutput()


def test_parameterless_output_anchor_provides_device_and_dtype_temporarily():
    model = _TiedModel()

    anchored = _add_parameterless_output_anchors(model, ["linear_qkv", "output_layer"])

    assert anchored == [model.output_layer]
    anchor = next(model.output_layer.parameters())
    assert anchor.numel() == 0
    assert anchor.device == model.embedding.weight.device
    assert anchor.dtype == model.embedding.weight.dtype

    _remove_parameterless_output_anchors(anchored)
    assert list(model.output_layer.parameters()) == []


def test_parameterless_output_anchor_is_not_added_when_unembed_is_not_targeted():
    model = _TiedModel()

    assert _add_parameterless_output_anchors(model, ["linear_qkv"]) == []
    assert list(model.output_layer.parameters()) == []
