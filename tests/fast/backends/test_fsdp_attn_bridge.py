from contextlib import nullcontext
from types import SimpleNamespace

import torch

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()


class _RecordingModelCls:
    def __init__(self):
        self.kwargs = None

    def from_pretrained(self, checkpoint_path, **kwargs):
        self.kwargs = dict(kwargs, checkpoint_path=checkpoint_path)
        return _StubModel()


def _actor(attn_implementation, model_cls):
    actor = object.__new__(FSDPTrainRayActor)
    actor.args = SimpleNamespace(attn_implementation=attn_implementation)
    actor._get_model_cls = lambda: model_cls
    return actor


def test_non_triton_attn_implementation_is_passed_through_unchanged():
    model_cls = _RecordingModelCls()
    actor = _actor("flash_attention_2", model_cls)

    _, patched = actor._build_model_with_attn_bridge("/ckpt", nullcontext)

    assert model_cls.kwargs["checkpoint_path"] == "/ckpt"
    assert model_cls.kwargs["attn_implementation"] == "flash_attention_2"
    assert patched == 0


def test_triton_is_ignored_off_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    model_cls = _RecordingModelCls()
    actor = _actor("triton", model_cls)

    _, patched = actor._build_model_with_attn_bridge("/ckpt", nullcontext)

    # Handed to from_pretrained unchanged, which rejects it — the behaviour a non-ROCm build
    # had before the bridge existed.
    assert model_cls.kwargs["attn_implementation"] == "triton"
    assert patched == 0


def test_triton_on_rocm_loads_eager_then_activates_the_bridge(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.setattr(torch.version, "hip", "6.0.0")
    previous = ALL_ATTENTION_FUNCTIONS.pop("triton", None)
    try:
        model_cls = _RecordingModelCls()
        actor = _actor("triton", model_cls)

        model, _ = actor._build_model_with_attn_bridge("/ckpt", nullcontext)

        # HF cannot construct with an implementation it does not know yet, so the model is
        # built eager and the bridge is registered and selected afterwards.
        assert model_cls.kwargs["attn_implementation"] == "eager"
        assert "triton" in ALL_ATTENTION_FUNCTIONS
        assert model.config._attn_implementation == "triton"
    finally:
        if previous is None:
            ALL_ATTENTION_FUNCTIONS.pop("triton", None)
        else:
            ALL_ATTENTION_FUNCTIONS["triton"] = previous
