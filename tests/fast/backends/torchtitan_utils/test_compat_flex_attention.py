"""Every micro-batch, in the log-prob pass and in training, must run the same flex_attention kernel."""

import sys
from types import ModuleType

import torch

from miles.backends.torchtitan_utils import compat


def _fake_torchtitan(monkeypatch, flex_attention_class):
    attention = ModuleType("torchtitan.models.common.attention")
    attention.FlexAttention = flex_attention_class
    for name in ("torchtitan", "torchtitan.models", "torchtitan.models.common"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, attention.__name__, attention)


def test_flex_attention_is_compiled_once_for_dynamic_shapes(monkeypatch):
    from torch.nn.attention.flex_attention import flex_attention

    options = {"max_autotune": True}
    flex_attention_class = type("FlexAttention", (), {"inductor_configs": options, "_compiled_flex_attn": object()})
    _fake_torchtitan(monkeypatch, flex_attention_class)
    compiles = []
    monkeypatch.setattr(
        torch, "compile", lambda fn, **kwargs: compiles.append((fn, kwargs)) or (lambda *a, **k: "out")
    )

    compat._compile_flex_attention_for_dynamic_shapes()
    compat._compile_flex_attention_for_dynamic_shapes()  # installed once

    assert compiles == [(flex_attention, {"dynamic": True, "options": options})]
    assert flex_attention_class._compiled_flex_attn(1, 2, 3) == "out"
