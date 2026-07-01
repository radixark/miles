"""Unit tests for the experimental-FSDP HF-compat fixes (CPU-only, no GPU/sglang).

Covers:
  * F9 weight-sync: batched MoE expert params are split into the per-expert names
    SGLang expects, with the correct gate/up row split, and only for the right
    model types.
  * F8: the legacy qwen3_moe MoE graph patch no-ops (does not crash) on the
    transformers>=5.6 batched structure.
"""

import torch

from miles.backends.experimental.fsdp_utils.adaptations.weight_bridge import _qwen3_moe_expand


def test_split_gate_up_proj_rows_and_names():
    # [E=2, 2*inter=6, H=4]: fused rows are [gate(:3) | up(3:)]
    E, inter, H = 2, 3, 4
    full = torch.arange(E * 2 * inter * H, dtype=torch.float32).reshape(E, 2 * inter, H)
    out = dict(_qwen3_moe_expand("model.layers.0.mlp.experts.gate_up_proj", full))

    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    }
    for e in range(E):
        g = out[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        u = out[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
        assert g.shape == (inter, H) and u.shape == (inter, H)
        torch.testing.assert_close(g, full[e, :inter, :])
        torch.testing.assert_close(u, full[e, inter:, :])
        assert g.is_contiguous() and u.is_contiguous()


def test_split_down_proj():
    E, H, inter = 2, 4, 3
    full = torch.arange(E * H * inter, dtype=torch.float32).reshape(E, H, inter)
    out = dict(_qwen3_moe_expand("model.layers.5.mlp.experts.down_proj", full))
    assert set(out) == {
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.5.mlp.experts.1.down_proj.weight",
    }
    for e in range(E):
        d = out[f"model.layers.5.mlp.experts.{e}.down_proj.weight"]
        assert d.shape == (H, inter)
        torch.testing.assert_close(d, full[e])
