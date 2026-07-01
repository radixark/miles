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
from miles.backends.experimental.fsdp_utils.update_weight_utils import _iter_sync_named_params


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


def test_iter_passthrough_for_non_expert():
    p = torch.zeros(4, 4)
    out = list(_iter_sync_named_params("model.embed_tokens.weight", p, "qwen3_moe"))
    assert len(out) == 1 and out[0][0] == "model.embed_tokens.weight" and out[0][1] is p
    # expert-named param under a model type that consumes batched layout -> passthrough
    g = torch.zeros(2, 6, 4)
    out = list(_iter_sync_named_params("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe"))
    assert len(out) == 1 and out[0][1] is g


def test_qwen3_moe_patch_noops_on_batched_structure():
    # On transformers>=5.6 the patch must not replace the forward (batched experts).
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    from miles.backends.experimental.fsdp_utils.models.qwen3_moe_hf import apply_fsdp_moe_patch

    original_forward = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward
    apply_fsdp_moe_patch()  # must not raise
    if hasattr(modeling_qwen3_moe, "Qwen3MoeExperts") or hasattr(modeling_qwen3_moe, "Qwen3MoeTopKRouter"):
        assert modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward is original_forward


def test_is_mamba_hybrid_gating():
    # The clobber-reload only runs for Mamba/SSM-hybrid archs (NemotronH _init_weights
    # re-inits dt_bias + out_proj post-load); it must be a no-op gate for everything else.
    from types import SimpleNamespace

    from miles.backends.experimental.fsdp_utils.adaptations.post_load_fixups import _is_mamba_hybrid

    assert _is_mamba_hybrid(SimpleNamespace(model_type="nemotron_h"))
    assert _is_mamba_hybrid(SimpleNamespace(model_type="mamba2"))
    # detected via layer_types even when model_type doesn't say "mamba"
    assert _is_mamba_hybrid(SimpleNamespace(model_type="hybrid", layer_types=["mamba", "attention"]))
    # dense / non-mamba archs must NOT trigger the reload
    assert not _is_mamba_hybrid(SimpleNamespace(model_type="qwen3"))
    assert not _is_mamba_hybrid(SimpleNamespace(model_type="qwen3_moe"))
    assert not _is_mamba_hybrid(SimpleNamespace(model_type="llama", layer_types=["attention"]))


def test_packed_seq_context_boundaries():
    # The shared boundary derivation (formerly duplicated verbatim in nemotron_h.py + qwen3_5_moe.py).
    from miles.backends.experimental.fsdp_utils.adaptations.packing.boundaries import packed_seq_context

    # single document / non-packed / wrong shape -> None (packing is a no-op)
    assert packed_seq_context(None) is None
    assert packed_seq_context(torch.arange(8).view(1, 8)) is None  # one doc, never resets to 0
    assert packed_seq_context(torch.arange(8)) is None  # not [1, T]
    assert packed_seq_context(torch.zeros(2, 4, dtype=torch.long)) is None  # batch > 1

    # three packed docs of length 3, 2, 4 -> position_ids reset to 0 at each start
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    ctx = packed_seq_context(pos)
    assert ctx is not None
    assert ctx.cu_seqlens.tolist() == [0, 3, 5, 9]
    assert ctx.cu_seqlens.dtype == torch.int32
    assert ctx.seq_idx.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]
    assert ctx.seq_idx.dtype == torch.int32
    assert ctx.seq_idx.shape == (1, 9)
    assert ctx.max_seqlen == 4
