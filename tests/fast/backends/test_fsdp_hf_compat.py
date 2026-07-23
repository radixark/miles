"""Unit tests for the experimental-FSDP HF-compat fixes (CPU-only, no GPU/sglang).

Covers:
  * F9 weight-sync: batched MoE expert params are unfused through transformers' own
    reverse conversion (``revert_weight_conversion``) into the per-expert names
    SGLang expects, with the correct gate/up row split, contiguous tensors, and
    only for the right model types. These tests pin the HF revert output to the
    on-disk dialect; any upstream drift in the qwen2_moe conversion family or in
    per-tensor revert semantics turns them red.
"""

import logging

import pytest
import torch

from miles.backends.experimental.fsdp_utils.adaptations.weight_bridge import (
    _hf_unfuse_experts_expand,
)
from miles.backends.experimental.fsdp_utils.update_weight_utils import _iter_sync_named_params


@pytest.fixture(scope="module")
def tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        decoder_sparse_step=1,
        head_dim=4,
    )
    return Qwen3MoeForCausalLM(cfg)


def test_unfuse_gate_up_proj_rows_and_names(tiny_qwen3_moe):
    # [E=2, 2*inter=6, H=4]: fused rows are [gate(:3) | up(3:)]
    E, inter, H = 2, 3, 4
    full = torch.arange(E * 2 * inter * H, dtype=torch.float32).reshape(E, 2 * inter, H)
    out = dict(_hf_unfuse_experts_expand("model.layers.0.mlp.experts.gate_up_proj", full, tiny_qwen3_moe))

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


def test_unfuse_down_proj(tiny_qwen3_moe):
    E, H, inter = 2, 4, 3
    full = torch.arange(E * H * inter, dtype=torch.float32).reshape(E, H, inter)
    out = dict(_hf_unfuse_experts_expand("model.layers.5.mlp.experts.down_proj", full, tiny_qwen3_moe))
    assert set(out) == {
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.5.mlp.experts.1.down_proj.weight",
    }
    for e in range(E):
        d = out[f"model.layers.5.mlp.experts.{e}.down_proj.weight"]
        assert d.shape == (H, inter)
        torch.testing.assert_close(d, full[e])
        assert d.is_contiguous()


def test_unfuse_glm4_moe_lite_same_family():
    # glm4_moe_lite shares qwen3_moe's conversion family (qwen2_moe); its real batched
    # params must unfuse to the same per-expert dialect.
    from transformers import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

    cfg = Glm4MoeLiteConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        first_k_dense_replace=1,
        n_shared_experts=1,
        head_dim=4,
    )
    model = Glm4MoeLiteForCausalLM(cfg)
    name = "model.layers.1.mlp.experts.gate_up_proj"
    full = model.state_dict()[name]
    E, two_inter = full.shape[0], full.shape[1]
    out = dict(_hf_unfuse_experts_expand(name, full, model))
    assert set(out) == {
        f"model.layers.1.mlp.experts.{e}.{proj}.weight" for e in range(E) for proj in ("gate_proj", "up_proj")
    }
    for e in range(E):
        torch.testing.assert_close(
            out[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"], full[e, : two_inter // 2, :]
        )
        torch.testing.assert_close(out[f"model.layers.1.mlp.experts.{e}.up_proj.weight"], full[e, two_inter // 2 :, :])


def test_iter_passthrough_for_non_expert():
    # model=None proves the passthrough path never consumes the model
    p = torch.zeros(4, 4)
    out = list(_iter_sync_named_params("model.embed_tokens.weight", p, "qwen3_moe", model=None))
    assert len(out) == 1 and out[0][0] == "model.embed_tokens.weight" and out[0][1] is p
    # expert-named param under a model type that consumes batched layout -> passthrough
    g = torch.zeros(2, 6, 4)
    out = list(_iter_sync_named_params("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe", model=None))
    assert len(out) == 1 and out[0][1] is g


def test_reload_nemotron_h_clobbered_weights_behavior(tmp_path, caplog):
    from safetensors.torch import save_file

    from miles.backends.experimental.fsdp_utils.models.nemotron_h import reload_nemotron_h_clobbered_weights

    class Mixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dt_bias = torch.nn.Parameter(torch.tensor([1.0005]))
            self.out_proj = torch.nn.Linear(1, 1, bias=False)
            self.in_proj = torch.nn.Linear(1, 1, bias=False)
            self.out_proj.weight.data.fill_(2.0)
            self.in_proj.weight.data.fill_(3.0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer = torch.nn.Module()
            layer.mixer = Mixer()
            self.backbone = torch.nn.ModuleList([layer])

    model = Model()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert reload_nemotron_h_clobbered_weights(model, empty) == 0

    disk = {
        "backbone.0.mixer.dt_bias": torch.ones(1),
        "backbone.0.mixer.out_proj.weight": torch.ones(1, 1),
        "backbone.0.mixer.in_proj.weight": torch.ones(1, 1),
    }
    save_file(disk, tmp_path / "model.safetensors")
    assert reload_nemotron_h_clobbered_weights(model, tmp_path) == 1
    torch.testing.assert_close(model.backbone[0].mixer.dt_bias, torch.tensor([1.0005]))
    torch.testing.assert_close(model.backbone[0].mixer.out_proj.weight, torch.ones(1, 1))
    torch.testing.assert_close(model.backbone[0].mixer.in_proj.weight, torch.tensor([[3.0]]))

    with caplog.at_level(
        logging.INFO,
        logger="miles.backends.experimental.fsdp_utils.models.nemotron_h",
    ):
        assert reload_nemotron_h_clobbered_weights(model, tmp_path, tol=1e-4) == 1
    torch.testing.assert_close(model.backbone[0].mixer.dt_bias, torch.ones(1))
    torch.testing.assert_close(model.backbone[0].mixer.in_proj.weight, torch.tensor([[3.0]]))
    assert any(
        record.name == "miles.backends.experimental.fsdp_utils.models.nemotron_h"
        and "restored 1 NemotronH mixer parameter(s)" in record.getMessage()
        for record in caplog.records
    )


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


def test_nemotron_attention_reuses_precomputed_max_seqlen(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from miles.backends.experimental.fsdp_utils.models import nemotron_h

    flash_calls = {}
    flash_attn = ModuleType("flash_attn")

    def flash_attn_varlen_func(q, k, v, **kwargs):
        flash_calls.update(kwargs)
        return q

    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    class UnreadableCuSeqlens:
        def __getitem__(self, key):
            raise AssertionError("attention must not recompute max_seqlen from cu_seqlens")

    cu_seqlens = UnreadableCuSeqlens()
    ctx = SimpleNamespace(cu_seqlens=cu_seqlens, seq_idx=None, max_seqlen=3)
    monkeypatch.setattr(nemotron_h, "packed_seq_context", lambda position_ids: ctx)

    class DummyMixer(torch.nn.Module):
        pass

    class DummyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.q_proj = torch.nn.Identity()
            self.k_proj = torch.nn.Identity()
            self.v_proj = torch.nn.Identity()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden_states, *args, **kwargs):
            raise AssertionError("packed attention should use flash_attn_varlen_func")

    class DummyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyAttention()

        def forward(self, hidden_states, position_ids=None):
            return self.attn(hidden_states)

    nemotron_h._patch_attn_forward(DummyAttention)
    nemotron_h._patch_causallm_forward(DummyCausalLM, DummyMixer, DummyAttention)

    model = DummyCausalLM()
    output, _ = model(torch.ones(1, 3, 2), position_ids=torch.zeros(1, 3, dtype=torch.long))

    assert output.shape == (1, 3, 2)
    assert flash_calls["cu_seqlens_q"] is cu_seqlens
    assert flash_calls["max_seqlen_q"] == 3
    assert flash_calls["max_seqlen_k"] == 3
