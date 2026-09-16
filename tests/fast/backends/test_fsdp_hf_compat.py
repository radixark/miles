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
import subprocess
import sys

import pytest
import torch

from miles.backends.fsdp_utils.adaptations.weight_bridge import _hf_unfuse_experts_expand, get_param_transform
from miles.backends.fsdp_utils.update_weight_utils import _iter_sync_named_params


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


def test_param_transform_gating():
    def applies(name, param, model_type):
        return get_param_transform(name, param, model_type) is not None

    gate_up = torch.zeros(2, 6, 4)
    name = "model.layers.0.mlp.experts.gate_up_proj"
    # only for model types whose SGLang loader expects per-expert weights
    assert applies(name, gate_up, "qwen3_moe")
    assert not applies(name, gate_up, "qwen3_5_moe")  # consumes batched directly
    assert not applies(name, gate_up, "qwen3")  # dense
    # non-expert params are never split
    assert not applies("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4), "qwen3_moe")
    # 2D tensor named like an expert param is not the batched layout
    assert not applies(name, torch.zeros(6, 4), "qwen3_moe")


def test_iter_passthrough_for_non_expert():
    # model=None proves the passthrough path never consumes the model
    p = torch.zeros(4, 4)
    out = list(_iter_sync_named_params("model.embed_tokens.weight", p, "qwen3_moe", model=None))
    assert len(out) == 1 and out[0][0] == "model.embed_tokens.weight" and out[0][1] is p
    # expert-named param under a model type that consumes batched layout -> passthrough
    g = torch.zeros(2, 6, 4)
    out = list(_iter_sync_named_params("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe", model=None))
    assert len(out) == 1 and out[0][1] is g


def test_nemotron_h_post_load_fixup_gating():
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.post_load_fixups import _FIXUPS

    by_name = {f.name: f for f in _FIXUPS}
    fixup = by_name["nemotron_h_clobber_reload"]
    assert fixup.applies_to(SimpleNamespace(model_type="nemotron_h"))
    assert not fixup.applies_to(SimpleNamespace(model_type="mamba2"))
    assert not fixup.applies_to(SimpleNamespace(model_type="hybrid", layer_types=["mamba", "attention"]))
    assert not fixup.applies_to(SimpleNamespace(model_type="qwen3_moe"))


def test_post_load_fixup_lazily_loads_model_implementation():
    script = """
import sys
import tempfile
from types import SimpleNamespace

from miles.backends.fsdp_utils.adaptations.post_load_fixups import (
    _FIXUPS,
    apply_post_load_fixups,
)

module_name = "miles.backends.fsdp_utils.models.nemotron_h"
assert [fixup.name for fixup in _FIXUPS].count("nemotron_h_clobber_reload") == 1
assert module_name not in sys.modules
apply_post_load_fixups(object(), SimpleNamespace(model_type="mamba2"), ".")
assert module_name not in sys.modules
with tempfile.TemporaryDirectory() as ckpt_path:
    apply_post_load_fixups(object(), SimpleNamespace(model_type="nemotron_h"), ckpt_path)
assert module_name in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reload_nemotron_h_clobbered_weights_behavior(tmp_path, caplog):
    from safetensors.torch import save_file

    from miles.backends.fsdp_utils.models.nemotron_h import reload_nemotron_h_clobbered_weights

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
        logger="miles.backends.fsdp_utils.models.nemotron_h",
    ):
        assert reload_nemotron_h_clobbered_weights(model, tmp_path, tol=1e-4) == 1
    torch.testing.assert_close(model.backbone[0].mixer.dt_bias, torch.ones(1))
    torch.testing.assert_close(model.backbone[0].mixer.in_proj.weight, torch.tensor([[3.0]]))
    assert any(
        record.name == "miles.backends.fsdp_utils.models.nemotron_h"
        and "restored 1 NemotronH mixer parameter(s)" in record.getMessage()
        for record in caplog.records
    )


def test_weight_bridge_registry():
    # The WeightBridge registry is the train->rollout param-name/shape contract: a model type with
    # a registered transform gets its params rewritten; unregistered types stream verbatim.
    import torch

    from miles.backends.fsdp_utils.adaptations.weight_bridge import get_param_transform, register_param_transform

    # qwen3_moe is registered (batched experts -> per-expert); a 3D experts param matches.
    g = torch.zeros(2, 6, 4)
    assert get_param_transform("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_moe") is not None
    # unregistered model type -> no transform (passthrough)
    assert get_param_transform("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe") is None
    # registering a new transform routes matching params through it
    register_param_transform(
        "_test_arch",
        matches=lambda name, p: name.endswith(".foo"),
        expand=lambda name, full: [(name.replace(".foo", ".bar"), full)],
    )
    fn = get_param_transform("x.foo", g, "_test_arch")
    assert fn is not None and list(fn("x.foo", g))[0][0] == "x.bar"
    assert get_param_transform("x.baz", g, "_test_arch") is None


def test_model_patch_registry_gating():
    # The ModelPatchHook registry replaces the hardcoded per-arch dispatch in apply_class_patches.
    # Verify the config-check predicates gate correctly. Packed-sequence layout patches (GDN, ...) moved
    # out of this registry into the unified packing registry (test_packing_registry below);
    # apply_class_patches now dispatches them via apply_packing.
    from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS

    by_name = {h.name: h for h in _MODEL_PATCH_HOOKS}
    # the expected generic hooks are registered in order (GDN packing is not a ModelPatchHook)
    assert [h.name for h in _MODEL_PATCH_HOOKS][:3] == [
        "fp8_checkpoint_guard",
        "dsa_train_infer_warn",
        "model_type_verified",
    ]
    assert [h.name for h in _MODEL_PATCH_HOOKS].count("model_type_verified") == 1
    assert "gated_deltanet_packing" not in by_name
    assert not by_name["fp8_checkpoint_guard"].applies_to(None)
    # the qwen3_moe MoE-block patch is a hook now (moved out of _enable_true_on_policy_optimizations),
    # gated on model_type; the backend-level enable_batch_invariant_mode stays in the actor.
    from types import SimpleNamespace

    assert "qwen3_moe_moe_patch" in by_name
    assert by_name["qwen3_moe_moe_patch"].applies_to(SimpleNamespace(model_type="qwen3_moe"))
    assert not by_name["qwen3_moe_moe_patch"].applies_to(SimpleNamespace(model_type="qwen3"))
    # Batched experts need no off-mode patch under the pinned transformers version.
    by_name["qwen3_moe_moe_patch"].apply(
        SimpleNamespace(model_type="qwen3_moe"), SimpleNamespace(true_on_policy_mode=False)
    )


@pytest.mark.parametrize("model_type", ["glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"])
def test_model_type_verified_accepts_recorded_models(model_type, caplog):
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.class_patches import check_model_type_verified

    check_model_type_verified(SimpleNamespace(model_type=model_type), SimpleNamespace(rank=0))

    assert not caplog.records


def test_verified_model_types_match_recorded_validation():
    from miles.backends.fsdp_utils.adaptations.class_patches import VERIFIED_MODEL_TYPES

    assert VERIFIED_MODEL_TYPES == frozenset({"glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"})


def test_model_type_verified_warns_once_on_rank_zero(caplog):
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS

    hook = next(h for h in _MODEL_PATCH_HOOKS if h.name == "model_type_verified")
    hook.apply(SimpleNamespace(model_type="qwen3_5_moe"), SimpleNamespace(rank=0))

    assert len(caplog.records) == 1
    assert "model_type='qwen3_5_moe' has no recorded FSDP validation" in caplog.text


def test_model_type_verified_is_silent_on_nonzero_rank(caplog):
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS

    hook = next(h for h in _MODEL_PATCH_HOOKS if h.name == "model_type_verified")
    hook.apply(SimpleNamespace(model_type="qwen3_5_moe"), SimpleNamespace(rank=1))

    assert not caplog.records


def test_packed_seq_context_boundaries():
    # The shared boundary derivation (formerly duplicated verbatim in nemotron_h.py + qwen3_5_moe.py).
    from miles.backends.fsdp_utils.adaptations.packing.boundaries import packed_seq_context

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

    from miles.backends.fsdp_utils.models import nemotron_h

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


def test_nemotron_pattern_to_list_repair():
    # sglang's import monkeypatches NemotronHConfig._pattern_to_list to drop unmapped chars, deleting
    # every '-' (MLP) layer; the config-time repair must restore the full mapping, and must leave a
    # healthy implementation untouched.
    from types import SimpleNamespace

    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    from miles.backends.fsdp_utils.adaptations.specs.nemotron_h import _repair_pattern_to_list

    hf_config = SimpleNamespace(model_type="nemotron_h")
    original = NemotronHConfig.__dict__["_pattern_to_list"]
    healthy = staticmethod(
        lambda pattern: [{"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}[c] for c in pattern]
    )
    broken = staticmethod(
        lambda pattern: [{"M": "mamba", "E": "moe", "*": "attention"}[c] for c in pattern if c in "ME*"]
    )
    try:
        NemotronHConfig._pattern_to_list = healthy
        _repair_pattern_to_list(hf_config, None)
        assert NemotronHConfig.__dict__["_pattern_to_list"] is healthy

        NemotronHConfig._pattern_to_list = broken
        _repair_pattern_to_list(hf_config, None)
        assert NemotronHConfig._pattern_to_list("M-*E") == ["mamba", "mlp", "attention", "moe"]
    finally:
        NemotronHConfig._pattern_to_list = original


def test_packing_registry():
    # The unified packing registry dispatches per (model_type, lifetime); GDN is config-lifetime,
    # NemotronH is post-load-lifetime, and archs that pack natively / don't pack match nothing.
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.packing import get_packing_patches

    gdn = SimpleNamespace(model_type="qwen3_5_moe", layer_types=["linear_attention", "full_attention"])
    nemo = SimpleNamespace(model_type="nemotron_h")
    glm = SimpleNamespace(model_type="glm4_moe_lite", layer_types=["full_attention"])
    dense = SimpleNamespace(model_type="qwen3", layer_types=["full_attention"])

    def names(cfg, lifetime):
        return {p.name for p in get_packing_patches(cfg, lifetime)}

    # GatedDeltaNet: config lifetime only
    assert names(gdn, "config") == {"gated_deltanet_packing"}
    assert names(gdn, "post_load") == set()
    # NemotronH: post-load lifetime only
    assert names(nemo, "post_load") == {"nemotron_h_packing"}
    assert names(nemo, "config") == set()
    # glm4_moe_lite (native MLA varlen) and dense qwen3: no packing patch at either lifetime
    for cfg in (glm, dense, None):
        assert names(cfg, "config") == set()
        assert names(cfg, "post_load") == set()
