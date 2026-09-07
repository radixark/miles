"""Fused RoPE is enforced at construction and at SelfAttention's call site."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from megatron.core.extensions import transformer_engine as te
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.transformer import attention

from miles_plugins.top import install, spec
from miles_plugins.top.rope import apply_fused_rope, enforce_fused_rope


def _config(**kwargs):
    return SimpleNamespace(
        **{
            "apply_rope_fusion": False,
            "rotary_interleaved": False,
            "mrope_section": None,
            "multi_latent_attention": False,
            **kwargs,
        }
    )


@pytest.fixture(autouse=True)
def restore_patch(monkeypatch):
    monkeypatch.setattr(attention, "apply_rotary_pos_emb", rope_utils.apply_rotary_pos_emb)
    monkeypatch.setattr(install, "_INSTALLED", {})


@pytest.mark.parametrize("num_experts", [None], ids=["qwen_dense"])
def test_spec_enforces_fusion_before_building_layers(monkeypatch, num_experts):
    from megatron.core.models.gpt import gpt_layer_specs

    args = SimpleNamespace(
        transformer_impl="local",
        num_experts=num_experts,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=False,
        normalization="RMSNorm",
        apply_rope_fusion=False,
    )
    config = _config(
        hidden_size=128,
        pipeline_hidden_size=None,
        fp32_residual_connection=False,
        hidden_dropout=0,
        add_bias_linear=False,
        use_kitchen=False,
        use_kitchen_attention=False,
        kitchen_attention_backend=None,
    )
    monkeypatch.setattr(spec, "_pin_sglang", lambda args: "true_on_policy_v1")
    monkeypatch.setattr(spec, "_hf_model_type", lambda args: "qwen3_moe" if num_experts else "qwen3")
    monkeypatch.setattr(spec, "_ACTIVE_PROGRAM", None)

    class ReachedBuilder(Exception):
        pass

    def build(**kwargs):
        assert args.apply_rope_fusion is True
        assert config.apply_rope_fusion is True
        assert attention.apply_rotary_pos_emb is apply_fused_rope
        raise ReachedBuilder

    monkeypatch.setattr(gpt_layer_specs, "get_gpt_layer_local_spec", build)
    with pytest.raises(ReachedBuilder):
        spec.get_top_spec(args, config, None)


def test_patch_is_idempotent_and_asserted(monkeypatch):
    assert install.install_fused_rope()
    assert install.install_fused_rope()
    assert attention.apply_rotary_pos_emb is apply_fused_rope
    monkeypatch.setattr(attention, "apply_rotary_pos_emb", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="unexpected SelfAttention call site"):
        install.install_fused_rope()


@pytest.mark.parametrize("symbol", ["fused_apply_rotary_pos_emb", "fused_apply_rotary_pos_emb_thd"])
def test_missing_te_fails_construction(monkeypatch, symbol):
    monkeypatch.setattr(te, symbol, None)
    with pytest.raises(RuntimeError, match="required TE fused RoPE implementation is unavailable"):
        enforce_fused_rope(SimpleNamespace(), _config())


@pytest.mark.parametrize("option", [{"mrope_section": [16, 16, 32]}, {"fused_single_qkv_rope": True}])
def test_unmatched_construction_options_raise(option):
    with pytest.raises(NotImplementedError):
        enforce_fused_rope(SimpleNamespace(), _config(**option))


def test_disabling_fusion_after_construction_raises():
    config = _config()
    enforce_fused_rope(SimpleNamespace(), config)
    config.apply_rope_fusion = False
    with pytest.raises(RuntimeError, match="disabled after TOP construction"):
        attention.apply_rotary_pos_emb(None, None, config)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "option",
    [
        {"mscale": 1.1},
        {"mla_rotary_interleaved": True},
        {"inverse": True},
        {"mla_output_remove_interleaving": True},
    ],
)
def test_unmatched_runtime_expression_raises(packed, option):
    with pytest.raises(NotImplementedError, match="not supported by the matched TE fused path"):
        apply_fused_rope(
            None, None, _config(apply_rope_fusion=True), cu_seqlens=object() if packed else None, **option
        )


@pytest.mark.parametrize("packed", [False, True])
def test_fused_failure_is_not_retried_unfused(monkeypatch, packed):
    def fail(*args, **kwargs):
        raise RuntimeError("TE failed")

    def refuse(*args, **kwargs):
        raise AssertionError("must never enter unfused RoPE")

    symbol = "fused_apply_rotary_pos_emb_thd" if packed else "fused_apply_rotary_pos_emb"
    monkeypatch.setattr(te, symbol, fail)
    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_bshd", refuse)
    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_thd", refuse)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    with pytest.raises(RuntimeError, match="TE failed"):
        apply_fused_rope(
            None, None, _config(apply_rope_fusion=True), cu_seqlens=object() if packed else None, cp_group=group
        )


def test_packed_rope_requires_explicit_cp_group():
    with pytest.raises(RuntimeError, match="caller's CP group"):
        apply_fused_rope(None, None, _config(apply_rope_fusion=True), cu_seqlens=object())
