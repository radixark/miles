"""Unit tests for miles_plugins.models.minimax_m2.

Covers the new TP-aware ``_PerLayerRMSNorm`` wrapper:

- It builds the inner RMSNorm class with ``hidden_size = head_dim * num_heads``
  (the per-layer feature count for this TP rank, NOT just ``head_dim``).
- For TP=1: forward is unmodified -- numerically equal to running the inner
  RMSNorm directly on the flat ``[*, num_heads*head_dim]`` tensor (which is
  the HF reference semantic for per-layer QK-Norm).
- For TP>1: forward is wrapped with ``all_gather`` (along the last dim) and
  ``slice`` so each rank computes the variance from the FULL per-layer
  tensor, then keeps only its local slice. We simulate this with a
  monkey-patched ``_all_gather`` that concatenates two known tensors and
  verify the slice/result.
- Spec construction: ``get_minimax_m2_spec`` / ``get_minimax_m2_layer_spec``
  swap both q_layernorm and k_layernorm slots.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


def _fake_config(num_heads=48, num_kv=8, head_dim=128, hidden=3072,
                 num_layers=62, eps=1e-6, num_experts=256, moe_grouped_gemm=True):
    return types.SimpleNamespace(
        num_attention_heads=num_heads,
        num_query_groups=num_kv,
        kv_channels=head_dim,
        hidden_size=hidden,
        num_layers=num_layers,
        num_moe_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        layernorm_epsilon=eps,
        pipeline_model_parallel_layout=None,
        moe_layer_freq=None,
    )


# ---------------------------------------------------------------------------
# _PerLayerRMSNorm construction
# ---------------------------------------------------------------------------


class _FakeInnerNorm(nn.Module):
    """Stand-in for MCore's RMSNorm; just stores the hidden_size."""

    def __init__(self, hidden_size, config=None, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x32.to(input_dtype))


def test_per_layer_norm_built_with_full_hidden_size_q():
    """q_layernorm hidden = head_dim * num_attention_heads."""
    from miles_plugins.models.minimax_m2 import (
        _PerLayerRMSNorm,
        _PerLayerRMSNormExtraArgs,
    )

    extra = _PerLayerRMSNormExtraArgs(
        inner_cls=_FakeInnerNorm,
        num_heads=48,
        tp_group=None,
        tp_rank=0,
        tp_world_size=1,
    )
    obj = _PerLayerRMSNorm(hidden_size=128, extra=extra, eps=1e-6)

    assert isinstance(obj, _FakeInnerNorm)
    assert obj.hidden_size == 128 * 48 == 6144
    assert tuple(obj.weight.shape) == (6144,)


def test_per_layer_norm_built_with_full_hidden_size_k():
    """k_layernorm hidden = head_dim * num_key_value_heads."""
    from miles_plugins.models.minimax_m2 import (
        _PerLayerRMSNorm,
        _PerLayerRMSNormExtraArgs,
    )

    extra = _PerLayerRMSNormExtraArgs(
        inner_cls=_FakeInnerNorm,
        num_heads=8,
        tp_group=None,
        tp_rank=0,
        tp_world_size=1,
    )
    obj = _PerLayerRMSNorm(hidden_size=128, extra=extra, eps=1e-6)

    assert obj.hidden_size == 128 * 8 == 1024


# ---------------------------------------------------------------------------
# Numerical equivalence with HF reference (TP=1)
# ---------------------------------------------------------------------------


def test_per_layer_norm_tp1_matches_hf_rmsnorm():
    """At TP=1 the wrapper is just the inner norm with hidden=num_heads*head_dim."""
    from miles_plugins.models.minimax_m2 import (
        _PerLayerRMSNorm,
        _PerLayerRMSNormExtraArgs,
    )

    num_heads, head_dim = 4, 8
    extra = _PerLayerRMSNormExtraArgs(
        inner_cls=_FakeInnerNorm,
        num_heads=num_heads,
        tp_group=None,
        tp_rank=0,
        tp_world_size=1,
    )
    norm = _PerLayerRMSNorm(hidden_size=head_dim, extra=extra, eps=1e-6)
    nn.init.normal_(norm.weight, std=0.5)

    # Input shape as MCore would feed it post reshape-to-heads
    torch.manual_seed(0)
    sq, b = 3, 2
    x = torch.randn(sq, b, num_heads * head_dim, dtype=torch.float32)

    out = norm(x)

    # Reference: HF MiniMaxM2RMSNorm over num_heads*head_dim
    var = x.pow(2).mean(-1, keepdim=True)
    ref = norm.weight * (x * torch.rsqrt(var + 1e-6))
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# TP>1: forward wrapping
# ---------------------------------------------------------------------------


def test_per_layer_norm_tp_gt_1_wraps_forward_with_gather_and_slice():
    """Simulate TP=2 by patching _all_gather to splice in a fake remote shard.

    Implementation contract: ``num_heads`` in ``extra`` is the **global**
    head count. The inner norm is built with size ``head_dim * num_heads``
    on every TP rank. Each rank's input is ``[*, (num_heads/tp_world) * head_dim]``;
    all_gather concatenates across ranks to the full ``[*, num_heads*head_dim]``
    tensor, norm is applied, and the result is sliced back to the local share.
    """
    from miles_plugins.models import minimax_m2 as mm
    from miles_plugins.models.minimax_m2 import _wrap_forward_for_tp

    num_heads_global = 4
    head_dim = 8
    tp_world_size = 2
    local_head_count = num_heads_global // tp_world_size  # 2
    local_features = local_head_count * head_dim  # 16
    full_features = num_heads_global * head_dim  # 32

    extra = mm._PerLayerRMSNormExtraArgs(
        inner_cls=_FakeInnerNorm,
        num_heads=num_heads_global,
        tp_group="<fake-group>",
        tp_rank=0,
        tp_world_size=tp_world_size,
    )

    # Build the wrapped norm at the full size (matches what __new__ does:
    # hidden_size=head_dim, num_heads=4 -> inner built at 32).
    wrapped = _FakeInnerNorm(hidden_size=full_features, eps=1e-6)
    nn.init.normal_(wrapped.weight, std=0.5)
    full_inner_ref = _FakeInnerNorm(hidden_size=full_features, eps=1e-6)
    full_inner_ref.load_state_dict(wrapped.state_dict())

    _wrap_forward_for_tp(wrapped, hidden_size=head_dim, extra=extra)

    # Local-rank input
    torch.manual_seed(0)
    sq, b = 3, 2
    local = torch.randn(sq, b, local_features, dtype=torch.float32)
    remote = torch.randn(sq, b, local_features, dtype=torch.float32)

    def fake_all_gather(x, *, group, concat_dim):
        assert group == "<fake-group>"
        # Place local at rank 0, remote at rank 1
        return torch.concat([local, remote], dim=concat_dim)

    with patch("miles_plugins.models.minimax_m2._all_gather", side_effect=fake_all_gather):
        out = wrapped(local)

    # Reference: full per-layer norm on [local | remote], slice rank 0
    full = torch.concat([local, remote], dim=-1)
    full_normed = full_inner_ref(full)
    expected = torch.narrow(full_normed, dim=-1, start=0, length=local_features)

    assert out.shape == local.shape
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_slice_helper_picks_correct_chunk():
    from miles_plugins.models.minimax_m2 import _slice

    x = torch.arange(24).reshape(2, 12).float()
    # rank 1 of 3 -> [4:8]
    out = _slice(x, dim=-1, rank=1, num_slices=3)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, x[:, 4:8])


def test_slice_helper_requires_divisibility():
    from miles_plugins.models.minimax_m2 import _slice

    x = torch.zeros(2, 10)
    with pytest.raises(AssertionError):
        _slice(x, dim=-1, rank=0, num_slices=3)


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def _make_fake_block_spec(num_layers):
    from megatron.core.transformer.attention import SelfAttentionSubmodules

    class _LayerSubs:
        def __init__(self):
            sa = types.SimpleNamespace()
            sa.submodules = SelfAttentionSubmodules(
                linear_qkv=None,
                core_attention=None,
                linear_proj=None,
                q_layernorm="SENTINEL_Q",
                k_layernorm="SENTINEL_K",
            )
            self.self_attention = sa

    class _Layer:
        def __init__(self):
            self.submodules = _LayerSubs()

    class _Block:
        def __init__(self, n):
            self.layer_specs = [_Layer() for _ in range(n)]

    return _Block(num_layers)


def test_get_minimax_m2_spec_swaps_qk_norms_on_every_layer():
    from miles_plugins.models import minimax_m2 as mm

    fake_block = _make_fake_block_spec(num_layers=4)
    cfg = _fake_config(num_layers=4)
    args = types.SimpleNamespace(num_experts=256)

    with (
        patch.object(mm, "get_gpt_decoder_block_spec", return_value=fake_block),
        patch.object(mm, "get_num_layers_to_build", return_value=4),
    ):
        spec = mm.get_minimax_m2_spec(args, cfg, vp_stage=None)

    for i, layer in enumerate(spec.layer_specs):
        sub = layer.submodules.self_attention.submodules
        assert sub.q_layernorm.module is mm._PerLayerRMSNorm, f"layer {i} q_layernorm not wrapped"
        assert sub.k_layernorm.module is mm._PerLayerRMSNorm, f"layer {i} k_layernorm not wrapped"

        q_extra = sub.q_layernorm.params["extra"]
        k_extra = sub.k_layernorm.params["extra"]
        assert q_extra.num_heads == cfg.num_attention_heads
        assert k_extra.num_heads == cfg.num_query_groups
        # inner_cls is the SENTINEL we put in the fake spec
        assert q_extra.inner_cls == "SENTINEL_Q"
        assert k_extra.inner_cls == "SENTINEL_K"


def test_get_minimax_m2_layer_spec_returns_single_layer_with_swap():
    """The single-layer spec entry used by provider.transformer_layer_spec."""
    from miles_plugins.models import minimax_m2 as mm

    cfg = _fake_config()
    layer_spec = mm.get_minimax_m2_layer_spec(cfg)

    sub = layer_spec.submodules.self_attention.submodules
    assert sub.q_layernorm.module is mm._PerLayerRMSNorm
    assert sub.k_layernorm.module is mm._PerLayerRMSNorm
    assert sub.q_layernorm.params["extra"].num_heads == cfg.num_attention_heads
    assert sub.k_layernorm.params["extra"].num_heads == cfg.num_query_groups
