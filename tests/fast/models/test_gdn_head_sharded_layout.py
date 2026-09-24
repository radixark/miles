"""Parameter-layout contracts of the head-sharded GDN core (CPU only).

The Megatron module keeps the HF ``linear_attn`` parameter names but stores Qwen3.5's
``in_proj_qkv.weight`` and both models' ``conv1d.weight`` head-interleaved, so a contiguous TP chunk
is a head shard.  These tests pin the HF <-> Megatron translation, the TP shard/merge round trip for
both HF layouts, and the two converter paths (megatron_to_hf direct converters and the mbridge
plugins) against the pure layout helpers.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from miles_plugins.models.gdn_attention import (  # noqa: E402
    GdnLayout,
    deinterleave_qkv_rows,
    hf_linear_attn_to_local,
    hf_to_megatron_linear_attn,
    interleave_qkv_rows,
    local_to_hf_linear_attn,
    megatron_to_hf_linear_attn,
)


def _layout(hf_layout: str, num_k_heads: int = 4, group: int = 2) -> GdnLayout:
    return GdnLayout(
        hidden_size=64,
        num_k_heads=num_k_heads,
        num_v_heads=num_k_heads * group,
        head_k_dim=16,
        head_v_dim=8,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
        hf_layout=hf_layout,
    )


def _hf_state(layout: GdnLayout, gen: torch.Generator) -> dict[str, torch.Tensor]:
    def rn(*shape):
        return torch.randn(*shape, generator=gen)

    hf = {
        "A_log": rn(layout.num_v_heads),
        "dt_bias": rn(layout.num_v_heads),
        "conv1d.weight": rn(layout.conv_dim, 1, layout.conv_kernel_size),
        "norm.weight": rn(layout.head_v_dim),
        "out_proj.weight": rn(layout.hidden_size, layout.value_dim),
    }
    if layout.hf_layout == "qwen3_next":
        hf["in_proj_qkvz.weight"] = rn(
            layout.num_k_heads * (2 * layout.head_k_dim + 2 * layout.group * layout.head_v_dim), layout.hidden_size
        )
        hf["in_proj_ba.weight"] = rn(layout.num_k_heads * 2 * layout.group, layout.hidden_size)
    else:
        hf["in_proj_qkv.weight"] = rn(layout.conv_dim, layout.hidden_size)
        hf["in_proj_z.weight"] = rn(layout.value_dim, layout.hidden_size)
        hf["in_proj_b.weight"] = rn(layout.num_v_heads, layout.hidden_size)
        hf["in_proj_a.weight"] = rn(layout.num_v_heads, layout.hidden_size)
    return hf


def test_interleave_groups_rows_per_key_head():
    layout = _layout("qwen3_5", num_k_heads=2, group=2)
    flat = torch.arange(layout.conv_dim).unsqueeze(1).float()  # row index as value
    inter = interleave_qkv_rows(layout, flat).squeeze(1).long().tolist()
    hk, hv, g, kd = layout.head_k_dim, layout.head_v_dim, layout.group, layout.key_dim
    expected = []
    for h in range(layout.num_k_heads):
        expected += list(range(h * hk, (h + 1) * hk))  # q_h
        expected += list(range(kd + h * hk, kd + (h + 1) * hk))  # k_h
        expected += list(range(2 * kd + h * g * hv, 2 * kd + (h + 1) * g * hv))  # v_{h,*}
    assert inter == expected
    assert torch.equal(deinterleave_qkv_rows(layout, interleave_qkv_rows(layout, flat)), flat)


@pytest.mark.parametrize("hf_layout", ["qwen3_5", "qwen3_next"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_hf_local_round_trip_and_head_sharding(hf_layout, tp_size):
    layout = _layout(hf_layout, num_k_heads=4, group=2)
    hf = _hf_state(layout, torch.Generator().manual_seed(0))
    shards = [hf_linear_attn_to_local(layout, hf, tp_rank=r, tp_size=tp_size) for r in range(tp_size)]
    merged = local_to_hf_linear_attn(layout, shards)
    assert set(merged) == set(hf)
    for name in hf:
        assert torch.equal(merged[name], hf[name]), name

    # Every rank's shard holds exactly its key-head group's rows (a real head shard, not a flat slice).
    lk, lv = layout.num_k_heads // tp_size, layout.num_v_heads // tp_size
    for r, shard in enumerate(shards):
        assert shard["A_log"].tolist() == hf["A_log"][r * lv : (r + 1) * lv].tolist()
        assert torch.equal(
            shard["out_proj.weight"],
            hf["out_proj.weight"][:, r * lv * layout.head_v_dim : (r + 1) * lv * layout.head_v_dim],
        )
        conv = shard["conv1d.weight"].reshape(lk, layout.rows_per_k_head, 1, -1)
        q_full, k_full, v_full = torch.split(
            hf["conv1d.weight"], [layout.key_dim, layout.key_dim, layout.value_dim], dim=0
        )
        for i in range(lk):
            h = r * lk + i
            assert torch.equal(
                conv[i, : layout.head_k_dim], q_full[h * layout.head_k_dim : (h + 1) * layout.head_k_dim]
            )
            assert torch.equal(
                conv[i, layout.head_k_dim : 2 * layout.head_k_dim],
                k_full[h * layout.head_k_dim : (h + 1) * layout.head_k_dim],
            )
            assert torch.equal(
                conv[i, 2 * layout.head_k_dim :],
                v_full[h * layout.group * layout.head_v_dim : (h + 1) * layout.group * layout.head_v_dim],
            )
        assert torch.equal(shard["norm.weight"], hf["norm.weight"])


def test_only_declared_params_are_reordered():
    for hf_layout in ("qwen3_5", "qwen3_next"):
        layout = _layout(hf_layout)
        hf = _hf_state(layout, torch.Generator().manual_seed(1))
        for name, value in hf.items():
            reordered = hf_to_megatron_linear_attn(layout, name, value)
            if name in ("conv1d.weight",) or (hf_layout == "qwen3_5" and name == "in_proj_qkv.weight"):
                assert not torch.equal(reordered, value), name
            else:
                assert reordered is value, name
            assert torch.equal(megatron_to_hf_linear_attn(layout, name, reordered), value), name


def _fake_hf_config(layout: GdnLayout):
    return SimpleNamespace(
        hidden_size=layout.hidden_size,
        linear_num_key_heads=layout.num_k_heads,
        linear_num_value_heads=layout.num_v_heads,
        linear_key_head_dim=layout.head_k_dim,
        linear_value_head_dim=layout.head_v_dim,
        linear_conv_kernel_dim=layout.conv_kernel_size,
        rms_norm_eps=layout.rms_norm_eps,
        hidden_act="silu",
    )


@pytest.mark.parametrize(
    "module_name, hf_layout, prefix",
    [
        ("miles.backends.megatron_utils.megatron_to_hf.qwen3_5", "qwen3_5", "model.language_model.layers.3"),
        ("miles.backends.megatron_utils.megatron_to_hf.qwen3_next", "qwen3_next", "model.layers.3"),
    ],
)
def test_direct_converter_restores_hf_row_order(monkeypatch, module_name, hf_layout, prefix):
    import importlib

    converter = importlib.import_module(module_name)
    layout = _layout(hf_layout)
    monkeypatch.setattr(converter, "_gdn_layout", lambda hf_checkpoint: layout)
    args = SimpleNamespace(
        hf_checkpoint="/nonexistent", kv_channels=16, hidden_size=64, num_attention_heads=4, num_query_groups=2
    )
    convert = converter.convert_qwen3_5_to_hf if hf_layout == "qwen3_5" else converter.convert_qwen3_next_to_hf
    hf = _hf_state(layout, torch.Generator().manual_seed(2))
    for name, value in hf.items():
        megatron_value = hf_to_megatron_linear_attn(layout, name, value)
        out = convert(args, f"module.module.decoder.layers.3.self_attention.linear_attn.{name}", megatron_value)
        assert out == [(f"{prefix}.linear_attn.{name}", out[0][1])]
        assert torch.equal(out[0][1], value), name


@pytest.mark.parametrize("hf_layout", ["qwen3_5", "qwen3_next"])
def test_bridge_formats_are_inverse_and_tp_split_is_contiguous(hf_layout):
    pytest.importorskip("mbridge")
    if hf_layout == "qwen3_5":
        from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge as Bridge
    else:
        from miles_plugins.mbridge.qwen3_next import Qwen3NextBridge as Bridge
    layout = _layout(hf_layout)
    bridge = Bridge.__new__(Bridge)
    bridge.hf_config = _fake_hf_config(layout)
    bridge.dtype = None
    bridge.make_vocab_size_divisible_by = None
    hf = _hf_state(layout, torch.Generator().manual_seed(3))
    for name, value in hf.items():
        mcore_name = f"decoder.layers.0.self_attention.linear_attn.{name}"
        mcore = bridge._weight_to_mcore_format(mcore_name, [value])
        assert torch.equal(mcore, hf_to_megatron_linear_attn(layout, name, value)), name
        # ``_weight_name_mapping_mcore_to_hf`` needs a real bridge; check the tensor half of the export path.
        bridge._weight_name_mapping_mcore_to_hf = lambda n: [n]
        names, tensors = bridge._weight_to_hf_format(mcore_name, mcore)
        assert torch.equal(tensors[0], value), name


def test_head_interleaved_name_filter():
    from miles_plugins.mbridge.gdn_layout import _head_interleaved_linear_attn_param as pick

    assert pick("decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight", "qwen3_5") == "in_proj_qkv.weight"
    assert pick("decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight", "qwen3_next") is None
    assert pick("decoder.layers.0.self_attention.linear_attn.conv1d.weight", "qwen3_next") == "conv1d.weight"
    assert pick("decoder.layers.0.self_attention.linear_attn.A_log", "qwen3_5") is None
    assert pick("decoder.layers.0.self_attention.linear_qkv.weight", "qwen3_5") is None


def test_layout_helpers_importable_without_cuda():
    # The layout math is plain torch; the module must not need a GPU to import.
    assert isinstance(sys.modules["miles_plugins.models.gdn_attention"], types.ModuleType)
