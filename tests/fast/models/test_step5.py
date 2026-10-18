"""Step5 model plugin: model flags, the checkpoint-config check, the norms and activation, and the bridge."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
from safetensors.torch import save_file  # noqa: E402
from tests.fast.launch_scripts.model_args_harness import expand_model_args  # noqa: E402

from miles.utils.arguments import hf_validate_args  # noqa: E402
from miles_plugins.models.step5.checkpoint_config import check_matches_checkpoint  # noqa: E402
from miles_plugins.models.step5.layer_math import zero_centered_rms_norm  # noqa: E402
from miles_plugins.models.step5.layers import is_full_attention_layer  # noqa: E402
from miles_plugins.models.step5.qkv_layout import fuse_qkv_and_gate, split_qkv_and_gate  # noqa: E402

_SLICE = (0, 3, 4, 88, 91)


@pytest.fixture
def step5_bridge():
    """The bridge module. mbridge is installed in the training image but not in CPU CI, so only these tests skip there."""
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge import step5

    return step5


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_sliced_model_flags_keep_each_layers_kind():
    argv = expand_model_args("step5-5layer")
    assert _flag(argv, "--num-layers") == "5"
    # 0, 4 and 88 are sliding-window, 3 and 91 full attention; 3, 4 and 88 are MoE.
    assert _flag(argv, "--window-attn-skip-freq") == "[1,0,1,1,0]"
    assert _flag(argv, "--moe-layer-freq") == "[0,1,1,1,0]"


def test_full_model_flags_cover_every_layer():
    argv = expand_model_args("step5")
    window = json.loads(_flag(argv, "--window-attn-skip-freq"))
    moe = json.loads(_flag(argv, "--moe-layer-freq"))
    assert len(window) == len(moe) == 92
    assert [index for index, swa in enumerate(window) if not swa] == list(range(3, 92, 4))
    assert [index for index, is_moe in enumerate(moe) if is_moe] == list(range(3, 91))


def _megatron_config(**overrides):
    config = dict(
        num_layers=5,
        hidden_size=4096,
        ffn_hidden_size=13824,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=192,
        layernorm_epsilon=1e-5,
        fp32_residual_connection=True,
        head_wise_attn_gate=True,
        window_size=(511, 0),
        window_attn_skip_freq=[1, 0, 1, 1, 0],
        num_moe_experts=352,
        moe_router_topk=8,
        moe_ffn_hidden_size=1536,
        moe_shared_expert_intermediate_size=1536,
        moe_router_topk_scaling_factor=3.0,
        moe_router_enable_expert_bias=True,
        moe_layer_freq=[0, 1, 1, 1, 0],
    )
    return SimpleNamespace(**(config | overrides))


def _checkpoint_config(**overrides):
    full = [index % 4 == 3 for index in _SLICE]
    config = dict(
        num_hidden_layers=5,
        hidden_size=4096,
        intermediate_size=13824,
        num_attention_heads=64,
        swa_num_attention_heads=None,
        num_attention_groups=4,
        head_dim=192,
        rms_norm_eps=1e-5,
        zero_centered=True,
        fp32_residual_connection=True,
        use_head_wise_attn_gate=True,
        sliding_window=512,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 1.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
            "original_max_position_embeddings": 1048576,
        },
        sparse_config={
            "enabled": True,
            "apply_to_layer_types": ["full_attention"],
            "num_provider_groups": 4,
            "sparse_indexer_num_heads": 16,
            "sparse_indexer_num_k_heads": 1,
            "proxy_dim": 256,
            "region_block_size": 8,
            "topk": 512,
            "compression_method": "csa_block_compress",
            "sparse_indexer_use_rope": True,
            "sparse_indexer_rope_dim": 32,
            "sparse_indexer_q_norm_type": "rmsnorm",
            "sparse_indexer_k_norm_type": "layernorm",
            "sparse_indexer_csa_z_norm_type": "none",
        },
        moe_num_experts=352,
        moe_top_k=8,
        moe_intermediate_size=1536,
        share_expert_dim=1536,
        moe_router_scaling_factor=3.0,
        use_moe_router_bias=True,
        norm_expert_weight=True,
        moe_layer_list=[1, 2, 3],
        layer_types=["full_attention" if f else "sliding_attention" for f in full],
        rope_theta=[1e7 if f else 1e4 for f in full],
        partial_rotary_factors=[0.3333333333333333 if f else 1.0 for f in full],
        swiglu_limits=[7.0] * 5,
        swiglu_limits_shared=[7.0] * 5,
    )
    return SimpleNamespace(**(config | overrides))


def test_full_attention_follows_the_skip_pattern():
    config = _megatron_config()
    assert [is_full_attention_layer(config, number) for number in range(1, 6)] == [False, True, False, False, True]


def test_matching_checkpoint_passes():
    check_matches_checkpoint(_megatron_config(), _checkpoint_config())


@pytest.mark.parametrize(
    "megatron_overrides,match",
    [
        ({"moe_layer_freq": [0, 1, 0, 1, 0]}, r"moe\[2\]"),
        ({"window_attn_skip_freq": [1, 1, 1, 1, 0]}, r"layer_types\[1\]"),
        ({"window_size": (512, 0)}, "sliding_window"),
        ({"num_query_groups": 8}, "num_attention_groups"),
    ],
)
def test_mismatched_flags_are_named(megatron_overrides, match):
    with pytest.raises(ValueError, match=match):
        check_matches_checkpoint(_megatron_config(**megatron_overrides), _checkpoint_config())


@pytest.mark.parametrize(
    "sparse_overrides,match",
    [
        ({"num_provider_groups": 8}, r"sparse_config\.num_provider_groups"),
        ({"topk": 1024}, r"sparse_config\.topk"),
        ({"sparse_indexer_k_norm_type": "rmsnorm"}, r"sparse_config\.sparse_indexer_k_norm_type"),
    ],
)
def test_indexer_profile_is_checked(sparse_overrides, match):
    checkpoint = _checkpoint_config()
    checkpoint.sparse_config = checkpoint.sparse_config | sparse_overrides
    with pytest.raises(ValueError, match=match):
        check_matches_checkpoint(_megatron_config(), checkpoint)


def test_indexer_rope_scaling_is_checked():
    checkpoint = _checkpoint_config()
    checkpoint.rope_scaling = checkpoint.rope_scaling | {"high_freq_factor": 4.0}
    with pytest.raises(ValueError, match=r"rope_scaling\.high_freq_factor"):
        check_matches_checkpoint(_megatron_config(), checkpoint)


def test_rope_theta_is_checked_per_layer():
    with pytest.raises(ValueError, match=r"rope_theta\[4\]"):
        check_matches_checkpoint(_megatron_config(), _checkpoint_config(rope_theta=[1e4, 1e7, 1e4, 1e4, 1e4]))


def test_training_requires_rollout_routing_replay():
    def validate(replay: bool) -> None:
        args = SimpleNamespace(model_name="step5", context_parallel_size=1, use_rollout_routing_replay=replay)
        hf_validate_args(args, SimpleNamespace(model_type="step5"))

    validate(replay=True)
    with pytest.raises(AssertionError, match="--use-rollout-routing-replay"):
        validate(replay=False)


def test_zero_centered_norm_scales_by_one_plus_weight():
    x = torch.randn(3, 8, dtype=torch.float64)
    weight = torch.linspace(-0.5, 0.5, 8)
    out = zero_centered_rms_norm(x, weight, eps=1e-5)
    expected = x.float() / (x.float().pow(2).mean(-1, keepdim=True) + 1e-5).sqrt() * (1 + weight)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_each_tp_rank_gets_its_groups_then_its_gate_rows(tp_size):
    groups, heads, head_dim, hidden = 4, 8, 3, 5
    query = torch.arange(heads * head_dim * hidden, dtype=torch.float32).view(heads * head_dim, hidden)
    key = -torch.arange(groups * head_dim * hidden, dtype=torch.float32).view(groups * head_dim, hidden) - 1
    value = key - 1000
    gate = torch.arange(heads * hidden, dtype=torch.float32).view(heads, hidden) + 5000

    fused = fuse_qkv_and_gate(query=query, key=key, value=value, gate=gate, num_query_groups=groups, tp_size=tp_size)

    per_group = heads // groups
    for rank, block in enumerate(fused.chunk(tp_size)):
        rank_groups = range(rank * groups // tp_size, (rank + 1) * groups // tp_size)
        expected_qkv = [
            torch.cat(
                [
                    query.view(groups, per_group * head_dim, hidden)[group],
                    key.view(groups, head_dim, hidden)[group],
                    value.view(groups, head_dim, hidden)[group],
                ]
            )
            for group in rank_groups
        ]
        expected = torch.cat(expected_qkv + [gate.chunk(tp_size)[rank]])
        torch.testing.assert_close(block, expected, rtol=0, atol=0)


def test_norms_and_router_bias_keep_fp32_through_the_bridge(step5_bridge):
    bridge = object.__new__(step5_bridge.Step5Bridge)
    bridge.dtype = torch.bfloat16
    weight = torch.tensor([0.1234567], dtype=torch.float32)
    for name in (
        "decoder.layers.0.input_layernorm.weight",
        "decoder.layers.0.pre_mlp_layernorm.weight",
        "decoder.layers.0.self_attention.q_layernorm.weight",
        "decoder.final_layernorm.weight",
        "decoder.layers.1.mlp.router.expert_bias",
    ):
        converted = bridge._weight_to_mcore_format(name, [weight])
        assert converted.dtype == torch.float32
        torch.testing.assert_close(converted, weight, rtol=0, atol=0)


def test_indexer_fp32_tensors_stay_fp32_through_the_bridge(step5_bridge):
    bridge = object.__new__(step5_bridge.Step5Bridge)
    bridge.dtype = torch.bfloat16
    bridge.make_vocab_size_divisible_by = None
    prefix = "decoder.layers.3.self_attention.indexer"
    weight = torch.tensor([0.1234567], dtype=torch.float32)
    for name in ("w_weight", "q_norm_weight", "k_norm_weight", "k_norm_bias", "ssmax_s"):
        converted = bridge._weight_to_mcore_format(f"{prefix}.{name}", [weight])
        assert converted.dtype == torch.float32
        torch.testing.assert_close(converted, weight, rtol=0, atol=0)
    assert bridge._weight_to_mcore_format(f"{prefix}.q_weight", [weight]).dtype == torch.bfloat16


@pytest.mark.parametrize("tp_size", [2, 4])
def test_indexer_q_and_w_rows_split_by_provider_group(step5_bridge, tp_size):
    bridge = object.__new__(step5_bridge.Step5Bridge)
    q = torch.arange(16 * 3, dtype=torch.float32).view(16 * 3, 1)  # 16 heads of 3 rows
    w = torch.arange(16, dtype=torch.float32).view(16, 1)
    name = "decoder.layers.3.self_attention.indexer"
    q_shards = bridge._weight_split_across_tp(f"{name}.q_weight", q, torch.empty(16 * 3 // tp_size, 1), tp_size)
    w_shards = bridge._weight_split_across_tp(f"{name}.w_weight", w, torch.empty(16 // tp_size, 1), tp_size)
    for rank in range(tp_size):
        heads = range(rank * 16 // tp_size, (rank + 1) * 16 // tp_size)  # this rank's provider groups
        torch.testing.assert_close(q_shards[rank], q.view(16, 3, 1)[list(heads)].reshape(-1, 1), rtol=0, atol=0)
        torch.testing.assert_close(w_shards[rank], w[list(heads)], rtol=0, atol=0)
    k = torch.ones(3, 1)
    assert all(shard is k for shard in bridge._weight_split_across_tp(f"{name}.k_weight", k, k, tp_size))


def test_grouped_expert_names_map_to_virtual_expert_names(step5_bridge):
    bridge = object.__new__(step5_bridge.Step5Bridge)
    assert bridge._weight_name_mapping_mlp("decoder.layers.1.mlp.experts.linear_fc1.weight17") == [
        "model.layers.1.moe.experts.17.gate_proj.weight",
        "model.layers.1.moe.experts.17.up_proj.weight",
    ]
    assert bridge._weight_name_mapping_mlp("decoder.layers.1.mlp.shared_experts.linear_fc2.weight") == [
        "model.layers.1.share_expert.down_proj.weight"
    ]
    assert bridge._weight_name_mapping_mlp("decoder.layers.1.mlp.router.expert_bias") == [
        "model.layers.1.moe.router_bias"
    ]


def test_virtual_expert_names_read_one_packed_row(step5_bridge, tmp_path):
    experts, rows, cols = 3, 130, 260  # past one 128 block in both dims
    bf16 = torch.randn(experts, rows, cols).bfloat16()
    fp8 = torch.randn(experts, rows, cols).to(torch.float8_e4m3fn)
    scale = torch.rand(experts, 2, 3) + 0.5
    save_file(
        {
            "model.layers.1.moe.up_proj.weight": bf16,
            "model.layers.2.moe.up_proj.weight": fp8,
            "model.layers.2.moe.up_proj.weight_scale_inv": scale,
        },
        tmp_path / "model.safetensors",
    )

    io = step5_bridge.Step5SafeTensorIO(str(tmp_path))
    loaded = io.load_some_hf_weight(
        ["model.layers.1.moe.experts.2.up_proj.weight", "model.layers.2.moe.experts.1.up_proj.weight"]
    )

    torch.testing.assert_close(loaded["model.layers.1.moe.experts.2.up_proj.weight"], bf16[2], rtol=0, atol=0)
    torch.testing.assert_close(
        loaded["model.layers.2.moe.experts.1.up_proj.weight"],
        step5_bridge.dequantize_block_fp8(fp8[1], scale[1]),
        rtol=0,
        atol=0,
    )


def test_block_fp8_dequant_scales_each_block(step5_bridge):
    weight = torch.ones(130, 129).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = step5_bridge.dequantize_block_fp8(weight, scale)
    assert out[0, 0] == 1.0 and out[0, 128] == 2.0 and out[128, 0] == 3.0 and out[129, 128] == 4.0


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_split_undoes_fuse(tp_size):
    groups, heads, head_dim, hidden = 4, 8, 3, 5
    query, key, value = (torch.randn(rows * head_dim, hidden) for rows in (heads, groups, groups))
    gate = torch.randn(heads, hidden)
    fused = fuse_qkv_and_gate(query=query, key=key, value=value, gate=gate, num_query_groups=groups, tp_size=tp_size)

    split = split_qkv_and_gate(
        fused, num_query_groups=groups, num_attention_heads=heads, head_dim=head_dim, tp_size=tp_size
    )

    for got, want in zip(split, (query, key, value, gate), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def _sync(name: str, param: torch.Tensor) -> dict[str, torch.Tensor]:
    from miles.backends.megatron_utils.megatron_to_hf.step5 import convert_step5_to_hf

    args = SimpleNamespace(num_query_groups=4, num_attention_heads=8, kv_channels=3, tensor_model_parallel_size=2)
    return dict(convert_step5_to_hf(args, name, param))


@pytest.mark.parametrize(
    "megatron_name,shape,hf_names",
    [
        ("module.module.embedding.word_embeddings.weight", (6, 5), ["model.embed_tokens.weight"]),
        ("module.module.decoder.final_layernorm.weight", (5,), ["model.norm.weight"]),
        (
            "module.module.decoder.layers.2.pre_mlp_layernorm.weight",
            (5,),
            ["model.layers.2.post_attention_layernorm.weight"],
        ),
        (
            "module.module.decoder.layers.2.self_attention.linear_qkv.weight",
            (8 * 3 + 2 * 4 * 3 + 8, 5),
            [f"model.layers.2.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "g")],
        ),
        (
            "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight351",
            (6, 5),
            ["model.layers.2.moe.experts.351.gate_proj.weight", "model.layers.2.moe.experts.351.up_proj.weight"],
        ),
        (
            "module.module.decoder.layers.2.mlp.experts.linear_fc2.weight0",
            (5, 3),
            ["model.layers.2.moe.experts.0.down_proj.weight"],
        ),
        (
            "module.module.decoder.layers.2.mlp.shared_experts.linear_fc1.weight",
            (6, 5),
            ["model.layers.2.share_expert.gate_proj.weight", "model.layers.2.share_expert.up_proj.weight"],
        ),
        ("module.module.decoder.layers.2.mlp.router.expert_bias", (352,), ["model.layers.2.moe.router_bias"]),
        ("module.module.decoder.layers.2.mlp.router.weight", (352, 5), ["model.layers.2.moe.gate.weight"]),
    ],
)
def test_weight_sync_uses_checkpoint_and_per_expert_names(megatron_name, shape, hf_names):
    assert list(_sync(megatron_name, torch.zeros(shape))) == hf_names


@pytest.mark.parametrize(
    "module_name,hf_prefix",
    [("mlp.experts.linear_fc1.weight7", "moe.experts.7"), ("mlp.shared_experts.linear_fc1.weight", "share_expert")],
)
def test_weight_sync_sends_the_gate_rows_then_the_up_rows(module_name, hf_prefix):
    gate, up = torch.randn(3, 5), torch.randn(3, 5)
    sent = _sync(f"module.module.decoder.layers.2.{module_name}", torch.cat([gate, up]))
    torch.testing.assert_close(sent[f"model.layers.2.{hf_prefix}.gate_proj.weight"], gate, rtol=0, atol=0)
    torch.testing.assert_close(sent[f"model.layers.2.{hf_prefix}.up_proj.weight"], up, rtol=0, atol=0)


def test_weight_sync_splits_the_qkv_and_gate_rows_the_bridge_fused():
    query, key, value, gate = torch.randn(8 * 3, 5), torch.randn(4 * 3, 5), torch.randn(4 * 3, 5), torch.randn(8, 5)
    fused = fuse_qkv_and_gate(query=query, key=key, value=value, gate=gate, num_query_groups=4, tp_size=2)
    sent = _sync("module.module.decoder.layers.2.self_attention.linear_qkv.weight", fused)
    for projection, expected in (("q", query), ("k", key), ("v", value), ("g", gate)):
        torch.testing.assert_close(
            sent[f"model.layers.2.self_attn.{projection}_proj.weight"], expected, rtol=0, atol=0
        )


def test_fp8_sync_leaves_modules_the_checkpoint_keeps_in_bf16():
    quantizer = pytest.importorskip("miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8")
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [f"model.layers.2.self_attn.{p}_proj" for p in ("q", "k", "v", "g")],
    }
    converted = [(f"model.layers.2.self_attn.{p}_proj.weight", torch.zeros(4, 4)) for p in ("q", "k", "v", "g")]

    out = quantizer.quantize_params_fp8(
        SimpleNamespace(), "module.module.decoder.layers.2.self_attention.linear_qkv.weight", converted, config
    )

    assert out is converted
