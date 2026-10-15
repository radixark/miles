import json
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")

# text_config of pb09204048/Step-5-Preview-5layer (original layers 0, 3, 4, 88, 91), as SGLang's Step5Config reads it
_TEXT_CONFIG = dict(
    hidden_size=4096,
    intermediate_size=13824,
    num_hidden_layers=5,
    vocab_size=128896,
    rms_norm_eps=1e-5,
    max_position_embeddings=1048576,
    num_attention_heads=64,
    num_attention_groups=4,
    swa_num_attention_heads=64,
    head_dim=192,
    sliding_window=512,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    rope_theta=[1e4, 1e7, 1e4, 1e4, 1e7],
    partial_rotary_factors=[1.0, 1 / 3, 1.0, 1.0, 1 / 3],
    rope_scaling={
        "rope_type": "llama3",
        "factor": 1.0,
        "original_max_position_embeddings": 1048576,
        "low_freq_factor": 1.0,
        "high_freq_factor": 32.0,
    },
    zero_centered=True,
    fp32_residual_connection=True,
    use_head_wise_attn_gate=True,
    moe_layer_list=[1, 2, 3],
    moe_num_experts=352,
    moe_top_k=8,
    moe_intermediate_size=1536,
    share_expert_dim=1536,
    moe_router_scaling_factor=3.0,
    use_moe_router_bias=True,
    norm_expert_weight=True,
    swiglu_limits=[7.0] * 5,
    swiglu_limits_shared=[7.0] * 5,
    sparse_config={
        "enabled": True,
        "proxy_dim": 256,
        "sparse_indexer_rope_dim": 32,
        "sparse_indexer_num_heads": 16,
        "num_provider_groups": 4,
        "topk": 512,
        "region_block_size": 8,
        "apply_to_layer_types": ["full_attention"],
    },
)

_ATTENTION = [
    *(f"self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "q_norm", "k_norm")),
]
_INDEXER = [
    *(f"self_attn.sparse_indexer_{p}.weight" for p in ("q", "k", "z", "w", "q_norm", "k_norm")),
    "self_attn.sparse_indexer_k_norm.bias",
    "self_attn.ssmax_s",
]
_NORMS = ["input_layernorm.weight", "post_attention_layernorm.weight"]
_DENSE = [f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
_MOE = [
    "moe.gate.weight",
    "moe.router_bias",
    *(f"moe.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")),
    *(f"share_expert.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")),
]


def _checkpoint_text_keys() -> set[str]:
    """The slice's model.safetensors.index.json, less the vision tower and the FP8 scales."""
    keys = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
    for index, layer_type in enumerate(_TEXT_CONFIG["layer_types"]):
        suffixes = _NORMS + _ATTENTION + (_INDEXER if layer_type == "full_attention" else [])
        suffixes += _MOE if index in _TEXT_CONFIG["moe_layer_list"] else _DENSE
        keys |= {f"model.layers.{index}.{suffix}" for suffix in suffixes}
    return keys


def _model_config(**overrides):
    from miles.backends.torchtitan_utils.models.step5 import build_model_config

    return build_model_config(SimpleNamespace(text_config=SimpleNamespace(**{**_TEXT_CONFIG, **overrides})))


def test_the_model_is_sized_from_the_checkpoint_config():
    from miles.backends.torchtitan_utils.models.step5.layers import Step5SparseAttention

    hf = _TEXT_CONFIG
    model = _model_config()
    assert (model.dim, model.vocab_size, len(model.layers), model.sliding_window) == (
        hf["hidden_size"],
        hf["vocab_size"],
        hf["num_hidden_layers"],
        hf["sliding_window"],
    )
    assert [isinstance(layer.attention, Step5SparseAttention.Config) for layer in model.layers] == [
        kind == "full_attention" for kind in hf["layer_types"]
    ]
    assert [layer.attention.rope.theta for layer in model.layers] == hf["rope_theta"]
    assert [layer.moe is not None for layer in model.layers] == [i in hf["moe_layer_list"] for i in range(5)]
    attention = model.layers[1].attention
    assert (attention.n_heads, attention.n_kv_heads, attention.head_dim) == (64, 4, 192)
    assert model.layers[0].feed_forward.w1.out_features == hf["intermediate_size"]
    moe = model.layers[1].moe
    assert (moe.num_experts, moe.routed_experts.inner_experts.hidden_dim) == (352, hf["moe_intermediate_size"])
    assert moe.shared_experts.w1.out_features == hf["share_expert_dim"]
    assert (moe.router.top_k, moe.router.score_func, moe.router.route_norm, moe.router.route_scale) == (
        8,
        "sigmoid",
        True,
        3.0,
    )
    assert moe.routed_experts.inner_experts.swiglu_limit == moe.shared_experts.swiglu_limit == 7.0


def test_every_trained_tensor_maps_onto_exactly_the_checkpoint_text_keys():
    from miles.backends.torchtitan_utils.models.step5.state_dict_adapter import from_hf_name, to_hf_name

    config = _model_config()
    with torch.device("meta"):
        model = config.build()
    names = list(model.state_dict())
    hf_names = [to_hf_name(name) for name in names]
    assert set(hf_names) == _checkpoint_text_keys()
    assert [from_hf_name(name) for name in hf_names] == names


def test_a_checkpoint_variant_the_package_does_not_implement_is_refused():
    with pytest.raises(ValueError, match="SwiGLU clamped"):
        _model_config(swiglu_limits=[10.0] * 5)
    with pytest.raises(ValueError, match="RoPE theta"):
        _model_config(rope_theta=[1e4] * 5)


def test_packed_fp8_experts_dequantize_across_unaligned_block_edges(tmp_path):
    """A shard of a packed expert tensor starts and ends mid-block, and each element takes its block's scale."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.metadata import MetadataIndex
    from torch.distributed.checkpoint.planner import LoadItemType, ReadItem

    from miles.backends.torchtitan_utils.models.step5.state_dict_adapter import PackedFp8StorageReader

    generator = torch.Generator().manual_seed(0)
    weight = (torch.randn(2, 200, 300, generator=generator) * 100).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 2, 3, generator=generator) + 0.5
    save_file({"experts.weight": weight, "experts.weight_scale_inv": scale}, str(tmp_path / "model-00001.safetensors"))
    index = {"weight_map": dict.fromkeys(["experts.weight", "experts.weight_scale_inv"], "model-00001.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    reader = PackedFp8StorageReader(str(tmp_path))
    reader.read_metadata()

    offsets, lengths = (1, 100, 130), (1, 90, 140)
    request = ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=MetadataIndex("experts.weight"),
        dest_offsets=torch.Size([0, 0, 0]),
        storage_index=MetadataIndex("experts.weight"),
        storage_offsets=torch.Size(offsets),
        lengths=torch.Size(lengths),
    )
    with safe_open(str(tmp_path / "model-00001.safetensors"), framework="pt") as handle:
        read = reader._read_quantized_tensor_with_block_alignment(request, handle)

    full = weight.float() * scale.repeat_interleave(128, 1)[:, :200].repeat_interleave(128, 2)[:, :, :300]
    expected = full[tuple(slice(o, o + n) for o, n in zip(offsets, lengths, strict=True))]
    assert torch.equal(read, expected)


@pytest.mark.parametrize("window", [1, 4, 6])
def test_context_parallel_window_spans_give_each_query_exactly_its_window(window):
    """Varlen attention aligns each query span to the end of its key span; under that alignment every
    query of a head-tail shard must see the ``window`` keys ending at itself, cut at its document's start."""
    from miles.backends.torchtitan_utils.models.step5.packed_sequence import _window_spans

    bounds = [0, 3, 11, 16]
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    # cp=2 head-tail over 16 tokens: rank 0 holds chunks 0 and 3, rank 1 chunks 1 and 2
    for local in ([0, 1, 2, 3, 12, 13, 14, 15], [4, 5, 6, 7, 8, 9, 10, 11]):
        query_spans, key_spans = _window_spans(cu_seqlens, torch.tensor(local), window=window)
        assert [row for start, end in query_spans for row in range(start, end)] == list(range(len(local)))
        for (q_start, q_end), (k_start, k_end) in zip(query_spans, key_spans, strict=True):
            q_len, k_len = q_end - q_start, k_end - k_start
            for j in range(q_len):
                position = local[q_start + j]
                aligned = k_start + k_len - q_len + j
                visible = set(range(max(k_start, aligned - window + 1), aligned + 1))
                document_start = max(b for b in bounds if b <= position)
                assert aligned == position
                assert visible == set(range(max(document_start, position - window + 1), position + 1))
