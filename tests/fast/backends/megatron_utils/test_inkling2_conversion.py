import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mbridge")
from miles.backends.megatron_utils.megatron_to_hf.inkling2 import (  # noqa: E402
    convert_inkling2_to_hf,
    load_inkling2_text_config,
)
from miles_plugins.mbridge.inkling2 import Inkling2Bridge, dequantize_mxfp8, dequantize_nvfp4  # noqa: E402

_TEXT_CONFIG = {
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "d_rel": 2,
    "local_layer_ids": [0, 2],
}


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": _TEXT_CONFIG}))
    quant = {
        "quant_method": "inkling2",
        "model": {
            "parameter_groups": [
                {"compute_dtype": "nvfp4", "tensors": ["model.llm.layers.0.mlp.experts.w13_weight"]},
                {"compute_dtype": "bfloat16", "tensors": ["model.llm.layers.0.mlp.experts.w2_weight"]},
            ]
        },
    }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))
    load_inkling2_text_config.cache_clear()
    return str(tmp_path)


def _convert(checkpoint, name, param):
    return convert_inkling2_to_hf(SimpleNamespace(hf_checkpoint=checkpoint), f"module.module.{name}", param)


def _bridge(checkpoint) -> Inkling2Bridge:
    bridge = Inkling2Bridge.__new__(Inkling2Bridge)
    text = SimpleNamespace(to_dict=lambda: _TEXT_CONFIG)
    bridge.hf_config = SimpleNamespace(name_or_path=checkpoint, text_config=text)
    bridge.dtype = torch.bfloat16
    return bridge


def test_linear_q_splits_into_query_rel_gate_sink(checkpoint):
    """linear_q packs four projections of the Q-side source; a wrong order or size
    silently feeds the gate into the relative-position features."""
    rows = 4 * 8 + 4 * 2 + 4 * 8 + 4
    param = torch.arange(rows).unsqueeze(-1).float()
    parts = _convert(checkpoint, "decoder.layers.1.self_attention.linear_q.weight", param)
    assert [name for name, _ in parts] == [
        "model.llm.layers.1.attn.wq_du.weight",
        "model.llm.layers.1.attn.wr_du.weight",
        "model.llm.layers.1.attn.gate_attn.weight",
        "model.llm.layers.1.attn.attn_sink_proj.weight",
    ]
    assert [p.shape[0] for _, p in parts] == [32, 8, 32, 4]
    assert parts[3][1].squeeze(-1).tolist() == [72, 73, 74, 75]


def test_attention_residual_queries_follow_the_engine_indexing(checkpoint):
    """attn_queries has no entry for layer 0, so layer i's query is index i - 1."""
    assert _convert(checkpoint, "decoder.layers.3.attn_query", torch.zeros(1))[0][0] == (
        "model.llm.attn_res.attn_queries.2"
    )
    assert _convert(checkpoint, "decoder.layers.3.mlp_query", torch.zeros(1))[0][0] == (
        "model.llm.attn_res.mlp_queries.3"
    )


def test_expert_fc1_splits_gate_before_up(checkpoint):
    param = torch.arange(8).reshape(4, 2)
    expert = _convert(checkpoint, "decoder.layers.0.mlp.experts.linear_fc1.weight5", param)
    assert [name for name, _ in expert] == [
        "model.llm.layers.0.mlp.experts.5.gate_proj.weight",
        "model.llm.layers.0.mlp.experts.5.up_proj.weight",
    ]
    assert expert[0][1].tolist() == [[0, 1], [2, 3]]


@pytest.mark.parametrize(
    "name",
    [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.attn_norm.weight",
        "decoder.layers.1.q_input_norm.weight",
        "decoder.layers.1.self_attention.k_sconv_weight_proj",
        "decoder.layers.1.self_attention.linear_q.weight",
        "decoder.layers.2.attn_query",
        "decoder.layers.2.mlp.router.expert_bias",
    ],
)
def test_bridge_loads_what_the_converter_exports(checkpoint, name):
    """The trainer loads HF weights through the bridge and exports them through the
    converter; the two name maps must be inverses."""
    rows = 4 * 8 + 4 * 2 + 4 * 8 + 4 if name.endswith("linear_q.weight") else 2
    exported = [hf for hf, _ in _convert(checkpoint, name, torch.zeros(rows, 1))]
    assert _bridge(checkpoint)._weight_name_mapping_mcore_to_hf(name) == exported


def test_bridge_reads_quantized_experts_with_their_scales(checkpoint):
    bridge = _bridge(checkpoint)
    assert bridge._weight_name_mapping_mcore_to_hf("decoder.layers.0.mlp.experts.linear_fc1.weight3") == [
        "model.llm.layers.0.mlp.experts.w13_weight",
        "model.llm.layers.0.mlp.experts.w13_weight_scale",
        "model.llm.layers.0.mlp.experts.w13_weight_amax",
        "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator",
    ]
    assert bridge._weight_name_mapping_mcore_to_hf("decoder.layers.0.mlp.experts.linear_fc2.weight3") == [
        "model.llm.layers.0.mlp.experts.w2_weight"
    ]


def test_bridge_keeps_fp32_parameters_in_fp32(checkpoint):
    query = torch.tensor([1.0 + 2**-20])
    loaded = _bridge(checkpoint)._weight_to_mcore_format("decoder.layers.2.attn_query", [query])
    assert loaded.dtype == torch.float32 and loaded.item() == query.item()


def test_nvfp4_dequantization():
    # codes 0x21 -> low nibble 1 (0.5), high nibble 2 (1.0); 16 values share one block scale
    packed = torch.full((1, 8), 0x21, dtype=torch.uint8)
    scales = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    weight = dequantize_nvfp4(packed, scales, torch.tensor(3.0), torch.tensor([6.0]))
    assert weight.tolist() == [[0.5, 1.0] * 8]


def test_mxfp8_dequantization():
    weight = torch.full((1, 32), 1.5, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[129]], dtype=torch.uint8)
    assert dequantize_mxfp8(weight, scales).tolist() == [[6.0] * 32]


def test_bridge_loads_this_ep_ranks_experts(checkpoint):
    """load_weights hands the bridge rank-local expert names; on EP rank 1 of 4 with 8
    experts, local expert 1 is global expert 3."""
    bridge = _bridge(checkpoint)
    bridge.mpu = SimpleNamespace(ep_rank=1, ep_size=4)
    bridge.hf_config.text_config.n_routed_experts = 8
    packed = torch.arange(8, dtype=torch.bfloat16).view(8, 1, 1).expand(8, 4, 2).contiguous()
    loaded = bridge._weight_to_mcore_format("decoder.layers.0.mlp.experts.linear_fc2.weight1", [packed])
    assert loaded.unique().tolist() == [3.0]
