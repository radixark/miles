import json
import re
from argparse import Namespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")
pytest.importorskip("fla")

_HF_TEXT_CONFIG = dict(
    model_type="glm5_next_text",
    hidden_size=4096,
    num_hidden_layers=4,
    num_attention_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_head_dim=256,
    qk_rope_head_dim=0,
    v_head_dim=256,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    index_kpool=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    n_routed_experts=288,
    num_experts_per_tok=8,
    routed_scaling_factor=2.5,
    norm_topk_prob=True,
    moe_intermediate_size=2048,
    intermediate_size=12288,
    swiglu_limit=10.0,
    vocab_size=154880,
    rms_norm_eps=1e-5,
    max_position_embeddings=1048576,
    n_shared_experts=1,
    hc_eps=1e-6,
    mhc=True,
    scoring_func="sigmoid",
    n_group=1,
    topk_group=1,
    index_kpool_compress=True,
    index_kpool_always_select_tail=True,
    linear_attn_config={"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
    layer_types=["linear_attention", "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"],
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
)

_HC = ["hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"]
_NORMS = ["input_layernorm.weight", "post_attention_layernorm.weight"]
_KDA = [
    f"self_attn.{name}"
    for name in [
        "A_log",
        "dt_bias",
        "o_norm.weight",
        *(
            f"{proj}.weight"
            for proj in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "b_proj",
                "f_a_proj",
                "f_b_proj",
                "g_a_proj",
                "g_b_proj",
            ]
        ),
        *(f"{c}_conv1d.weight" for c in "qkv"),
    ]
]
_DSA = [
    f"self_attn.{name}"
    for name in [
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.weights_proj.weight",
        "indexer.index_kpool_compress_gate",
        "indexer.index_kpool_compress_ape",
    ]
]
_DENSE = [f"mlp.{p}.weight" for p in ["gate_proj", "up_proj", "down_proj"]]
_MOE = [
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    *(f"mlp.shared_experts.{p}.weight" for p in ["gate_proj", "up_proj", "down_proj"]),
    *(f"mlp.experts.E.{p}.weight" for p in ["gate_proj", "up_proj", "down_proj"]),
]


def _checkpoint_keys() -> set[str]:
    keys = {"lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    for i, (attn, mlp) in enumerate(
        zip(_HF_TEXT_CONFIG["layer_types"], _HF_TEXT_CONFIG["mlp_layer_types"], strict=True)
    ):
        suffixes = _HC + _NORMS + (_KDA if attn == "linear_attention" else _DSA) + (_DENSE if mlp == "dense" else _MOE)
        keys |= {f"model.language_model.layers.{i}.{s}" for s in suffixes}
    return keys


def _write_checkpoint_config(tmp_path, **text_overrides) -> str:
    config = {"model_type": "glm5_next", "text_config": {**_HF_TEXT_CONFIG, **text_overrides}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _spec(tmp_path, **text_overrides):
    from miles.backends.torchtitan_utils.config import resolve_model_spec

    checkpoint = _write_checkpoint_config(tmp_path, **text_overrides)
    return resolve_model_spec(
        Namespace(titan_model_name="glm5_next", titan_model_flavor="4layer", hf_checkpoint=checkpoint)
    )


def test_the_model_is_sized_from_the_checkpoint_config(tmp_path):
    hf = _HF_TEXT_CONFIG
    model = _spec(tmp_path).model
    assert (model.dim, model.vocab_size, len(model.layers), model.num_streams) == (
        hf["hidden_size"],
        hf["vocab_size"],
        hf["num_hidden_layers"],
        hf["hc_mult"],
    )
    dsa = model.layers[1].attention
    assert (dsa.n_heads, dsa.kv_lora_rank, dsa.qk_head_dim, dsa.v_head_dim) == (
        hf["num_attention_heads"],
        hf["kv_lora_rank"],
        hf["qk_head_dim"],
        hf["v_head_dim"],
    )
    assert dsa.wq_a.out_features == hf["q_lora_rank"]
    assert (dsa.indexer.num_heads, dsa.indexer.head_dim, dsa.indexer.topk, dsa.indexer.kpool) == (
        hf["index_n_heads"],
        hf["index_head_dim"],
        hf["index_topk"],
        hf["index_kpool"],
    )
    kda = model.layers[0].kda
    linear = hf["linear_attn_config"]
    assert (kda.num_heads, kda.head_dim, kda.q_conv1d.kernel_size, kda.gate.lower_bound) == (
        linear["num_heads"],
        linear["head_dim"],
        linear["short_conv_kernel_size"],
        linear["gate_lower_bound"],
    )
    assert model.layers[0].hc_attn.sinkhorn_iterations == hf["hc_sinkhorn_iters"]
    assert model.layers[0].feed_forward.w1.out_features == hf["intermediate_size"]
    moe = model.layers[1].moe
    assert (moe.num_experts, moe.routed_experts.inner_experts.hidden_dim) == (
        hf["n_routed_experts"],
        hf["moe_intermediate_size"],
    )
    assert (moe.router.top_k, moe.router.score_func, moe.router.route_scale, moe.router.route_norm) == (
        hf["num_experts_per_tok"],
        "sigmoid",
        hf["routed_scaling_factor"],
        hf["norm_topk_prob"],
    )
    assert moe.routed_experts.inner_experts.swiglu_limit == hf["swiglu_limit"]
    assert [(layer.kda is not None, layer.moe is not None) for layer in model.layers] == [
        (attn == "linear_attention", mlp == "sparse")
        for attn, mlp in zip(hf["layer_types"], hf["mlp_layer_types"], strict=True)
    ]


def test_every_trained_tensor_maps_onto_exactly_the_checkpoint_text_keys(tmp_path):
    spec = _spec(tmp_path)
    with torch.device("meta"):
        model = spec.model.build()
    adapter = spec.state_dict_adapter(spec.model, str(tmp_path))
    state = {k: v for k, v in model.state_dict().items() if "tokens_per_expert" not in k}
    hf_keys = {re.sub(r"\.experts\.\d+\.", ".experts.E.", k) for k in adapter.to_hf(state)}
    assert hf_keys == _checkpoint_keys()


def test_a_checkpoint_variant_the_package_does_not_implement_is_refused(tmp_path):
    with pytest.raises(ValueError, match="NoPE"):
        _spec(tmp_path, qk_rope_head_dim=64)
