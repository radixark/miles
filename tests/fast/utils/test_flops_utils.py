import json
from types import SimpleNamespace

import pytest
from huggingface_hub import hf_hub_download

from miles.utils.flops_utils import KeyTerm, LayerFlops, ModelFlops, _total_keys, model_flops_from_hf_config
from miles.utils.hf_utils.config import load_hf_config


def config(**fields):
    return SimpleNamespace(**({"model_type": "llama"} | fields))


def per_layer(model_flops: ModelFlops, layer: int) -> LayerFlops:
    return model_flops.layers[layer]


def released_model_flops(repo_id: str) -> ModelFlops:
    with open(hf_hub_download(repo_id, "config.json")) as f:
        return model_flops_from_hf_config(json.load(f))


@pytest.mark.parametrize("pool", [1, 3, 128])
@pytest.mark.parametrize("cap", [None, 1, 5, 2048])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 128, 129, 1000])
def test_total_keys_matches_counting_every_position(seq_len, pool, cap):
    counted = sum(min(q // pool, cap if cap is not None else q) for q in range(1, seq_len + 1))
    assert _total_keys(seq_len, pool, cap) == counted


def test_fwd_flops_sums_fixed_cost_and_per_key_cost_per_sequence():
    model_flops = ModelFlops(layers=(LayerFlops(10.0, (KeyTerm(2.0), KeyTerm(3.0, cap=4))),), lm_head=1.0)
    seq_len = 6
    expected = seq_len * 11.0 + 2.0 * (seq_len * (seq_len + 1) // 2) + 3.0 * (1 + 2 + 3 + 4 + 4 + 4)
    assert model_flops.fwd_flops([seq_len, seq_len]) == 2 * expected


LLAMA_8B = dict(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=14336,
    num_hidden_layers=32,
    vocab_size=128256,
)


def test_dense_gqa_layer():
    flops = model_flops_from_hf_config(config(**LLAMA_8B))
    hidden, heads, kv_heads, head_dim = 4096, 32, 8, 128
    projections = 2 * hidden * heads * head_dim + 4 * hidden * kv_heads * head_dim + 2 * heads * head_dim * hidden
    assert per_layer(flops, 0) == LayerFlops(projections + 6 * hidden * 14336, (KeyTerm(4 * heads * head_dim),))
    assert flops.lm_head == 2 * hidden * 128256


def test_output_projection_reads_heads_times_head_dim_not_hidden():
    # Qwen3-4B: 32 heads x 128 = 4096 wide attention on a 2560 residual
    qwen3_4b = config(
        model_type="qwen3",
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=9728,
        num_hidden_layers=36,
        vocab_size=151936,
    )
    layer = per_layer(model_flops_from_hf_config(qwen3_4b), 0)
    attention = 2 * 2560 * 4096 + 4 * 2560 * 1024 + 2 * 4096 * 2560
    assert layer.per_token == attention + 6 * 2560 * 9728


def test_sliding_window_layers_cap_their_keys():
    gpt_oss = config(
        model_type="gpt_oss",
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2880,
        num_hidden_layers=4,
        num_local_experts=32,
        experts_per_token=4,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"] * 2,
        vocab_size=201088,
    )
    flops = model_flops_from_hf_config(gpt_oss)
    assert per_layer(flops, 0).key_terms == (KeyTerm(4 * 64 * 64, cap=128),)
    assert per_layer(flops, 1).key_terms == (KeyTerm(4 * 64 * 64),)
    moe = 2 * 2880 * 32 + 4 * 6 * 2880 * 2880
    attention = 2 * 2880 * 4096 + 4 * 2880 * 512 + 2 * 4096 * 2880
    assert per_layer(flops, 0).per_token == attention + moe


def test_moe_layers_follow_the_dense_prefix_and_count_router_and_shared_experts():
    deepseek_v3 = config(
        model_type="deepseek_v3",
        hidden_size=7168,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        num_hidden_layers=61,
        vocab_size=129280,
    )
    flops = model_flops_from_hf_config(deepseek_v3)
    hidden, heads = 7168, 128
    mla = (
        2 * 1536 * (hidden + heads * 192)
        + 2 * hidden * (512 + 64)
        + 2 * 512 * heads * (128 + 128)
        + 2 * heads * 128 * hidden
    )
    assert per_layer(flops, 2).per_token == mla + 6 * hidden * 18432
    assert per_layer(flops, 3).per_token == mla + 2 * hidden * 256 + 6 * 8 * hidden * 2048 + 6 * hidden * 2048
    assert per_layer(flops, 3).key_terms == (KeyTerm(2 * heads * 192 + 2 * heads * 128),)


QWEN38_27B = dict(
    model_type="qwen3_5_text",
    hidden_size=5120,
    num_hidden_layers=64,
    full_attention_interval=4,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248320,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    attn_output_gate=True,
)


def test_hybrid_model_charges_quadratic_attention_only_on_full_attention_layers():
    flops = model_flops_from_hf_config({"model_type": "qwen3_5", "text_config": QWEN38_27B})
    full = [i for i, layer in enumerate(flops.layers) if layer.key_terms]
    assert full == list(range(3, 64, 4))
    hidden = 5120
    gated_attention = 2 * hidden * 6144 * 2 + 4 * hidden * 1024 + 2 * 6144 * hidden
    assert per_layer(flops, 3).per_token == gated_attention + 6 * hidden * 17408
    conv_channels = 2 * 16 * 128 + 48 * 128
    delta_net = 2 * hidden * (conv_channels + 48 * 128 + 2 * 48) + 2 * 48 * 128 * hidden
    delta_net += 2 * 4 * conv_channels + 6 * 48 * 128 * 128
    assert per_layer(flops, 0) == LayerFlops(delta_net + 6 * hidden * 17408)
    # 128k-token sequences: 51.6 GFLOP of projections and MLPs plus 25.8 GFLOP of attention over 16
    # layers per token, not the 148 GFLOP of charging all 64 layers quadratic attention
    seq_len = 131072
    assert 77e9 < flops.fwd_flops([seq_len]) / seq_len < 78e9


def test_qwen3_5_moe_layers_are_all_experts_with_a_gated_shared_expert():
    qwen35_a3b = QWEN38_27B | dict(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        linear_num_value_heads=32,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )
    flops = model_flops_from_hf_config(config(**qwen35_a3b))
    moe = 2 * 2048 * 256 + 6 * 8 * 2048 * 512 + 6 * 2048 * 512
    attention = 2 * 2048 * 4096 * 2 + 4 * 2048 * 512 + 2 * 4096 * 2048
    assert per_layer(flops, 3).per_token == attention + moe


def test_mla_with_dsa_caps_keys_at_topk_and_charges_the_indexer_only_where_it_runs():
    glm5 = config(
        model_type="glm_moe_dsa",
        hidden_size=6144,
        num_attention_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        head_dim=192,
        intermediate_size=12288,
        num_hidden_layers=4,
        vocab_size=154880,
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        indexer_types=["full", "shared", "shared", "full"],
    )
    flops = model_flops_from_hf_config(glm5)
    attention_per_key = 2 * 64 * 256 + 2 * 64 * 256
    assert per_layer(flops, 1).key_terms == (KeyTerm(attention_per_key, cap=2048),)
    indexer = 2 * 2048 * 32 * 128 + 2 * 6144 * 128 + 2 * 6144 * 32
    assert per_layer(flops, 0).per_token - per_layer(flops, 1).per_token == indexer
    assert per_layer(flops, 0).key_terms[-1] == KeyTerm(2 * 32 * 128, indexer=True)
    assert per_layer(flops, 0).indexer_per_token == indexer
    assert per_layer(flops, 1).indexer_per_token == 0


def test_kda_layers_are_one_indexed_and_mla_fills_the_rest():
    kimi_linear = config(
        model_type="kimi_linear",
        hidden_size=2304,
        num_attention_heads=32,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        intermediate_size=9216,
        num_hidden_layers=4,
        vocab_size=163840,
        linear_attn_config={"kda_layers": [1, 2, 3], "num_heads": 32, "head_dim": 128, "short_conv_kernel_size": 4},
    )
    flops = model_flops_from_hf_config(kimi_linear)
    assert [bool(layer.key_terms) for layer in flops.layers] == [False, False, False, True]
    hidden, width = 2304, 32 * 128
    kda = 2 * hidden * (3 * width + 32) + 2 * width * hidden + 2 * (2 * hidden * 128 + 2 * 128 * width)
    kda += 2 * 4 * 3 * width + 6 * 32 * 128 * 128
    assert per_layer(flops, 0).per_token == kda + 6 * hidden * 9216


def test_compressed_attention_layers_read_window_plus_compressed_history():
    deepseek_v4 = config(
        model_type="deepseek_v4",
        hidden_size=4096,
        num_attention_heads=64,
        head_dim=512,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        sliding_window=128,
        compress_ratios=[0, 4, 128, 0],
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        num_hidden_layers=3,
        vocab_size=129280,
    )
    flops = model_flops_from_hf_config(deepseek_v4)
    per_key = 4 * 64 * 512
    assert per_layer(flops, 0).key_terms == (KeyTerm(per_key, cap=128),)
    assert per_layer(flops, 1).key_terms == (
        KeyTerm(per_key, cap=128),
        KeyTerm(per_key, pool=4, cap=512),
        KeyTerm(2 * 64 * 128, pool=4, indexer=True),
    )
    assert per_layer(flops, 2).key_terms == (KeyTerm(per_key, cap=128), KeyTerm(per_key, pool=128))
    assert len(flops.layers) == 3


NEMOTRON_NANO = dict(
    model_type="nemotron_h",
    hidden_size=2688,
    hybrid_override_pattern="MEM*-",
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=128,
    intermediate_size=1856,
    mamba_num_heads=64,
    mamba_head_dim=64,
    ssm_state_size=128,
    n_groups=8,
    conv_kernel=4,
    moe_intermediate_size=1856,
    moe_shared_expert_intermediate_size=3712,
    n_shared_experts=1,
    n_routed_experts=128,
    num_experts_per_tok=6,
    vocab_size=131072,
)


def test_nemotron_h_counts_one_mixer_per_pattern_character_with_two_gemm_mlps():
    flops = model_flops_from_hf_config(config(**NEMOTRON_NANO))
    hidden = 2688
    assert len(flops.layers) == 5
    inner, conv_channels = 64 * 64, 64 * 64 + 2 * 8 * 128
    mamba = 2 * hidden * (inner + conv_channels + 64) + 2 * inner * hidden + 2 * 4 * conv_channels
    mamba += 4 * 64 * 64 * 128
    assert per_layer(flops, 0) == LayerFlops(mamba)
    assert per_layer(flops, 1) == LayerFlops(2 * hidden * 128 + 4 * 6 * hidden * 1856 + 4 * hidden * 3712)
    assert per_layer(flops, 4) == LayerFlops(4 * hidden * 1856)
    assert per_layer(flops, 3).key_terms == (KeyTerm(4 * 32 * 128),)


def test_moe_shared_expert_intermediate_size_is_the_whole_shared_width():
    one, two = (model_flops_from_hf_config(config(**(NEMOTRON_NANO | {"n_shared_experts": n}))) for n in (1, 2))
    assert per_layer(one, 1) == per_layer(two, 1)


def test_latent_moe_runs_routed_experts_in_the_latent_width():
    flops = model_flops_from_hf_config(config(**NEMOTRON_NANO, moe_latent_size=1024))
    hidden = 2688
    expected = 2 * hidden * 128 + 4 * 6 * 1024 * 1856 + 4 * hidden * 1024 + 4 * hidden * 3712
    assert per_layer(flops, 1) == LayerFlops(expected)


# text_config of meta-models/Muse-Spark-Open-816B-A42B, inlined because the repo is gated
MUSE_SPARK_OPEN = {
    "model_type": "muse_spark",
    "text_config": {
        "model_type": "muse_spark_text",
        "hidden_size": 8192,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "intermediate_size": 16384,
        "moe_hidden_size": 4096,
        "moe_intermediate_size": 4096,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "num_hidden_layers": 62,
        "layer_types": ["full_attention" if i % 4 == 1 else "sliding_attention" for i in range(62)],
        "mlp_layer_types": ["sparse"] * 62,
        "sliding_window": 2048,
        "vocab_size": 202048,
    },
}


def test_muse_spark_open_counts_its_output_gate_and_latent_experts():
    flops = model_flops_from_hf_config(MUSE_SPARK_OPEN)
    # 2 x the active matmul params read from the released checkpoint's safetensors headers
    assert flops.per_token == pytest.approx(88865767424, rel=1e-12)
    assert per_layer(flops, 0).key_terms == (KeyTerm(4 * 128 * 64, cap=2048),)
    assert per_layer(flops, 1).key_terms == (KeyTerm(4 * 128 * 64),)


def test_muse_spark_open_flattened_config_keeps_its_output_gate():
    flattened = MUSE_SPARK_OPEN["text_config"] | {"model_type": "muse_spark"}
    assert model_flops_from_hf_config(flattened) == model_flops_from_hf_config(MUSE_SPARK_OPEN)


# (per-token FLOPs without attention keys, forward FLOPs of one 131072-token sequence) of each released config
RELEASED_MODEL_FLOPS = {
    "deepseek-ai/DeepSeek-V3": (73249193984, 52526179793502208),
    "deepseek-ai/DeepSeek-V3.2-Exp": (74952212480, 19740118264840192),
    "deepseek-ai/DeepSeek-V4.1-Flash": (32181125120, 5096652402589696),
    "deepseek-ai/DeepSeek-V4-Flash": (26468679680, 4661701693669376),
    "zai-org/GLM-4.5-Air": (25604653056, 22777974654763008),
    "zai-org/GLM-4.7-Flash": (7158628352, 9206686258561024),
    "zai-org/GLM-5": (81663885312, 17554128204988416),
    "zai-org/GLM-5.2": (80595517440, 13403046071500800),
    "zai-org/GLM-5.3-Flash": (33626390528, 4792992309182464),
    "openai/gpt-oss-120b": (10262200320, 3883326643372032),
    "thinkingmachines/Inkling": (79719383040, 13665937857183744),
    "thinkingmachines/Inkling-Small": (22371270656, 3955825013948416),
    "google/gemma-4-31b-it": (61392027648, 13895362502721536),
    "google/gemma-4-26b-a4b-it": (7643660288, 2464016504979456),
    "XiaomiMiMo/MiMo-V2.6-Flash-RL": (29641146368, 7078529595867136),
    "moonshotai/Kimi-K2.5": (63372132352, 29768942851457024),
    "moonshotai/Kimi-K3": (209019699200, 40063100602286080),
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": (6276415488, 2054124749193216),
    "unsloth/Llama-3.1-8B-Instruct": (15009316864, 6470935167107072),
    "MiniMaxAI/MiniMax-M2.5": (20830224384, 15818945445494784),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": (6502973440, 1696789107310592),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": (24632229888, 4354504132657152),
    "Qwen/Qwen3-30B-A3B": (6083313664, 7552803069231104),
    "Qwen/Qwen3.5-35B-A3B": (5986713600, 2192076145950720),
    "Qwen/Qwen3.5-397B-A17B": (32945438720, 8540381406822400),
    "Qwen/Qwen3.8-Flash-Next": (13497024512, 1874016537149440),
    "Qwen/Qwen3-8B": (15136194560, 7050519528865792),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": (7240155136, 2637844359151616),
}


@pytest.mark.parametrize("repo_id", sorted(RELEASED_MODEL_FLOPS))
def test_released_config_flops(repo_id):
    flops = released_model_flops(repo_id)
    per_token, at_128k = RELEASED_MODEL_FLOPS[repo_id]
    assert flops.per_token == pytest.approx(per_token, rel=1e-9)
    assert flops.fwd_flops([131072]) == pytest.approx(at_128k, rel=1e-9)


def test_deepseek_v4_1_shares_compressed_kv_and_index_scores_from_source_layers():
    flops = released_model_flops("deepseek-ai/DeepSeek-V4.1-Flash")
    window = KeyTerm(4 * 64 * 512, cap=128)
    top_k = KeyTerm(4 * 64 * 512, pool=2, cap=512)
    index_scores = KeyTerm(2 * 32 * 128, pool=2, indexer=True)
    assert per_layer(flops, 0).key_terms == (window,)
    assert per_layer(flops, 2).key_terms == (window, top_k, index_scores)
    assert per_layer(flops, 3).key_terms == (window, top_k)
    compressor = 2 * 5120 * 512 * 2
    indexer = 2 * 1280 * 32 * 128 + 2 * 5120 * 32 + 2 * 512 * 128 / 2
    assert per_layer(flops, 2).per_token - per_layer(flops, 3).per_token == pytest.approx(compressor + indexer)
    assert per_layer(flops, 2).indexer_per_token == pytest.approx(indexer)
    assert per_layer(flops, 3).indexer_per_token == 0
    engram = 2 * 3 * 8 * 256 * 5120 * 5
    assert per_layer(flops, 1).per_token - per_layer(flops, 0).per_token == pytest.approx(engram)


def test_inkling_windows_local_layers():
    flops = released_model_flops("thinkingmachines/Inkling")
    assert per_layer(flops, 0).key_terms == (KeyTerm(4 * 64 * 128, cap=512),)
    assert per_layer(flops, 5).key_terms == (KeyTerm(4 * 64 * 128),)


def test_qwen3_8_next_full_attention_layers_are_sparse_when_the_config_has_an_indexer_budget():
    flops = released_model_flops("Qwen/Qwen3.8-Flash-Next")
    assert per_layer(flops, 0).key_terms == ()
    assert per_layer(flops, 3).key_terms == (
        KeyTerm(4 * 24 * 256, cap=2048),
        KeyTerm(2 * 4 * 128, pool=4, indexer=True),
    )


def test_deepseek_v4_sizes_the_same_from_transformers_per_type_compress_rates():
    with open(hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", "config.json")) as f:
        checkpoint_config = json.load(f)
    ratios = checkpoint_config["compress_ratios"][: checkpoint_config["num_hidden_layers"]]
    layer_type = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
    transformers_config = {k: v for k, v in checkpoint_config.items() if k != "compress_ratios"} | {
        "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        "layer_types": [layer_type[ratio] for ratio in ratios],
        "mlp_layer_types": ["hash_moe"] * 3 + ["moe"] * (len(ratios) - 3),
        "first_k_dense_replace": 3,
    }
    assert model_flops_from_hf_config(transformers_config) == model_flops_from_hf_config(checkpoint_config)


@pytest.mark.parametrize("repo_id", ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"])
def test_deepseek_v4_config_objects_size_like_their_checkpoint_configs(repo_id):
    assert model_flops_from_hf_config(load_hf_config(repo_id)) == released_model_flops(repo_id)


@pytest.mark.parametrize(
    "unknown_structure",
    [{"layer_types": ["chunked_attention"]}, {"mlp_layer_types": ["mystery_mlp"]}, {"hybrid_override_pattern": "X"}],
)
def test_unknown_layer_structure_has_no_flops_definition(unknown_structure):
    with pytest.raises(ValueError, match="no FLOPs definition"):
        model_flops_from_hf_config(config(**(LLAMA_8B | {"num_hidden_layers": 1} | unknown_structure)))


def test_training_runs_a_frozen_indexer_forward_only():
    flops = released_model_flops("deepseek-ai/DeepSeek-V3.2-Exp")
    forward = flops.fwd_flops([131072])
    assert forward / 131072 == pytest.approx(150.61e9, rel=1e-4)
    assert flops.train_flops([131072], trains_indexers=True) == pytest.approx(3 * forward)
    assert flops.train_flops([131072]) / 131072 == pytest.approx(317.41e9, rel=1e-4)


def test_a_model_without_an_indexer_trains_at_three_forwards():
    flops = released_model_flops("Qwen/Qwen3-8B")
    assert flops.train_flops([4096, 8192]) == pytest.approx(3 * flops.fwd_flops([4096, 8192]))
    assert not flops.has_indexers
    assert released_model_flops("deepseek-ai/DeepSeek-V3.2-Exp").has_indexers
