"""Scratch structural test: Glm5NextBridge name-mapping completeness (no GPU).

Generates the expected HF tensor-name list for the 8-layer GLM-5.3-Flash cut
(layers 0-7: KDA on 0,1,2,4,5,6; DSA on 3,7; dense MLP on 0-2; 288-expert MoE
with one shared expert on 3-7; mHC on every layer) purely from the architecture
knowledge in the design doc, writes it as a safetensors-index-style JSON, then
audits the bridge both ways:

* every expected mcore parameter name resolves through
  ``_weight_name_mapping_mcore_to_hf`` without raising (the qwen3.8 audit
  lesson: an unmapped-but-built param aborts the load, a mapped-but-unbuilt one
  is silently dropped);
* every HF name the bridge produces is in the expected list;
* the expected list is covered exactly, so no checkpoint tensor is silently
  ignored (``visual.*``, MTP ``model.layers.8.*`` and ``hc_head_*`` are the
  deliberate exclusions -- untrained tower, untrained MTP, and the orphan head
  contraction the spec replaces with a plain mean).

Usage: python tests/glm5_next/test_bridge_names.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

NUM_LAYERS = 8
DSA_LAYERS = (3, 7)
FIRST_K_DENSE = 3
NUM_EXPERTS = 288

_KDA_TENSORS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_conv1d.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.b_proj.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.g_a_proj.weight",
    "self_attn.g_b_proj.weight",
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.o_norm.weight",
    "self_attn.o_proj.weight",
]

_DSA_TENSORS = [
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.index_kpool_compress_gate",
    "self_attn.indexer.index_kpool_compress_ape",
]

_HC_TENSORS = [
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
]

_DENSE_MLP_TENSORS = [
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def _moe_tensors():
    tensors = [
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    ]
    for expert in range(NUM_EXPERTS):
        tensors += [
            f"mlp.experts.{expert}.gate_proj.weight",
            f"mlp.experts.{expert}.up_proj.weight",
            f"mlp.experts.{expert}.down_proj.weight",
        ]
    return tensors


def expected_hf_names() -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
        attn = _DSA_TENSORS if layer in DSA_LAYERS else _KDA_TENSORS
        names.update(f"{prefix}.{t}" for t in attn)
        names.update(f"{prefix}.{t}" for t in _HC_TENSORS)
        mlp = _DENSE_MLP_TENSORS if layer < FIRST_K_DENSE else _moe_tensors()
        names.update(f"{prefix}.{t}" for t in mlp)
    return names


def expected_mcore_names() -> list[str]:
    names = ["embedding.word_embeddings.weight", "decoder.final_layernorm.weight", "output_layer.weight"]
    for layer in range(NUM_LAYERS):
        prefix = f"decoder.layers.{layer}"
        names.append(f"{prefix}.input_layernorm.weight")
        if layer in DSA_LAYERS:
            names += [
                f"{prefix}.self_attention.linear_q_down_proj.weight",
                f"{prefix}.self_attention.linear_q_up_proj.layer_norm_weight",
                f"{prefix}.self_attention.linear_q_up_proj.weight",
                f"{prefix}.self_attention.linear_kv_down_proj.weight",
                f"{prefix}.self_attention.linear_kv_up_proj.layer_norm_weight",
                f"{prefix}.self_attention.linear_kv_up_proj.weight",
                f"{prefix}.self_attention.linear_proj.weight",
                f"{prefix}.self_attention.wq_b.weight",
                f"{prefix}.self_attention.wk.weight",
                f"{prefix}.self_attention.weights_proj.weight",
                f"{prefix}.self_attention.k_norm.weight",
                f"{prefix}.self_attention.k_norm.bias",
                f"{prefix}.self_attention.index_kpool_compress_gate",
                f"{prefix}.self_attention.index_kpool_compress_ape",
            ]
        else:
            names += [
                f"{prefix}.self_attention.kda.{t}"
                for t in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "conv1d.weight",
                    "b_proj.weight",
                    "f_a_proj.weight",
                    "f_b_proj.weight",
                    "g_a_proj.weight",
                    "g_b_proj.weight",
                    "A_log",
                    "dt_bias",
                    "o_norm.weight",
                    "o_proj.weight",
                ]
            ]
        for site in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            names += [
                f"{prefix}.{site}.mapping_proj.weight",
                f"{prefix}.{site}.bias",
                f"{prefix}.{site}.alpha_pre",
                f"{prefix}.{site}.alpha_post",
                f"{prefix}.{site}.alpha_res",
            ]
        if layer < FIRST_K_DENSE:
            names += [
                f"{prefix}.mlp.linear_fc1.layer_norm_weight",
                f"{prefix}.mlp.linear_fc1.weight",
                f"{prefix}.mlp.linear_fc2.weight",
            ]
        else:
            names += [
                f"{prefix}.pre_mlp_layernorm.weight",
                f"{prefix}.mlp.router.weight",
                f"{prefix}.mlp.router.expert_bias",
                f"{prefix}.mlp.shared_experts.linear_fc1.weight",
                f"{prefix}.mlp.shared_experts.linear_fc2.weight",
            ]
            for expert in range(NUM_EXPERTS):
                names += [
                    f"{prefix}.mlp.experts.linear_fc1.weight{expert}",
                    f"{prefix}.mlp.experts.linear_fc2.weight{expert}",
                ]
    return names


def main():
    from miles_plugins.mbridge.glm5_next import Glm5NextBridge

    names_json = Path(__file__).parent / "glm5_3_flash_8layer_hf_names.json"
    expected_hf = expected_hf_names()
    names_json.write_text(json.dumps(sorted(expected_hf), indent=1))
    print(f"wrote {len(expected_hf)} expected HF tensor names to {names_json}")

    bridge = object.__new__(Glm5NextBridge)
    produced_hf = set()
    failures = []
    for mcore_name in expected_mcore_names():
        try:
            hf_names = bridge._weight_name_mapping_mcore_to_hf(mcore_name)
        except Exception as exc:
            failures.append(f"{mcore_name}: {type(exc).__name__}: {exc}")
            continue
        for hf_name in hf_names:
            if hf_name not in expected_hf:
                failures.append(f"{mcore_name} -> {hf_name} not an expected HF tensor")
            produced_hf.add(hf_name)

    uncovered = expected_hf - produced_hf
    assert not failures, "mapping failures:\n" + "\n".join(failures)
    assert not uncovered, f"expected HF tensors never produced by any mcore param: {sorted(uncovered)}"
    print(f"PASS bridge maps {len(produced_hf)} HF tensors, complete both ways")


if __name__ == "__main__":
    main()
