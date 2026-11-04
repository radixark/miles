import json
import re
from functools import cache

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix

GLOBAL_PARAM_NAMES = {
    "embedding.word_embeddings.weight": "model.llm.embed.weight",
    "embed_norm.weight": "model.llm.embed_norm.weight",
    "decoder.final_layernorm.weight": "model.llm.norm.weight",
    "output_layer.weight": "model.llm.unembed.weight",
}

# Megatron leaf under decoder.layers.{i}. -> HF leaf under model.llm.layers.{i}.
LAYER_PARAM_NAMES = {
    "attn_norm.weight": "attn_norm.weight",
    "q_input_norm.weight": "q_input_norm.weight",
    "k_input_norm.weight": "k_input_norm.weight",
    "v_input_norm.weight": "v_input_norm.weight",
    "pre_mlp_layernorm.weight": "mlp_norm.weight",
    "mlp_sconv_weight": "mlp_sconv.weight",
    "self_attention.attn_sconv_weight": "attn_sconv.weight",
    "self_attention.linear_k.weight": "attn.wk_dv.weight",
    "self_attention.linear_v.weight": "attn.wv_dv.weight",
    "self_attention.linear_proj.weight": "attn.wo_ud.weight",
    "self_attention.q_l2_clip_scale": "attn.q_l2_clip.scale",
    "self_attention.rel_l2_clip_scale": "attn.rel_l2_clip.scale",
    "self_attention.rel_proj": "attn.rel_logits_proj.proj",
    "self_attention.k_sconv_weight": "attn.k_sconv.weight",
    "self_attention.v_sconv_weight": "attn.v_sconv.weight",
    "self_attention.k_sconv_static_w": "attn.k_sconv.static_w",
    "self_attention.v_sconv_static_w": "attn.v_sconv.static_w",
    "self_attention.k_sconv_weight_proj": "attn.k_sconv.weight_proj",
    "self_attention.v_sconv_weight_proj": "attn.v_sconv.weight_proj",
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.router.expert_bias": "mlp.gate.bias",
}

# linear_q packs the Q-side projections in this order.
LINEAR_Q_PARTS = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj")


@cache
def load_inkling2_text_config(hf_checkpoint: str) -> dict:
    with open(f"{hf_checkpoint}/config.json") as f:
        return json.load(f)["text_config"]


def inkling2_quantization_config(hf_checkpoint: str) -> dict:
    """The checkpoint's expert precision (hf_quant_config.json): HF expert tensor -> compute dtype."""
    with open(f"{hf_checkpoint}/hf_quant_config.json") as f:
        quant_config = json.load(f)
    assert quant_config["quant_method"] == "inkling2", quant_config["quant_method"]
    return {
        "quant_method": "inkling2",
        "expert_formats": {
            tensor: group["compute_dtype"]
            for group in quant_config["model"]["parameter_groups"]
            for tensor in group["tensors"]
            if tensor.endswith((".w13_weight", ".w2_weight"))
        },
    }


def linear_q_split(text_cfg: dict, layer: int) -> list[int]:
    """Row sizes of linear_q's parts for one layer (local layers use the swa_* geometry)."""
    is_local = layer in text_cfg["local_layer_ids"]
    num_heads = text_cfg["swa_num_attention_heads" if is_local else "num_attention_heads"]
    head_dim = text_cfg["swa_head_dim" if is_local else "head_dim"]
    return [num_heads * head_dim, num_heads * text_cfg["d_rel"], num_heads * head_dim, num_heads]


def convert_inkling2_to_hf(args, name, param):
    name = strip_param_name_prefix(name)
    if name in GLOBAL_PARAM_NAMES:
        return [(GLOBAL_PARAM_NAMES[name], param)]

    layer, rest = re.match(r"decoder\.layers\.(\d+)\.(.+)", name).groups()
    layer = int(layer)
    prefix = f"model.llm.layers.{layer}."
    if rest in LAYER_PARAM_NAMES:
        return [(prefix + LAYER_PARAM_NAMES[rest], param)]
    if rest == "attn_query":
        return [(f"model.llm.attn_res.attn_queries.{layer - 1}", param)]
    if rest == "mlp_query":
        return [(f"model.llm.attn_res.mlp_queries.{layer}", param)]
    if rest == "output_query":
        return [("model.llm.attn_res.output_query", param)]
    if rest == "self_attention.linear_q.weight":
        parts = param.split(linear_q_split(load_inkling2_text_config(args.hf_checkpoint), layer), dim=0)
        return [(f"{prefix}attn.{part}.weight", w) for part, w in zip(LINEAR_Q_PARTS, parts, strict=True)]

    expert = re.match(r"mlp\.experts\.linear_fc(1|2)\.weight(\d+)", rest)
    assert expert is not None, f"convert_inkling2_to_hf: unhandled parameter {name!r}"
    fc, expert_id = expert.groups()
    expert_prefix = f"{prefix}mlp.experts.{expert_id}."
    if fc == "2":
        return [(expert_prefix + "down_proj.weight", param)]
    gate, up = param.chunk(2, dim=0)
    return [(expert_prefix + "gate_proj.weight", gate), (expert_prefix + "up_proj.weight", up)]
