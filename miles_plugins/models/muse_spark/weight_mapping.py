import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


# Checkpoint tensor names (``MuseSparkForConditionalGeneration``) per Megatron role; experts are stacked
# ``[experts, ...]`` tensors, the router is stored ``[experts, hidden]`` with its bias next to it.
_NAMES = {
    "embed": "model.language_model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "final_norm": "model.language_model.norm.weight",
    "layer": "model.language_model.layers.{i}.",
    "input_norm": "input_layernorm.weight",
    "pre_mlp_norm": "pre_feedforward_layernorm.weight",
    "post_attn_norm": "post_attention_layernorm.weight",
    "post_mlp_norm": "post_feedforward_layernorm.weight",
    "attn_gate": "post_attention_residual_gate.gate",
    "mlp_gate": "post_feedforward_residual_gate.gate",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "g": "self_attn.gate_proj.weight",
    "o": "self_attn.o_proj.weight",
    "router": "mlp.gate.weight",
    "router_bias": "mlp.gate.e_score_correction_bias",
    "pooler": "mlp.pre_expert_proj.weight",
    "unpooler": "mlp.post_expert_proj.weight",
    "pre_expert_norm": "mlp.pre_expert_norm.weight",
    "post_expert_norm": "mlp.experts.post_expert_norm.weight",
    "experts_w13": "mlp.experts.gate_up_proj",
    "experts_w2": "mlp.experts.down_proj",
    "dense_gate": "mlp.gate_proj.weight",
    "dense_up": "mlp.up_proj.weight",
    "dense_down": "mlp.down_proj.weight",
}
# Non-text tensors of the omni checkpoint; the text coverage check skips these prefixes.
NON_TEXT_PREFIXES = ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.", "model.audio_tower.")


def rope_pairs_hf_to_kernel(tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Checkpoint rows (per head: first halves, then second halves) -> interleaved rotary pairs.

    Mirrors the SGLang ``load_weights``: ``reshape(heads, 2, head_dim // 2, ...).transpose(1, 2)``. Works for a
    ``[heads*head_dim, in]`` projection weight, a ``[heads*head_dim]`` bias and any head-aligned row slice of them.
    """
    if head_dim % 2 or tensor.shape[0] % head_dim:
        raise ValueError(f"rows {tensor.shape[0]} are not whole heads of even head_dim {head_dim}")
    rest = tensor.shape[1:]
    return tensor.reshape(-1, 2, head_dim // 2, *rest).transpose(1, 2).reshape(tensor.shape).contiguous()


def rope_pairs_kernel_to_hf(tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
    if head_dim % 2 or tensor.shape[0] % head_dim:
        raise ValueError(f"rows {tensor.shape[0]} are not whole heads of even head_dim {head_dim}")
    rest = tensor.shape[1:]
    return tensor.reshape(-1, head_dim // 2, 2, *rest).transpose(1, 2).reshape(tensor.shape).contiguous()


class MuseSpark1xParamMapping:
    """Exact checkpoint mapping with explicit local MCore shard metadata.

    The checkpoint stores the q/k rows of a ``qgkv`` mapping with the two rotary halves consecutive; the kernels
    want them interleaved, so they are re-ordered on the way in and out.
    """

    def __init__(
        self,
        megatron_param,
        hf_param,
        spark,
        *,
        hf_specs,
        kind="direct",
        expert_index=None,
    ):
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self.spark = spark
        self.hf_specs = hf_specs
        self.kind = kind
        self.expert_index = expert_index
        self.partition_dim = None
        self.partition_rank = 0
        self.partition_size = 1
        self.partition_stride = 1

    def hf_slices_to_megatron(self, hf_weights):
        """Pack already selected local HF slices without partitioning them again."""
        if self.kind == "transpose":
            return hf_weights.transpose(0, 1).contiguous()
        if self.kind == "expert":
            return hf_weights.contiguous()
        if self.kind == "gated_mlp":
            return torch.cat((hf_weights["gate"], hf_weights["up"]), dim=0)
        if self.kind == "qgkv":
            groups = self.spark.num_key_value_heads // self.partition_size
            hf_weights = dict(hf_weights)
            for key in ("q", "k"):
                hf_weights[key] = rope_pairs_hf_to_kernel(hf_weights[key], self.spark.head_dim)
            pieces = [hf_weights["q"].reshape(groups, -1, self.spark.hidden_size)]
            if self.spark.use_attn_o_gate:
                pieces.append(hf_weights["g"].reshape(groups, -1, self.spark.hidden_size))
            pieces.extend(hf_weights[key].reshape(groups, -1, self.spark.hidden_size) for key in ("k", "v"))
            return torch.cat(pieces, dim=1).reshape(-1, self.spark.hidden_size)
        return hf_weights

    def megatron_to_hf(self, megatron_weights):
        if megatron_weights is None:
            raise ValueError("Spark export requires the local tensor")
        if self.kind == "transpose":
            return {self.hf_param: megatron_weights.transpose(0, 1).contiguous()}
        if self.kind == "gated_mlp":
            gate, up = megatron_weights.chunk(2, dim=0)
            return {self.hf_param["gate"]: gate, self.hf_param["up"]: up}
        if self.kind == "qgkv":
            groups = self.spark.num_key_value_heads // self.partition_size
            head_dim = self.spark.head_dim
            query_width = self.spark.num_attention_heads // self.spark.num_key_value_heads * head_dim
            sizes = [query_width]
            keys = ["q"]
            if self.spark.use_attn_o_gate:
                sizes.append(query_width)
                keys.append("g")
            sizes.extend((head_dim, head_dim))
            keys.extend(("k", "v"))
            pieces = megatron_weights.reshape(groups, -1, self.spark.hidden_size).split(sizes, dim=1)
            converted = {
                key: piece.reshape(-1, self.spark.hidden_size).contiguous()
                for key, piece in zip(keys, pieces, strict=True)
            }
            for key in ("q", "k"):
                converted[key] = rope_pairs_kernel_to_hf(converted[key], self.spark.head_dim)
            return {self.hf_param[key]: tensor for key, tensor in converted.items()}
        # Expert mappings return one expert; raw export supplies its global expert name.
        return {self.hf_param: megatron_weights}

    def hf_selection(self, name, spec):
        selection = [slice(0, size) for size in spec.shape]
        if self.expert_index is not None:
            selection[0] = self.expert_index
        if self.partition_size > 1:
            axis = 0 if self.kind in ("qgkv", "gated_mlp") else self.partition_dim
            if self.expert_index is not None:
                axis += 1
            width = spec.shape[axis] // self.partition_size
            selection[axis] = slice(self.partition_rank * width, (self.partition_rank + 1) * width)
        return tuple(selection)


def build_mappings(spark, model=None):
    names = _NAMES
    mappings = []
    hidden = spark.hidden_size
    latent = spark.expert_hidden_size
    query = spark.num_attention_heads * spark.head_dim
    kv = spark.num_key_value_heads * spark.head_dim

    parameter_dtype = getattr(torch, spark.parameter_dtype)

    def add(megatron_name, hf_name, shape=(hidden,), *, dtype=parameter_dtype, **kwargs):
        specs = (
            {name: TensorSpec(shape[key], dtype) for key, name in hf_name.items()}
            if isinstance(hf_name, dict)
            else {hf_name: TensorSpec(shape, dtype)}
        )
        mappings.append(MuseSpark1xParamMapping(megatron_name, hf_name, spark, hf_specs=specs, **kwargs))

    add("embedding.word_embeddings.weight", names["embed"], (spark.vocab_size, hidden))
    add("output_layer.weight", names["lm_head"], (spark.vocab_size, hidden))
    if spark.norm_affine:
        add("decoder.final_layernorm.weight", names["final_norm"])
    for index in range(spark.num_hidden_layers):
        mg = f"decoder.layers.{index}."
        hf = names["layer"].format(i=index)
        if spark.norm_affine:
            add(mg + "input_layernorm.weight", hf + names["input_norm"])
            add(mg + "pre_mlp_layernorm.weight", hf + names["pre_mlp_norm"])
        if spark.post_norm_affine:
            add(mg + "post_self_attn_layernorm.weight", hf + names["post_attn_norm"])
        if spark.post_norm_affine or spark.use_ffn_learnable_scales:
            add(mg + "post_mlp_layernorm.weight", hf + names["post_mlp_norm"])
        if spark.use_residual_gate:
            add(mg + "self_attn_bda.gate", hf + names["attn_gate"])
            add(mg + "mlp_bda.gate", hf + names["mlp_gate"])
        projections = {key: hf + names[key] for key in ("q", "k", "v")}
        if spark.use_attn_o_gate:
            projections["g"] = hf + names["g"]
        add(
            mg + "self_attention.linear_qkv.weight",
            projections,
            {key: (query if key in ("q", "g") else kv, hidden) for key in projections},
            kind="qgkv",
        )
        add(mg + "self_attention.linear_proj.weight", hf + names["o"], (hidden, query))
        if spark.layer_is_moe(index):
            add(mg + "mlp.router.weight", hf + names["router"], (spark.num_local_experts, hidden), dtype=torch.float32)
            add(
                mg + "mlp.router.expert_bias",
                hf + names["router_bias"],
                (spark.num_local_experts,),
                dtype=torch.float32,
            )
            if spark.moe_compression_factor != 1:
                add(mg + "mlp.fc1_latent_proj.weight", hf + names["pooler"], (latent, hidden))
                add(mg + "mlp.fc2_latent_proj.weight", hf + names["unpooler"], (hidden, latent))
            if spark.moe_use_pre_expert_norm and spark.use_ffn_learnable_scales:
                add(mg + "mlp.pre_expert_norm.weight", hf + names["pre_expert_norm"], (latent,))
            if spark.moe_use_post_expert_norm and (spark.use_ffn_learnable_scales or spark.post_norm_affine):
                add(mg + "mlp.post_expert_norm.weight", hf + names["post_expert_norm"], (latent,))
            for expert in range(spark.num_local_experts):
                for fc, source, shape in (
                    (1, names["experts_w13"], (spark.num_local_experts, 2 * spark.moe_intermediate_size, latent)),
                    (2, names["experts_w2"], (spark.num_local_experts, latent, spark.moe_intermediate_size)),
                ):
                    add(
                        mg + f"mlp.experts.linear_fc{fc}.weight{expert}",
                        hf + source,
                        shape,
                        kind="expert",
                        expert_index=expert,
                    )
        else:
            add(
                mg + "mlp.linear_fc1.weight",
                {"gate": hf + names["dense_gate"], "up": hf + names["dense_up"]},
                {key: (spark.dense_intermediate_size, hidden) for key in ("gate", "up")},
                kind="gated_mlp",
            )
            add(mg + "mlp.linear_fc2.weight", hf + names["dense_down"], (hidden, spark.dense_intermediate_size))
    return _local_mappings(model, mappings) if model is not None else mappings


def _local_mappings(model, global_mappings):
    groups = model.pg_collection
    if groups.expt_tp.size() != 1:
        raise ValueError("Spark mappings require expert TP1")
    layers = {layer.layer_number - 1: (local, layer) for local, layer in enumerate(model.decoder.layers)}
    if len(layers) != len(model.decoder.layers):
        raise ValueError("duplicate global pipeline layer number")
    result = []
    for mapping in global_mappings:
        name = mapping.megatron_param
        match = re.match(r"decoder\.layers\.(\d+)\.(.*)", name)
        if match:
            global_index = int(match[1])
            if global_index not in layers:
                continue
            local, layer = layers[global_index]
            name = f"decoder.layers.{local}.{match[2]}"
            if mapping.expert_index is not None:
                experts = layer.mlp.local_expert_indices
                if mapping.expert_index not in experts:
                    continue
                local_expert = experts.index(mapping.expert_index)
                name = re.sub(r"weight\d+$", f"weight{local_expert}", name)
        elif name.startswith("embedding.") and not model.pre_process:
            continue
        elif name.startswith(("output_layer.", "decoder.final_layernorm.")) and not model.post_process:
            continue
        mapping.megatron_param = name
        module_name, param_name = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        parameter = getattr(module, param_name)
        if getattr(parameter, "tensor_model_parallel", False):
            group = groups.expt_tp if mapping.expert_index is not None else groups.tp
            mapping.partition_dim = parameter.partition_dim
            mapping.partition_stride = getattr(parameter, "partition_stride", 1)
            mapping.partition_size = group.size()
            mapping.partition_rank = group.rank()
            if mapping.partition_dim not in (0, 1) or mapping.partition_stride not in (1, 2):
                raise ValueError(f"unsupported MCore partition metadata for {name}")
            if mapping.partition_size > 1:
                expected_stride = 2 if mapping.kind == "gated_mlp" else 1
                if mapping.partition_stride != expected_stride:
                    raise ValueError(f"unexpected MCore partition stride for {name}")
        result.append(mapping)
    return result


def hf_tensor_specs(spark):
    return {name: spec for mapping in build_mappings(spark) for name, spec in mapping.hf_specs.items()}
