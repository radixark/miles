"""DeepSeek-V4.1 TOP parameters under the release checkpoint's raw names, in the form serving keeps after a sync.

Serving keeps synced tensors as sent (no loader requantization): 32x32 FP8 projections take the sync quantizer's E4M3
payload and FP32 power-of-two scales (fp8_b32.quantize_b32); wo_a travels the same way and serving's loader
dequantizes it to the BF16 it serves; routed experts take the MXFP4 sync rule followed by serving's cast and UE8M0
requant (dsv41_experts.sync_form); everything else travels as the trainer holds it, BF16 for norms and engram q/k.
"""

import re

import torch

_LAYER = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")
_B32 = {
    "self_attention.wq_a_linear.weight": "attn.wq_a",
    "self_attention.wq_b_linear.weight": "attn.wq_b",
    "self_attention.wkv_linear.weight": "attn.wkv",
    "self_attention.wo_b_linear.weight": "attn.wo_b",
    "self_attention.wo_a_master": "attn.wo_a",
    "engram.linear_wkv.weight": "engram.wkv",
    "mlp.shared_experts.linear_fc2.weight": "ffn.shared_experts.w2",
}
_PLAIN = {
    "input_layernorm.weight": ("attn_norm.weight", torch.bfloat16),
    "pre_mlp_layernorm.weight": ("ffn_norm.weight", torch.bfloat16),
    "self_attention.q_norm_weight": ("attn.q_norm.weight", torch.bfloat16),
    "self_attention.kv_norm": ("attn.kv_norm.weight", torch.bfloat16),
    "self_attention.attn_sink_param": ("attn.attn_sink", torch.float32),
    "self_attention.compressor.wkv.weight": ("attn.compressor.wkv.weight", torch.bfloat16),
    "self_attention.compressor.wgate.weight": ("attn.compressor.wgate.weight", torch.bfloat16),
    "self_attention.compressor.norm.weight": ("attn.compressor.norm.weight", torch.bfloat16),
    "engram.q_weight": ("engram.q_weight", torch.bfloat16),
    "engram.k_weight": ("engram.k_weight", torch.bfloat16),
    "mlp.router.weight": ("ffn.gate.weight", torch.bfloat16),
    "mlp.router.expert_bias": ("ffn.gate.bias", torch.float32),
}
_HC = re.compile(r"(self_attention|mlp)_hyper_connection\.(mapping_proj\.weight|bias|alpha_(?:pre|post|res))")
_EXPERT = re.compile(r"mlp\.experts\.linear_fc([12])\.weight(\d+)")
_TOP_LEVEL = {
    "module.module.embedding.word_embeddings.weight": "embed.weight",
    "module.module.output_layer.weight": "head.weight",
    "module.module.decoder.final_layernorm.weight": "norm.weight",
}


def _fp8_pair(name, weight, scale):
    return [(name + ".weight", weight), (name + ".scale", scale)]


def export_dsv41(args, model_name, name, param, quantization_config, packed_weight_basenames):
    """The HF tensors weight sync sends for a TOP V4.1 parameter; every name is owned here."""
    return _units(name, param, reference=False)


def reference_grads(name, grad):
    """The gradient of a TOP V4.1 parameter under the release's raw names, unquantized and in its own dtype (the
    per-layer VJP's reference, miles.utils.debug_utils.top_reference_dsv41)."""
    return _units(name, grad, reference=True)


def _units(name, param, *, reference):
    """``param`` under checkpoint names: as weight sync sends it, or (``reference``) as is."""
    from miles.backends.megatron_utils.megatron_to_hf.deepseekv4 import _packed_alphas
    from miles_plugins.top.fp8_b32 import quantize_b32

    if name in _TOP_LEVEL:
        return [(_TOP_LEVEL[name], param if reference else param.to(torch.bfloat16))]
    match = _LAYER.fullmatch(name)
    if match is None:
        raise NotImplementedError(f"[top] DeepSeek-V4.1 has no export for {name}")
    layer, rest = match.groups()
    p = f"layers.{layer}."
    if rest in _B32:
        return [(p + _B32[rest] + ".weight", param)] if reference else _fp8_pair(p + _B32[rest], *quantize_b32(param))
    if rest in _PLAIN:
        hf, dtype = _PLAIN[rest]
        return [(p + hf, param if reference else param.to(dtype))]
    if rest == "mlp.shared_experts.linear_fc1.weight":
        names = (p + "ffn.shared_experts.w1", p + "ffn.shared_experts.w3")
        if reference:
            return _halves(names, param)
        weight, scale = quantize_b32(param)
        rows = weight.shape[0] // 2
        return _fp8_pair(names[0], weight[:rows], scale[: rows // 32]) + _fp8_pair(
            names[1], weight[rows:], scale[rows // 32 :]
        )
    if expert := _EXPERT.fullmatch(rest):
        return _export_expert(p, *expert.groups(), param, reference=reference)
    if hc := _HC.fullmatch(rest):
        site = "hc_attn" if hc.group(1) == "self_attention" else "hc_ffn"
        field = hc.group(2)
        if field == "mapping_proj.weight":
            return [(p + site + "_fn", param.float())]
        if field == "bias":
            return [(p + site + "_base", param.float())]
        packed = _packed_alphas(name, param.float())
        return [] if packed is None else [(p + site + "_scale", packed)]
    raise NotImplementedError(f"[top] DeepSeek-V4.1 has no export for {name}")


def _halves(names, param):
    rows = param.shape[0] // 2
    return [(names[0] + ".weight", param[:rows]), (names[1] + ".weight", param[rows:])]


def _export_expert(p, fc, expert, param, *, reference):
    from miles_plugins.top.dsv41_experts import sync_form

    e = f"{p}ffn.experts.{expert}."
    if reference:
        return [(e + "w2.weight", param)] if fc == "2" else _halves((e + "w1", e + "w3"), param)
    weight, scale = sync_form(param)
    if fc == "2":
        return _fp8_pair(e + "w2", weight, scale)
    rows = weight.shape[0] // 2
    return _fp8_pair(e + "w1", weight[:rows], scale[:rows]) + _fp8_pair(e + "w3", weight[rows:], scale[rows:])


def convert_frozen_unit(unit):
    """Serving keeps a synced FP8 projection as sent, so the frozen indexer wq_b travels in the sync quantizer's form,
    requantized from its exact BF16 dequantization (power-of-two scales, the same values); the rest as stored."""
    from miles_plugins.top.fp8_b32 import dequantize_b32, quantize_b32

    tensors = dict(unit)
    out = []
    for name, tensor in unit:
        if name.endswith(".indexer.wq_b.weight"):
            base = name.removesuffix(".weight")
            out += _fp8_pair(base, *quantize_b32(dequantize_b32(tensor, tensors[base + ".scale"])))
        elif not name.endswith(".indexer.wq_b.scale"):
            out.append((name, tensor))
    return out


def checkpoint_units(args, model_name, quantization_config, *, materialize, layers):
    """The frozen serving state the TOP trainer holds no parameters for (the indexers: TOP replays serving's
    selections), read from --hf-checkpoint and sent on every sync; ``layers`` limits it to one pipeline stage."""
    from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import iter_checkpoint_units
    from miles_plugins.top.dsv4_export import _checkpoint_layer, frozen_unit_key, is_frozen_checkpoint_key

    yield from iter_checkpoint_units(
        args.hf_checkpoint,
        select=is_frozen_checkpoint_key,
        unit_key=frozen_unit_key,
        convert=convert_frozen_unit,
        materialize=materialize,
        keep=None if layers is None else (lambda name: _checkpoint_layer(name) in layers),
    )


def register():
    """Weight sync sends the TOP V4.1 parameters under checkpoint names, plus the frozen indexers: a colocated engine
    that releases its weights gets back only what is sent. The DSpark draft is not sent; serving keeps it in its
    draft-weights CPU backup."""
    from miles.backends.megatron_utils.megatron_to_hf.export_hooks import register_export_hooks

    register_export_hooks(converter=export_dsv41, checkpoint_units=checkpoint_units)
