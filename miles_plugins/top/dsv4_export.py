"""DeepSeek-V4 TOP parameters under the checkpoint names weight sync sends.

TopDSV4Attention keeps Megatron's stock names, which the stock converter handles, except where it holds serving's
fused or processed forms: wqkv_a, the compressor's wkv_gate and load-converted APE, and a grouped-scale FP8 wo_a;
hc_head lives in the block's contraction module. For those, weight sync must send the checkpoint's split names and
representation, which serving's loader fuses and processes again.
"""

import re

import torch

_ATTENTION = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.self_attention\.(.+)")
_HC_HEAD = re.compile(r"module\.module\.decoder\.hc_head_contraction\.(hc_head_\w+)")


def export_dsv4(args, model_name, name, param, quantization_config, packed_weight_basenames):
    """None for a name the stock converter handles; else the HF tensors weight sync sends for it."""
    # Deferred: megatron_to_hf imports this module.
    from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf_stock

    if (top := _checkpoint_form(args, name, param)) is None:
        return None
    renamed, converted = top
    units = [
        unit
        for megatron_name, tensor in renamed
        for unit in convert_to_hf_stock(
            args, model_name, megatron_name, tensor, quantization_config, packed_weight_basenames
        )
    ]
    return units + converted


def _checkpoint_form(args, name, param):
    """(Megatron names for the stock converter and quantizer, HF tensors already in checkpoint form)."""
    if match := _HC_HEAD.fullmatch(name):
        return [(f"module.module.decoder.{match.group(1)}", param)], []
    match = _ATTENTION.fullmatch(name)
    if match is None:
        return None
    layer, rest = match.groups()
    prefix = f"module.module.decoder.layers.{layer}.self_attention."
    if rest == "wqkv_a_linear.weight":
        q, kv = param.split([args.backend.q_lora_rank, param.shape[0] - args.backend.q_lora_rank])
        return [(prefix + "linear_q_down_proj.weight", q), (prefix + "linear_kv_proj.weight", kv)], []
    if rest == "core_attention.compressor.wkv_gate.weight":
        wkv, wgate = param.chunk(2)
        return [
            (prefix + "core_attention.compressor.linear_wkv.weight", wkv),
            (prefix + "core_attention.compressor.linear_wgate.weight", wgate),
        ], []
    if rest == "core_attention.compressor.ape":
        return [(prefix + "core_attention.compressor.ape", checkpoint_ape(param))], []
    if rest == "wo_a_weight":
        # A BF16 wo_a checkpoint: serving holds and loads it unquantized.
        return [], [(f"model.layers.{layer}.self_attn.wo_a.weight", param)]
    if rest == "wo_a_master":
        # A weight-update session keeps wo_a's processed grouped scale layout (_setup_fp8_wo_a_scales), so
        # send that, as Miles sends packed scales for the other FP8 weights.
        from miles_plugins.top.dsv4_attention import wo_a_layout
        from miles_plugins.top.dsv4_runtime import dsv4_hf_config
        from miles_plugins.top.fp8_masters import quantize_checkpoint_format

        weight, scale = wo_a_layout(*quantize_checkpoint_format(param), o_lora_rank=dsv4_hf_config().o_lora_rank)
        hf = f"model.layers.{layer}.self_attn.wo_a"
        return [], [(hf + ".weight", weight), (hf + ".weight_scale_inv", scale)]
    return None


def checkpoint_ape(ape):
    """A compressor APE (or its gradient) in the checkpoint's layout: serving's Compressor.load_ape_weight rearranges
    the overlapping (ratio 4) one."""
    if ape.shape[0] == 4:
        return torch.cat(ape.reshape(2 * ape.shape[0], -1).chunk(2), dim=-1)
    return ape


def is_frozen_checkpoint_key(name):
    """Serving state the TOP trainer never trains, so never exports: the C4 indexer and the hash-routing tables.

    Weight sync must still send it: a colocated engine that releases its weights gets back only what is sent."""
    return not name.startswith("mtp.") and (".attn.indexer." in name or name.endswith(".ffn.gate.tid2eid"))


def frozen_unit_key(name):
    """Serving's loader fuses each compressor's wkv and wgate in one load call; an FP8 weight travels with its scale."""
    name = re.sub(r"\.(wkv|wgate)\.weight$", ".wkv_gate", name)
    return re.sub(r"\.(weight|scale)$", "", name) if ".indexer.wq_b." in name else name


def convert_frozen_unit(args, model_name, quantization_config, unit):
    """Mid-session serving takes FP8 linears in the sync quantizer's form, so the frozen indexer wq_b goes through
    it from its exact BF16 dequantization (power-of-two scales); the rest loads as stored."""
    # Deferred: megatron_to_hf imports this module.
    from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from miles_plugins.top.fp8_masters import dequantize_block_fp8

    tensors = dict(unit)
    out = []
    for name, tensor in unit:
        if match := re.fullmatch(r"layers\.(\d+)\.attn\.indexer\.wq_b\.weight", name):
            megatron = f"module.module.decoder.layers.{match.group(1)}.self_attention.core_attention.indexer.linear_wq_b.weight"
            weight = dequantize_block_fp8(tensor, tensors[name.removesuffix("weight") + "scale"])
            out += convert_to_hf(args, model_name, megatron, weight, quantization_config, None)
        elif not name.endswith(".indexer.wq_b.scale"):
            out.append((name, tensor))
    return out


def checkpoint_units(args, model_name, quantization_config, *, materialize, layers):
    """The frozen serving state, read from --hf-checkpoint; ``layers`` limits it to one pipeline stage."""
    from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import iter_checkpoint_units

    yield from iter_checkpoint_units(
        args.hf_checkpoint,
        select=is_frozen_checkpoint_key,
        unit_key=frozen_unit_key,
        convert=lambda unit: convert_frozen_unit(args, model_name, quantization_config, unit),
        materialize=materialize,
        keep=None if layers is None else (lambda name: _checkpoint_layer(name) in layers),
    )


def _checkpoint_layer(name):
    """0-based decoder layer of a raw checkpoint key ``layers.<i>.``, or None."""
    match = re.match(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def register():
    """Weight sync sends the DS4 TOP parameters under checkpoint names, plus the frozen serving state."""
    from miles.backends.megatron_utils.megatron_to_hf.export_hooks import register_export_hooks

    register_export_hooks(converter=export_dsv4, checkpoint_units=checkpoint_units, frozen_indexer=True)
