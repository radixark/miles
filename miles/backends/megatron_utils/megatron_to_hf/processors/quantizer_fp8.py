import contextlib
import functools
import os
import re
from collections import defaultdict
from collections.abc import Callable

import torch

from miles.backends.megatron_utils.sglang import (
    is_layer_skipped,
    per_block_cast_to_fp8,
    quant_weight_ue8m0,
    should_deepgemm_weight_requant_ue8m0,
    transform_scale_ue8m0,
)
from miles.kernels.quant.fp8_blockwise import fp8_blockwise_cast

from .quantizer_mxfp4 import is_routed_expert_param


def quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "fp8"
    fmt = quantization_config.get("fmt", "e4m3")
    assert fmt == "e4m3", f"Unsupported FP8 format: {fmt}"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)
    is_skipped = _layer_skip_predicate(quantization_config)

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "").replace("mtp_model_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(
                    _quantize_param(args, converted_name, param, weight_block_size, is_skipped)
                )

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(
                    _quantize_param(args, converted_name, param, weight_block_size, is_skipped)
                )

            return quantize_named_params

    fp8_param_names = [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        # DSA indexer
        "self_attention.wq_b.weight",
        # linear attention
        "self_attention.linear_attn.in_proj_qkv.weight",
        "self_attention.linear_attn.in_proj_z.weight",
        "self_attention.linear_attn.out_proj.weight",
        # DeepSeek V4 attention
        "self_attention.linear_kv_proj.weight",
        "self_attention.core_attention.indexer.linear_wq_b.weight",
    ]
    if not getattr(
        args.backend, "indexer_rope_interleave", False
    ):  # config-access-exempt: only DSA model normalization introduces the indexer flag
        # Non-interleaved indexers keep wk as a standalone FP8 parameter in SGLang.
        fp8_param_names.extend(
            [
                "self_attention.wk.weight",
                "self_attention.core_attention.indexer.linear_wk.weight",
            ]
        )

    if rest in fp8_param_names:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(args, converted_name, param, weight_block_size, is_skipped))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def is_kept_in_high_precision(name: str, quantization_config: dict) -> bool:
    """Whether the rollout checkpoint's exclusions keep this HF weight unquantized."""
    return _layer_skip_predicate(quantization_config)(_skip_prefix(name))


def _layer_skip_predicate(quantization_config: dict) -> Callable[[str], bool]:
    return _build_layer_skip_predicate(
        tuple(quantization_config.get("ignored_layers", quantization_config.get("modules_to_not_convert", [])) or []),
        tuple(
            (name, tuple(members))
            for name, members in (quantization_config.get("packed_modules_mapping") or {}).items()
        ),
    )


@functools.lru_cache(maxsize=8)
def _build_layer_skip_predicate(
    ignored_names: tuple[str, ...], packed_modules_mapping: tuple[tuple[str, tuple[str, ...]], ...]
) -> Callable[[str], bool]:
    """Matching a prefix against the checkpoint's ignore list costs one pass over it, and every sync asks again for
    the same prefixes, so each decision is made once per quantization config."""
    ignored_layers = []
    for name in ignored_names:
        name = name.removeprefix("model.")
        ignored_layers.extend((name, f"model.{name}"))
    fused_mapping = {name: list(members) for name, members in packed_modules_mapping}

    @functools.cache
    def is_skipped(prefix: str) -> bool:
        return bool(ignored_layers) and is_layer_skipped(prefix, ignored_layers, fused_mapping=fused_mapping)

    return is_skipped


def _skip_prefix(name: str) -> str:
    prefix = name.removesuffix(".weight")
    if ".experts." in prefix:
        # SGLang chooses one precision for the entire FusedMoE module.
        prefix = prefix.split(".experts.", 1)[0] + ".experts"
    return prefix


def _quantize_param(args, name, weight, weight_block_size, is_skipped):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    # Honor the checkpoint's exclusions: rollout layers kept at their original
    # precision must receive unquantized weights, without FP8 scales.
    if is_skipped(_skip_prefix(name)):
        return [(name, weight)]
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        if _get_scale_format(args, name, weight_block_size) == "ue8m0":
            qweight, scale = quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
            scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        # TODO: this [128, 128] is hacky. need improve
        elif (
            os.environ["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"
            and per_block_cast_to_fp8 is not None
            and list(weight_block_size) == [128, 128]
        ):
            qweight, scale = per_block_cast_to_fp8(weight)
        else:
            qweight, scale = fp8_blockwise_cast(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]


def _get_scale_format(args, name, weight_block_size):
    is_moe_expert = ".experts." in name
    # grouped_moe mirrors SGLang's own MoE requant call: SM120 requantizes only grouped MoE scales.
    if not (
        should_deepgemm_weight_requant_ue8m0
        and should_deepgemm_weight_requant_ue8m0(weight_block_size=weight_block_size, grouped_moe=is_moe_expert)
    ):
        return None  # use default fp32 scale format

    if not is_moe_expert:
        # Non-MoE linear weights: ue8m0 when deepgemm is enabled
        return "ue8m0"

    # MoE expert weights: only ue8m0 when runner is deep_gemm
    is_deepgemm_moe_backend = args.sglang.common_value("moe_runner_backend") == "deep_gemm" or (
        args.sglang.common_value("moe_runner_backend") == "auto"
        and args.sglang.common_value("moe_a2a_backend") in ["deepep", "mooncake"]
    )
    return "ue8m0" if is_deepgemm_moe_backend else None


def _resolve_fp8_mode(args, name, weight_block_size):
    # Mirror _quantize_param's branch order so per-tensor and ue8m0 short-circuit before reading NVTE env.
    if weight_block_size is None:
        return "per_tensor", ".weight_scale"
    if _get_scale_format(args, name, weight_block_size) == "ue8m0":
        return "ue8m0", ".weight_scale_inv"
    if (
        os.environ["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"
        and per_block_cast_to_fp8 is not None
        and list(weight_block_size) == [128, 128]
    ):
        return "per_block", ".weight_scale_inv"
    return "blockwise", ".weight_scale_inv"


def quantize_expert_params_fp8(args, converted_units, quantization_config):
    fmt = quantization_config.get("fmt", "e4m3")
    assert fmt == "e4m3", f"Unsupported FP8 format: {fmt}"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)
    is_skipped = _layer_skip_predicate(quantization_config)

    units = [None] * len(converted_units)
    buckets = defaultdict(list)

    for unit_idx, (megatron_name, params) in enumerate(converted_units):
        if not (megatron_name.startswith("module.module.") and is_routed_expert_param(megatron_name)):
            units[unit_idx] = quantize_params_fp8(args, megatron_name, params, quantization_config)
            continue
        unit_out = []
        for name, weight in params:
            if name.endswith("_scale"):
                continue
            # Delegate exclusions, non-2D/empty shapes, and column-major (.T) views to the scalar path.
            if (
                not name.endswith(".weight")
                or weight.dim() != 2
                or weight.numel() == 0
                or weight.stride(-1) != 1
                or is_skipped(_skip_prefix(name))
            ):
                unit_out.extend(_quantize_param(args, name, weight, weight_block_size, is_skipped))
                continue
            mode, scale_suffix = _resolve_fp8_mode(args, name, weight_block_size)
            pos = len(unit_out)
            unit_out.extend((None, None))
            buckets[(weight.device, weight.dtype, weight.shape, mode)].append(
                (weight, unit_idx, pos, name, name.replace(".weight", scale_suffix))
            )
        units[unit_idx] = unit_out

    for (device, _dtype, _shape, mode), entries in buckets.items():
        weights = [entry[0] for entry in entries]
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            results = _quantize_fp8_batch(weights, mode, weight_block_size)
        for (_, unit_idx, pos, weight_name, scale_name), (qweight, scale) in zip(entries, results, strict=True):
            units[unit_idx][pos] = (weight_name, qweight)
            units[unit_idx][pos + 1] = (scale_name, scale)

    return units


def _quantize_fp8_batch(weights, mode, weight_block_size):
    rows, cols = weights[0].shape
    packed = torch.stack(weights, dim=0)
    if mode == "per_tensor":
        finfo = torch.finfo(torch.float8_e4m3fn)
        scales = packed.abs().amax(dim=(1, 2)).clamp(min=1e-12).to(torch.float32) / finfo.max
        # Match scalar (2-D weight / 0-D fp32 scale) rounding: CUDA casts the 0-D scale first; CPU divides in fp32.
        denom = scales[:, None, None].to(packed.dtype) if packed.is_cuda else scales[:, None, None]
        qweights = (packed / denom).to(packed.dtype).clamp(min=finfo.min, max=finfo.max)
        return list(zip(qweights.to(torch.float8_e4m3fn).unbind(0), scales.unsqueeze(1).unbind(0), strict=True))

    # Pad each expert to block_m so flattened 2-D block kernels never share a block across adjacent experts.
    block_m = weight_block_size[0]
    padded_rows = (rows + block_m - 1) // block_m * block_m
    packed = torch.nn.functional.pad(packed, (0, 0, 0, padded_rows - rows))

    if mode == "ue8m0":
        # SGLang's ue8m0 helpers accept 3-D (B, M, K) directly and preserve TMA scale strides.
        qweights, scales = quant_weight_ue8m0(packed, weight_block_size=weight_block_size)
        scales = transform_scale_ue8m0(scales, mn=rows)
    elif mode == "per_block":
        qweights, scales = per_block_cast_to_fp8(packed.view(-1, cols))
        scales = scales.view(len(weights), -1, scales.shape[-1])
    else:
        qweights, scales = fp8_blockwise_cast(packed.view(-1, cols), weight_block_size)
        scales = scales.view(len(weights), -1, scales.shape[-1])
    qweights = qweights.view(len(weights), padded_rows, cols)[:, :rows]
    return list(zip(qweights.unbind(0), scales.unbind(0), strict=True))
