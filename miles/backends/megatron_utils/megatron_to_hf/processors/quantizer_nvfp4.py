import re

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16

GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}


def _get_ignore_rules(quantization_config) -> list[str]:
    ignore_rules = quantization_config.get("ignore", []) or []
    if isinstance(ignore_rules, str):
        ignore_rules = [ignore_rules]
    exclude_rules = quantization_config.get("exclude_modules", []) or []
    if isinstance(exclude_rules, str):
        exclude_rules = [exclude_rules]
    return list(ignore_rules) + [rule for rule in exclude_rules if rule not in ignore_rules]


def _is_ignored(name: str, ignore_rules: list[str]) -> bool:
    for rule in ignore_rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return True
            continue
        if name == rule or name.startswith(f"{rule}."):
            return True
    return False


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config is not None
    assert quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4"

    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in megatron_name:
                return converted_named_params

    ignore_rules = _get_ignore_rules(quantization_config)

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Skip quantization for BF16 tail of main decoder layers.
    if getattr(args, "first_last_layers_bf16", False):
        num_layers = int(args.num_layers)
        num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0))
        num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0))
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return converted_named_params

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _ = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            return _quantize_moe_params(converted_named_params, ignore_rules)

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            return _quantize_moe_params(converted_named_params, ignore_rules)

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_moe_params(converted_named_params, ignore_rules):
    shared_global_amax = {}
    gated_candidates = {}
    for converted_name, param in converted_named_params:
        base, role = _split_gated_pair_name(converted_name)
        if base is None or role is None:
            continue
        if _should_quantize_param(converted_name, param, ignore_rules):
            roles = gated_candidates.setdefault(base, {})
            if role in roles:
                raise ValueError(
                    f"NVFP4 requires a single complete gate/up pair per conversion batch; "
                    f"found duplicate {role} tensor for {base}."
                )
            roles[role] = param

    for base, roles in gated_candidates.items():
        if set(roles) != {"gate", "up"}:
            present = ", ".join(sorted(roles))
            raise ValueError(
                f"NVFP4 requires gate/up tensors to be quantized together so they can share "
                f"one global amax; found only {{{present}}} for {base}."
            )
        gate_amax = roles["gate"].abs().max().to(torch.float32)
        up_amax = roles["up"].abs().max().to(torch.float32)
        shared_global_amax[base] = torch.max(gate_amax, up_amax)

    quantize_named_params = []
    for converted_name, param in converted_named_params:
        if not _should_quantize_param(converted_name, param, ignore_rules):
            quantize_named_params.append((converted_name, param))
            continue
        base, _role = _split_gated_pair_name(converted_name)
        global_amax = shared_global_amax.get(base) if base else None
        qweight, block_scale, weight_scale_2 = quantize_nvfp4(param, global_amax=global_amax)
        quantize_named_params.append((converted_name, qweight))
        quantize_named_params.append((converted_name.replace(".weight", ".weight_scale"), block_scale))
        quantize_named_params.append((converted_name.replace(".weight", ".weight_scale_2"), weight_scale_2))
        quantize_named_params.append(
            (converted_name.replace(".weight", ".input_scale"), torch.ones_like(weight_scale_2, dtype=torch.float32))
        )

    return quantize_named_params


def _should_quantize_param(name, weight, ignore_rules):
    if ignore_rules and _is_ignored(name, ignore_rules):
        return False
    if not name.endswith(".weight"):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} for NVFP4 ({name}).")
    return True


def _split_gated_pair_name(name: str):
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def _nvfp4_global_decode_scale_te(global_amax: torch.Tensor) -> torch.Tensor:
    fp4_max = torch.tensor(FP4_E2M1_MAX, device=global_amax.device, dtype=torch.float32)
    fp8_max = torch.tensor(FP8_E4M3_MAX, device=global_amax.device, dtype=torch.float32)
    global_encode_scale = torch.div(fp8_max * fp4_max, global_amax.to(torch.float32))
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_encode_scale.device,
            dtype=torch.float32,
        ),
    )
    if global_encode_scale.numel() == 1:
        if global_encode_scale == torch.tensor(0.0, device=global_amax.device, dtype=torch.float32):
            global_encode_scale = torch.tensor(1.0, device=global_amax.device, dtype=torch.float32)
    else:
        global_encode_scale = torch.where(
            global_encode_scale == 0.0,
            torch.ones_like(global_encode_scale),
            global_encode_scale,
        )
    return torch.div(1.0, global_encode_scale)


def _quantize_nvfp4_1d(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NVFP4 1D quantization (tile shape = 1x16), adapted from
    TransformerEngine NVFP4QuantizerRef._quantize_blockwise_reference.

    Returns:
      qweight: uint8 packed fp4, shape (M, K // 2)
      block_scale: float8_e4m3fn, shape (M, K // 16)
      global_scale: float32 scalar tensor
    """
    weight = weight.contiguous()
    m, n = weight.shape
    if n % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 requires K divisible by {NVFP4_GROUP_SIZE}, got {n}.")

    if global_amax is None:
        global_amax = torch.max(torch.abs(weight.to(torch.float32)))
    else:
        global_amax = global_amax.to(device=weight.device, dtype=torch.float32)

    from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef

    qweight, block_scale = NVFP4QuantizerRef._quantize_blockwise_reference(
        weight,
        global_amax,
        NVFP4_GROUP_SIZE,
        1,
        pow_2_scales=False,
        eps=0.0,
    )
    return qweight, block_scale, _nvfp4_global_decode_scale_te(global_amax)


def quantize_nvfp4(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() == 2:
        return _quantize_nvfp4_1d(weight, global_amax=global_amax)
    if weight.dim() == 3:
        if global_amax is not None:
            raise ValueError("global_amax override is only supported for 2D weights.")
        qweights = []
        block_scales = []
        global_scales = []
        for idx in range(weight.shape[0]):
            qweight, block_scale, global_scale = _quantize_nvfp4_1d(weight[idx])
            qweights.append(qweight)
            block_scales.append(block_scale)
            global_scales.append(global_scale)
        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )
    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")
