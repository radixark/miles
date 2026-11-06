import re
from functools import lru_cache

import torch

from miles.utils.nvfp4 import (
    NVFP4_GROUP_SIZE,
    nvfp4_quantize_1d,
    nvfp4_quantize_1d_batch,
    nvfp4_quantize_1d_pair,
    nvfp4_quantize_1d_pair_batch,
)
from miles.utils.timer import Timer

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


@lru_cache(maxsize=16)
def _literal_ignore_rules(ignore_rules: tuple[str, ...]) -> frozenset[str] | None:
    # Preserve the original ordered matching (including short-circuit behavior)
    # for regex policies. Snapshot keys also invalidate after config mutations.
    if any(rule.startswith("re:") for rule in ignore_rules):
        return None
    return frozenset(ignore_rules)


def _is_ignored(name: str, ignore_rules: list[str], literal_rules: frozenset[str] | None = None) -> bool:
    if literal_rules is not None:
        while True:
            if name in literal_rules:
                return True
            name, separator, _ = name.rpartition(".")
            if not separator:
                return False
    for rule in ignore_rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return True
            continue
        if name == rule or name.startswith(f"{rule}."):
            return True
    return False


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    _validate_nvfp4_config(args, quantization_config)
    if _is_nvfp4_expert(args, megatron_name):
        return _quantize_moe_params(converted_named_params, _get_ignore_rules(quantization_config))
    return converted_named_params


def _validate_nvfp4_config(args, quantization_config):
    assert quantization_config is not None
    assert quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4"
    if bool(
        getattr(args.backend, "fp4_param", False) or getattr(args.backend, "fp4_param_gather", False)
    ):  # config-access-exempt: FP4 option names differ across Megatron versions
        raise NotImplementedError("fp4-param-gather is unsupported for Miles NVFP4 checkpoint export.")


def _is_nvfp4_expert(args, megatron_name):
    if args.extra_high_precision_layers_megatron:
        for layer_name in args.extra_high_precision_layers_megatron:
            if layer_name in megatron_name:
                return False

    # Only the main language decoder is eligible; MTP and modality towers stay in high precision.
    decoder_layers_pattern = r"(?:module\.)*(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)
    if not match:
        return False
    layer_idx, rest = match.groups()

    # Skip quantization for BF16 tail of main decoder layers.
    if getattr(
        args.backend, "first_last_layers_bf16", False
    ):  # config-access-exempt: BF16 layer options are absent in older Megatron parsers
        num_layers = int(args.backend.num_layers)
        num_layers_at_start_in_bf16 = int(
            getattr(args.backend, "num_layers_at_start_in_bf16", 0)
        )  # config-access-exempt: BF16 layer options are absent in older Megatron parsers
        num_layers_at_end_in_bf16 = int(
            getattr(args.backend, "num_layers_at_end_in_bf16", 0)
        )  # config-access-exempt: BF16 layer options are absent in older Megatron parsers
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return False

    # routed experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _ = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            return True

    return False


def quantize_expert_params_nvfp4(args, converted_units, quantization_config):
    """Quantize one bounded expert batch while retaining each source's HF unit."""
    # This path bypasses quantize_params, so accumulate its host time once here.
    with Timer().accumulate("update_weights_quantization", parent="update_weights"):
        return _quantize_expert_params_nvfp4(args, converted_units, quantization_config)


def _quantize_expert_params_nvfp4(args, converted_units, quantization_config):
    _validate_nvfp4_config(args, quantization_config)
    ignore_rules = _get_ignore_rules(quantization_config)
    literal_rules = _literal_ignore_rules(tuple(ignore_rules))
    units = [params for _, params in converted_units]
    selected = []
    pairs, pair_targets = [], []
    singles, single_targets = [], []
    for index, (megatron_name, params) in enumerate(converted_units):
        if not _is_nvfp4_expert(args, megatron_name):
            continue
        eligible = [
            (name, weight)
            for name, weight in params
            if _should_quantize_param(name, weight, ignore_rules, literal_rules)
        ]
        # Direct converters emit matrices. Keep the scalar processor's handling
        # of stacked/other ranks for callers that already supply those layouts.
        if any(weight.dim() != 2 for _, weight in eligible):
            units[index] = _quantize_moe_params(params, ignore_rules)
            continue
        candidates = _gated_candidates(params, ignore_rules, literal_rules)
        paired_names = set()
        for roles in candidates.values():
            gate_name, gate = roles["gate"]
            up_name, up = roles["up"]
            pairs.append((gate, up))
            pair_targets.append((index, gate_name, up_name))
            paired_names.update((gate_name, up_name))
        for name, weight in eligible:
            if name not in paired_names:
                singles.append(weight)
                single_targets.append((index, name))
        selected.append(index)

    outputs = [{} for _ in units]
    pair_outputs = nvfp4_quantize_1d_pair_batch(pairs) if pairs else []
    for (index, gate_name, up_name), (gate, up) in zip(pair_targets, pair_outputs, strict=True):
        outputs[index][gate_name] = gate
        outputs[index][up_name] = up
    single_outputs = nvfp4_quantize_1d_batch(singles) if singles else []
    for (index, name), output in zip(single_targets, single_outputs, strict=True):
        outputs[index][name] = output
    for index in selected:
        units[index] = _format_quantized_params(units[index], outputs[index])
    return units


def _quantize_moe_params(converted_named_params, ignore_rules):
    # Build/hash the policy once per conversion batch, not once per weight check.
    literal_rules = _literal_ignore_rules(tuple(ignore_rules))
    candidates = _gated_candidates(converted_named_params, ignore_rules, literal_rules)
    outputs = {}
    for roles in candidates.values():
        gate_name, gate_weight = roles["gate"]
        up_name, up_weight = roles["up"]
        outputs[gate_name], outputs[up_name] = nvfp4_quantize_1d_pair(gate_weight, up_weight)
    quantized_params = []
    for name, param in converted_named_params:
        if not _should_quantize_param(name, param, ignore_rules, literal_rules):
            quantized_params.append((name, param))
            continue
        output = outputs[name] if name in outputs else quantize_nvfp4(param)
        quantized_params.extend(_named_quantized_params(name, output))
    return quantized_params


def _gated_candidates(converted_named_params, ignore_rules, literal_rules):
    candidates = {}
    for converted_name, param in converted_named_params:
        base, role = _split_gated_pair_name(converted_name)
        if base is None or role is None:
            continue
        if _should_quantize_param(converted_name, param, ignore_rules, literal_rules):
            roles = candidates.setdefault(base, {})
            if role in roles:
                raise ValueError(
                    f"NVFP4 requires a single complete gate/up pair per conversion batch; "
                    f"found duplicate {role} tensor for {base}."
                )
            roles[role] = (converted_name, param)

    for base, roles in candidates.items():
        if set(roles) != {"gate", "up"}:
            present = ", ".join(sorted(roles))
            raise ValueError(
                f"NVFP4 requires gate/up tensors to be quantized together so they can share "
                f"one global amax; found only {{{present}}} for {base}."
            )
    return candidates


def _format_quantized_params(converted_named_params, outputs):
    quantize_named_params = []
    for converted_name, param in converted_named_params:
        if converted_name not in outputs:
            quantize_named_params.append((converted_name, param))
            continue
        quantize_named_params.extend(_named_quantized_params(converted_name, outputs[converted_name]))

    return quantize_named_params


def _named_quantized_params(name, output):
    qweight, block_scale, weight_scale_2 = output
    return [
        (name, qweight),
        (name.replace(".weight", ".weight_scale"), block_scale),
        (name.replace(".weight", ".weight_scale_2"), weight_scale_2),
    ]


def _should_quantize_param(name, weight, ignore_rules, literal_rules=None):
    if ignore_rules and _is_ignored(name, ignore_rules, literal_rules):
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


def _quantize_nvfp4_1d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NVFP4 1D quantization (tile shape = 1x16).

    Returns:
      qweight: uint8 packed fp4, shape (M, K // 2)
      block_scale: float8_e4m3fn, shape (M, K // 16)
      global_scale: float32 scalar tensor
    """
    weight = weight.contiguous()
    _, n = weight.shape
    if n % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 requires K divisible by {NVFP4_GROUP_SIZE}, got {n}.")

    return nvfp4_quantize_1d(weight)


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() == 2:
        return _quantize_nvfp4_1d(weight)
    if weight.dim() == 3:
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
