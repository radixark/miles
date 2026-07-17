"""
python tools/convert_hf_to_nvfp4.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR]
                                   [--device DEVICE]
                                   [--num-layers-at-start-in-bf16 NUM_LAYERS_AT_START_IN_BF16]
                                   [--num-layers-at-end-in-bf16 NUM_LAYERS_AT_END_IN_BF16]
                                   [--extra-high-precision-layers-hf ...]

Convert a BF16/FP16/FP32 HF safetensors checkpoint to NVFP4 (E2M1) for MoE
expert GEMMs only. Dense linear layers are left unmodified.
Use --extra-high-precision-layers-hf to keep additional HF weight-name
substrings unquantized.

This follows the NVFP4 reference quantization in Transformer Engine and uses
1D block scaling (NVTE_NVFP4_1D_SCALING, group size = 16).
"""

import argparse
import gc
import json
import os
import shutil

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
DEFAULT_KV_CACHE_SCHEME = {"dynamic": False, "num_bits": 8, "type": "float"}
DEFAULT_KV_CACHE_QUANT_ALGO = "FP8"

EXPERT_WEIGHT_SUFFIXES = (
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
)

EXPERT_NAME_MARKERS = (
    ".experts.",
    ".shared_experts.",
    "block_sparse_moe.experts.",
    ".moe.experts.",
)

FUSED_QKV_SUFFIXES = (".q_proj", ".k_proj", ".v_proj")
GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}


def _is_moe_expert_weight_name(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if not any(marker in name for marker in EXPERT_NAME_MARKERS):
        return False
    return any(name.endswith(suffix) for suffix in EXPERT_WEIGHT_SUFFIXES)


def _get_num_hidden_layers(model_dir: str) -> int:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(
            "config.json is required to use --num-layers-at-start-in-bf16 or --num-layers-at-end-in-bf16."
        )
    cfg = json.load(open(config_path))
    num_layers = cfg.get("num_hidden_layers")
    if num_layers is None and isinstance(cfg.get("text_config"), dict):
        num_layers = cfg["text_config"].get("num_hidden_layers")
    if num_layers is None:
        raise ValueError("num_hidden_layers not found in config.json.")
    return int(num_layers)


def should_quantize(
    name: str,
    weight: torch.Tensor,
    skip_weight_substrings: tuple[str, ...] = (),
) -> bool:
    if any(substr in name for substr in skip_weight_substrings):
        return False
    if not _is_moe_expert_weight_name(name):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} " f"for NVFP4 quantization ({name})."
        )
    return True


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


class ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(self, filename: str, q_weights: dict[str, torch.Tensor], module_names: list[str]) -> None:
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def _update_quantization_config(cfg: dict, ignore_list: list[str]) -> None:
    quant_cfg = cfg.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}

    quant_cfg["quant_algo"] = "NVFP4"
    quant_cfg["quant_method"] = "modelopt"
    quant_cfg["group_size"] = NVFP4_GROUP_SIZE
    quant_cfg["ignore"] = ignore_list
    quant_cfg.setdefault("kv_cache_scheme", DEFAULT_KV_CACHE_SCHEME)

    config_groups = quant_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        for group in config_groups.values():
            if not isinstance(group, dict):
                continue
            group.setdefault("targets", ["Linear"])
            for key in ("input_activations", "weights"):
                section = group.get(key)
                if not isinstance(section, dict):
                    continue
                section.setdefault("dynamic", False)
                section.setdefault("num_bits", 4)
                section.setdefault("type", "float")
                section["group_size"] = NVFP4_GROUP_SIZE

    cfg["quantization_config"] = quant_cfg


def _write_hf_quant_config(output_path: str, ignore_list: list[str], input_path: str) -> None:
    hf_quant_path = os.path.join(input_path, "hf_quant_config.json")
    if os.path.exists(hf_quant_path):
        with open(hf_quant_path) as f:
            hf_quant_cfg = json.load(f)
    else:
        hf_quant_cfg = {"producer": {"name": "modelopt"}}

    quant_section = hf_quant_cfg.get("quantization")
    if not isinstance(quant_section, dict):
        quant_section = {}

    quant_section["quant_algo"] = "NVFP4"
    quant_section["kv_cache_quant_algo"] = DEFAULT_KV_CACHE_QUANT_ALGO
    quant_section["group_size"] = NVFP4_GROUP_SIZE
    quant_section["exclude_modules"] = ignore_list
    hf_quant_cfg["quantization"] = quant_section

    with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_cfg, f, indent=2)


def _augment_ignore_list(ignore_list: list[str]) -> list[str]:
    ignore_set = set(ignore_list)
    extra = set()
    for name in ignore_list:
        if name.endswith(FUSED_QKV_SUFFIXES):
            for suffix in FUSED_QKV_SUFFIXES:
                if name.endswith(suffix):
                    extra.add(name[: -len(suffix)] + ".qkv_proj")
                    break
    ignore_set.update(extra)
    return sorted(ignore_set)


def _split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def _collect_shared_global_amax(
    *,
    input_path: str,
    safetensors_files: list[str],
    device: str,
    skip_weight_substrings: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Collect shared gate/up amax across all shards to keep w1/w3 scales equal."""
    gate_amax: dict[str, torch.Tensor] = {}
    up_amax: dict[str, torch.Tensor] = {}
    for filename in safetensors_files:
        with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if not should_quantize(key, tensor, skip_weight_substrings=skip_weight_substrings):
                    continue
                base, role = _split_gated_pair_name(key)
                if base is None or role is None:
                    continue
                amax = tensor.abs().max().to(torch.float32)
                if role == "gate":
                    prev = gate_amax.get(base)
                    gate_amax[base] = amax if prev is None else torch.max(prev, amax)
                elif role == "up":
                    prev = up_amax.get(base)
                    up_amax[base] = amax if prev is None else torch.max(prev, amax)
                else:
                    continue

    shared_global_amax: dict[str, torch.Tensor] = {}
    for base in gate_amax.keys() & up_amax.keys():
        shared_global_amax[base] = torch.max(gate_amax[base], up_amax[base])
    return shared_global_amax


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
    num_hidden_layers: int,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
    extra_high_precision_layers_hf: tuple[str, ...],
    shared_global_amax: dict[str, torch.Tensor],
) -> None:
    if not filename.endswith(".safetensors"):
        return

    modules_to_not_convert: list[str] = []
    q_weights: dict[str, torch.Tensor] = {}
    head_end_idx = num_layers_at_start_in_bf16
    tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
    dynamic_skip_layer_prefixes: set[str] = set()
    dynamic_skip_layer_prefixes.update({f"model.layers.{i}." for i in range(0, head_end_idx)})
    dynamic_skip_layer_prefixes.update({f"model.layers.{i}." for i in range(tail_start_idx, num_hidden_layers)})

    if num_layers_at_end_in_bf16 > 0 or num_layers_at_start_in_bf16 > 0:
        modules_to_not_convert.extend(sorted(dynamic_skip_layer_prefixes))

    dynamic_skip_substrings = (
        *extra_high_precision_layers_hf,
        *sorted(dynamic_skip_layer_prefixes),
    )

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if should_quantize(key, tensor, skip_weight_substrings=dynamic_skip_substrings):
                base, _role = _split_gated_pair_name(key)
                global_amax = shared_global_amax.get(base) if base else None
                qweight, block_scale, weight_scale_2 = quantize_nvfp4(tensor, global_amax=global_amax)
                q_weights[key] = qweight
                q_weights[key.replace(".weight", ".weight_scale")] = block_scale
                q_weights[key.replace(".weight", ".weight_scale_2")] = weight_scale_2
                q_weights[key.replace(".weight", ".input_scale")] = torch.ones_like(
                    weight_scale_2, dtype=torch.float32
                )
            else:
                if key.endswith(".weight"):
                    modules_to_not_convert.append(key.replace(".weight", ""))
                q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_nvfp4(
    model_dir: str,
    save_dir: str,
    device: str,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
) -> None:
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    num_hidden_layers = _get_num_hidden_layers(input_path)
    head_end_idx = num_layers_at_start_in_bf16
    tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
    dynamic_skip_layer_prefixes: set[str] = set()
    dynamic_skip_layer_prefixes.update({f"model.layers.{i}." for i in range(0, head_end_idx)})
    dynamic_skip_layer_prefixes.update({f"model.layers.{i}." for i in range(tail_start_idx, num_hidden_layers)})
    dynamic_skip_substrings = (
        *extra_high_precision_layers_hf,
        *sorted(dynamic_skip_layer_prefixes),
    )

    shared_global_amax = _collect_shared_global_amax(
        input_path=input_path,
        safetensors_files=safetensors_files,
        device=device,
        skip_weight_substrings=dynamic_skip_substrings,
    )
    result_collector = ConversionResult()
    for filename in tqdm(safetensors_files, desc="Processing files"):
        process_file(
            input_path,
            output_path,
            filename,
            result_collector,
            device,
            num_hidden_layers,
            num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16,
            extra_high_precision_layers_hf,
            shared_global_amax,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ignore_list = _augment_ignore_list(result_collector.modules_to_not_convert)

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        _update_quantization_config(cfg, ignore_list)
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    _write_hf_quant_config(output_path, ignore_list, input_path)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.total_size},
    }
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HF safetensors model.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save converted model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run quantization on (default: cuda).",
    )
    parser.add_argument(
        "--num-layers-at-start-in-bf16",
        type=int,
        default=0,
        help="Keep first N decoder layers in BF16 and do not quantize them.",
    )
    parser.add_argument(
        "--num-layers-at-end-in-bf16",
        type=int,
        default=0,
        help="Keep last N decoder layers in BF16 and do not quantize them.",
    )
    parser.add_argument(
        "--extra-high-precision-layers-hf",
        type=str,
        nargs="*",
        default=[],
        help="Additional HF weight-name substrings to keep unquantized.",
    )
    args, _ = parser.parse_known_args()

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run NVFP4 quantization.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_nvfp4(
        args.model_dir,
        args.save_dir,
        str(device),
        num_layers_at_start_in_bf16=args.num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=args.num_layers_at_end_in_bf16,
        extra_high_precision_layers_hf=tuple(
            s.strip() for s in args.extra_high_precision_layers_hf if isinstance(s, str) and s.strip()
        ),
    )


if __name__ == "__main__":
    main()
