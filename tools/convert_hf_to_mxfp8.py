"""
python tools/convert_hf_to_mxfp8.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR]

Convert a BF16/FP16 HF safetensors checkpoint to MXFP8 with UE8M0 scales.
The scale layout mirrors sglang _quantize_and_swizzle_with_triton_kernel,
but keeps the scales in unswizzled group layout for serialization.
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

try:
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
except ImportError as exc:
    raise ImportError(
        "Missing sglang dependency: mxfp8_group_quantize must be importable from sglang.srt.layers.quantization.fp8_utils."
    ) from exc


SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
)


def should_quantize(name: str, weight: torch.Tensor) -> bool:
    if not name.endswith(".weight"):
        return False
    if any(substr in name for substr in SKIP_WEIGHT_SUBSTRINGS):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % 32 != 0:
        return False
    return True


def quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror sglang _quantize_and_swizzle_with_triton_kernel but do not swizzle scales.
    Returns:
      qweight: same shape as input, dtype float8_e4m3fn
      scale:  shape = (*weight.shape[:-1], weight.shape[-1] // 32), dtype uint8
    """
    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % 32 != 0:
        raise ValueError(f"Last dim {k} must be divisible by 32 for MXFP8.")

    weight_flat = weight.view(-1, k).contiguous()
    qweight, scale = mxfp8_group_quantize(weight_flat)
    qweight = qweight.view_as(weight)
    scale = scale.view(*weight.shape[:-1], k // 32).contiguous()
    return qweight, scale


class ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(
        self,
        filename: str,
        q_weights: dict[str, torch.Tensor],
        module_names: list[str],
    ) -> None:
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
) -> None:
    if not filename.endswith(".safetensors"):
        return

    weights: dict[str, torch.Tensor] = {}
    q_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    modules_to_not_convert: list[str] = []
    for key, tensor in weights.items():
        if should_quantize(key, tensor):
            qweight, scale = quantize_mxfp8(tensor)
            q_weights[key] = qweight
            q_weights[key.replace(".weight", ".weight_scale_inv")] = scale
        else:
            if key.endswith(".weight"):
                modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_mxfp8(model_dir: str, save_dir: str, device: str) -> None:
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()
    for filename in tqdm(safetensors_files, desc="Processing files"):
        process_file(input_path, output_path, filename, result_collector, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "mxfp8",
        "weight_block_size": [1, 32],
        "scale_fmt": "ue8m0",
    }
    if len(result_collector.modules_to_not_convert) > 0:
        quantization_config["modules_to_not_convert"] = list(set(result_collector.modules_to_not_convert))

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run MXFP8 quantization.")

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("MXFP8 quantization requires a CUDA device.")
    if device.index is None:
        device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_mxfp8(args.model_dir, args.save_dir, str(device))


if __name__ == "__main__":
    main()
