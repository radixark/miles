"""
Convert a MiniMax-M3 / MiniMax-M3.1 checkpoint (`MiniMaxM3SparseForConditionalGeneration`) to plain
BF16, optionally keeping a subset of decoder layers for partial-model work. A BF16 source (MiniMax-M3)
is only cut to the kept layers.

MiniMax-M3.1-Flash stores attention, dense and shared-expert linears in MXFP8 (e4m3 with one ue8m0 scale per
32 inputs) and the routed experts in compressed-tensors NVFP4 (packed e2m1, an e4m3 scale per 16 inputs
and a per-tensor encode scale; w = q * s / g). The output dequantizes both to BF16 and drops
`quantization_config`, so the HF remote code, the Megatron bridge and a BF16 SGLang engine read it as-is.
The vision tower and the `dspark/` draft model are not touched.

python tools/convert_minimax_m3_to_bf16.py --model-dir <src> --save-dir <dst> [--layers 0,1,3,4]
"""

import copy
import json
import re
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
MXFP8_BLOCK = 32
NVFP4_BLOCK = 16
SHARD_BYTES = 5 * 1024**3
SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale", "weight_global_scale", "input_global_scale")
PER_LAYER_SPARSE_KEYS = ("sparse_attention_freq", "sparse_disable_index_value")

_LAYER_RE = re.compile(r"^(language_model\.model\.layers\.)(\d+)(\..+)$")


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    assert scale.shape == (rows, cols // MXFP8_BLOCK), f"{weight.shape=} {scale.shape=}"
    blocks = weight.float().view(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK)
    return (blocks * torch.exp2(scale.float() - 127.0).unsqueeze(-1)).view(rows, cols).to(torch.bfloat16)


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Low nibble is the even element."""
    rows, cols = packed.shape[0], packed.shape[1] * 2
    assert scale.shape == (rows, cols // NVFP4_BLOCK), f"{packed.shape=} {scale.shape=}"
    table = E2M1_TABLE.to(packed.device)
    values = torch.stack([table[(packed & 0x0F).long()], table[(packed >> 4).long()]], dim=-1).view(rows, cols)
    blocks = values.view(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK) * scale.float().unsqueeze(-1)
    return (blocks / global_scale.float().reshape(())).view(rows, cols).to(torch.bfloat16)


def output_name(name: str, layer_map: dict[int, int] | None) -> str | None:
    """Output name of a source tensor, or None when its layer is not kept."""
    match = _LAYER_RE.match(name)
    if match is None or layer_map is None:
        return name
    layer = int(match.group(2))
    if layer not in layer_map:
        return None
    return f"{match.group(1)}{layer_map[layer]}{match.group(3)}"


def convert_config(config: dict, layers: list[int] | None) -> dict:
    config = copy.deepcopy(config)
    config.pop("quantization_config", None)
    if layers is not None:
        text = config["text_config"]
        text["num_hidden_layers"] = len(layers)
        text["moe_layer_freq"] = [text["moe_layer_freq"][layer] for layer in layers]
        sparse = text["sparse_attention_config"]
        for key in PER_LAYER_SPARSE_KEYS:
            sparse[key] = [sparse[key][layer] for layer in layers]
    return config


class ShardWriter:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.num_shards = 0
        self.weight_map: dict[str, str] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        assert name not in self.weight_map and name not in self.buffer, f"duplicate tensor {name}"
        self.buffer[name] = tensor.contiguous().cpu()
        self.buffer_bytes += tensor.numel() * tensor.element_size()
        if self.buffer_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.num_shards += 1
        file_name = f"model-{self.num_shards:05d}.safetensors"
        save_file(self.buffer, str(self.save_dir / file_name), metadata={"format": "pt"})
        self.weight_map.update(dict.fromkeys(self.buffer, file_name))
        self.buffer, self.buffer_bytes = {}, 0


def main(model_dir: str, save_dir: str, layers: list[int] | None, device: str):
    src, dst = Path(model_dir), Path(save_dir)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    assert config["architectures"] == ["MiniMaxM3SparseForConditionalGeneration"], config["architectures"]
    num_layers = config["text_config"]["num_hidden_layers"]
    assert layers is None or all(0 <= layer < num_layers for layer in layers), layers
    layer_map = {old: new for new, old in enumerate(layers)} if layers is not None else None

    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    names_by_file = defaultdict(list)
    for name, file_name in weight_map.items():
        names_by_file[file_name].append(name)
    handles = {}

    def load(name: str) -> torch.Tensor:
        file_name = weight_map[name]
        if file_name not in handles:
            handles[file_name] = safe_open(str(src / file_name), framework="pt", device="cpu")
        return handles[file_name].get_tensor(name).to(device)

    writer = ShardWriter(dst)
    counts = {"mxfp8": 0, "nvfp4": 0, "copied": 0}
    for file_name in tqdm(sorted(names_by_file), desc="shards"):
        for name in names_by_file[file_name]:
            if name.endswith(SCALE_SUFFIXES):
                continue
            if name.endswith(".weight_packed"):
                base = name.removesuffix(".weight_packed")
                new_name = output_name(f"{base}.weight", layer_map)
                if new_name is None:
                    continue
                tensor = dequant_nvfp4(load(name), load(f"{base}.weight_scale"), load(f"{base}.weight_global_scale"))
                counts["nvfp4"] += 1
            else:
                new_name = output_name(name, layer_map)
                if new_name is None:
                    continue
                scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
                if name.endswith(".weight") and scale_name in weight_map:
                    tensor = dequant_mxfp8(load(name), load(scale_name))
                    counts["mxfp8"] += 1
                else:
                    tensor = load(name)
                    assert tensor.dtype not in (torch.float8_e4m3fn, torch.uint8), f"unhandled {name} {tensor.dtype}"
                    counts["copied"] += 1
            writer.add(new_name, tensor)
    writer.flush()

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": writer.weight_map}, indent=2)
    )
    (dst / "config.json").write_text(json.dumps(convert_config(config, layers), indent=2, ensure_ascii=False))
    for path in src.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy(path, dst / path.name)
    print(f"{counts} -> {dst} ({writer.num_shards} shards)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.model_dir, args.save_dir, args.layers, args.device)
