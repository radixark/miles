"""
Convert an official XiaomiMiMo MiMo-V2 checkpoint (`MiMoV2ForCausalLM`, e.g. MiMo-V2.6-Flash-RL)
to plain BF16 with split q/k/v projections, optionally keeping a subset of decoder layers and
routed experts for partial-model work.

The source stores `self_attn.qkv_proj` fused and interleaved across `num_key_value_heads`
shards (shard i is `[q_i | k_i | v_i]` for its heads, the layout SGLang's TP loader expects) in
blockwise FP8 quantized shard by shard (each shard padded to whole 128-row blocks, so global-attention
layers carry 4 x 27 scale rows for 13568 weight rows), the dense and MTP linears in blockwise FP8, and the routed experts in MXFP4
(packed e2m1 in uint8 plus one E8M0 scale per 32 inputs). The output dequantizes all of them to
BF16, splits qkv into q_proj/k_proj/v_proj in head order, and drops `quantization_config` and
`attention_projection_layout`, so the HF remote code, Megatron-Bridge and a BF16 SGLang engine
all read it as-is.

--keep-quant only reindexes layers and copies every tensor in its source format.

python tools/convert_mimo_v2_to_bf16.py --model-dir <src> --save-dir <dst> [--layers 0,1,5,6] [--num-experts 32]
"""

import json
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP8_BLOCK = 128
MXFP4_BLOCK = 32
SHARD_BYTES = 5 * 1024**3

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.")
_ROUTER_KEYS = ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")


def dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    scale = scale_inv.float().repeat_interleave(FP8_BLOCK, dim=0)[:rows].repeat_interleave(FP8_BLOCK, dim=1)
    return (weight.float() * scale[:, :cols]).to(torch.bfloat16)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = packed.shape[0], packed.shape[1] * 2
    assert scale.shape == (out_dim, in_dim // MXFP4_BLOCK), f"{packed.shape=} {scale.shape=}"
    packed = packed.view(torch.uint8)
    table = FP4_TABLE.to(packed.device)
    values = torch.stack([table[(packed & 0x0F).long()], table[(packed >> 4).long()]], dim=-1).reshape(out_dim, in_dim)
    return (values * torch.exp2(scale.float() - 127.0).repeat_interleave(MXFP4_BLOCK, dim=1)).to(torch.bfloat16)


def dequant_fused_qkv(weight: torch.Tensor, scale_inv: torch.Tensor, shards: int) -> torch.Tensor:
    """Each kv-head shard of a fused qkv_proj is block-quantized on its own, padded to whole blocks."""
    rows = weight.shape[0] // shards
    scale_rows = -(-rows // FP8_BLOCK)
    assert scale_inv.shape[0] == shards * scale_rows, f"{weight.shape=} {scale_inv.shape=} {shards=}"
    return torch.cat(
        [dequant_fp8_block(w, s) for w, s in zip(weight.chunk(shards), scale_inv.split(scale_rows), strict=True)]
    )


def split_fused_qkv(qkv: torch.Tensor, config: dict, is_swa: bool) -> list[torch.Tensor]:
    """Undo the kv-head interleave of a fused qkv_proj and return [q, k, v] in head order."""
    shards = config["num_key_value_heads"]
    prefix = "swa_" if is_swa else ""
    num_heads, kv_heads = config[f"{prefix}num_attention_heads"], config[f"{prefix}num_key_value_heads"]
    assert num_heads % shards == 0 and kv_heads % shards == 0, f"{num_heads=} {kv_heads=} {shards=}"
    sizes = [
        num_heads // shards * config[f"{prefix}head_dim"],
        kv_heads // shards * config[f"{prefix}head_dim"],
        kv_heads // shards * config[f"{prefix}v_head_dim"],
    ]
    assert qkv.shape[0] == shards * sum(sizes), f"{qkv.shape=} does not match {shards=} x {sizes=}"
    return [part.reshape(-1, qkv.shape[1]) for part in qkv.view(shards, sum(sizes), -1).split(sizes, dim=1)]


def output_name(name: str, layer_map: dict[int, int] | None, num_experts: int | None) -> str | None:
    """Output name of a source tensor, or None when the tensor is not kept."""
    match = _LAYER_RE.match(name)
    if match is None:
        return name
    layer, rest = int(match.group(1)), match.group(2)
    if layer_map is not None and layer not in layer_map:
        return None
    expert = _EXPERT_RE.match(rest)
    if expert is not None and num_experts is not None and int(expert.group(1)) >= num_experts:
        return None
    return f"model.layers.{layer_map[layer] if layer_map is not None else layer}.{rest}"


def convert_config(config: dict, layer_map: dict[int, int] | None, num_experts: int | None, keep_quant: bool) -> dict:
    config = json.loads(json.dumps(config))
    if layer_map is not None:
        assert config.get("hybrid_block_size") is None, "per-layer schedules must be explicit lists"
        kept = list(layer_map)
        config["num_hidden_layers"] = len(kept)
        for key in ("hybrid_layer_pattern", "moe_layer_freq"):
            config[key] = [config[key][layer] for layer in kept]
    if num_experts is not None:
        assert config["num_experts_per_tok"] <= num_experts <= config["n_routed_experts"], num_experts
        config["n_routed_experts"] = num_experts
    if keep_quant:
        quant = config["quantization_config"]
        ignored = []
        for module in quant.get("ignored_layers", []):
            renamed = output_name(module + ".weight", layer_map, num_experts)
            if renamed is not None:
                ignored.append(renamed.removesuffix(".weight"))
        quant["ignored_layers"] = ignored
    else:
        config.pop("quantization_config", None)
        config.pop("attention_projection_layout", None)
    return config


class ShardWriter:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.num_shards = 0
        self.total_bytes = 0
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
        self.total_bytes += self.buffer_bytes
        self.buffer, self.buffer_bytes = {}, 0


def check_output(weight_map: dict[str, str], config: dict) -> None:
    """Every kept decoder layer carries the tensors its attention/MLP variant needs."""
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        names = {name[len(prefix) :] for name in weight_map if name.startswith(prefix)}
        required = {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.o_proj.weight"}
        required |= (
            {f"self_attn.{p}_proj.weight" for p in "qkv"} if "self_attn.qkv_proj.weight" not in names else set()
        )
        if config["hybrid_layer_pattern"][layer] == 1 and config.get("add_swa_attention_sink_bias"):
            required.add("self_attn.attention_sink_bias")
        if config["moe_layer_freq"][layer]:
            required |= set(_ROUTER_KEYS)
            experts = {int(m.group(1)) for n in names if (m := _EXPERT_RE.match(n))}
            assert experts == set(
                range(config["n_routed_experts"])
            ), f"layer {layer}: experts {sorted(experts)[:4]}..."
        else:
            required |= {f"mlp.{p}_proj.weight" for p in ("gate", "up", "down")}
        missing = required - names
        assert not missing, f"layer {layer} is missing {sorted(missing)}"


def main(
    model_dir: str, save_dir: str, layers: list[int] | None, num_experts: int | None, keep_quant: bool, device: str
):
    src, dst = Path(model_dir), Path(save_dir)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    assert config["architectures"] == ["MiMoV2ForCausalLM"], config["architectures"]
    layer_map = {old: new for new, old in enumerate(layers)} if layers is not None else None
    assert layer_map is None or all(0 <= layer < config["num_hidden_layers"] for layer in layer_map), layers
    assert not (keep_quant and num_experts is not None), "--num-experts needs the dequantized output"
    fused_qkv = config.get("attention_projection_layout") == "fused_qkv"

    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def load(name: str) -> torch.Tensor:
        file_name = weight_map[name]
        if file_name not in handles:
            handles[file_name] = safe_open(str(src / file_name), framework="pt", device="cpu")
        return handles[file_name].get_tensor(name).to(device)

    writer = ShardWriter(dst)
    counts = {"fp8": 0, "mxfp4": 0, "qkv_split": 0, "copied": 0}
    scale_suffixes = ("weight_scale_inv", "weight_scale")
    for file_name in tqdm(sorted(set(weight_map.values())), desc="shards"):
        for name in [n for n, f in weight_map.items() if f == file_name]:
            new_name = output_name(name, layer_map, num_experts)
            if new_name is None or (not keep_quant and name.endswith(scale_suffixes)):
                continue
            tensor = load(name)
            if keep_quant:
                writer.add(new_name, tensor)
                counts["copied"] += 1
                continue
            if fused_qkv and new_name.endswith("self_attn.qkv_proj.weight"):
                match = _LAYER_RE.match(name)
                is_swa = match is None or config["hybrid_layer_pattern"][int(match.group(1))] == 1
                if tensor.dtype == torch.float8_e4m3fn:
                    tensor = dequant_fused_qkv(tensor, load(name + "_scale_inv"), config["num_key_value_heads"])
                    counts["fp8"] += 1
                for proj, part in zip("qkv", split_fused_qkv(tensor, config, is_swa), strict=True):
                    writer.add(new_name.replace("qkv_proj", f"{proj}_proj"), part)
                counts["qkv_split"] += 1
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                tensor = dequant_fp8_block(tensor, load(name + "_scale_inv"))
                counts["fp8"] += 1
            elif tensor.dtype == torch.uint8:
                tensor = dequant_mxfp4(tensor, load(name.removesuffix("weight") + "weight_scale"))
                counts["mxfp4"] += 1
            if num_experts is not None and new_name.endswith(_ROUTER_KEYS):
                tensor = tensor[:num_experts]
            writer.add(new_name, tensor)
            counts["copied"] += 1
    writer.flush()

    new_config = convert_config(config, layer_map, num_experts, keep_quant)
    if not keep_quant:
        check_output(writer.weight_map, new_config)
    (dst / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")
    index = {"metadata": {"total_size": writer.total_bytes}, "weight_map": writer.weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for path in src.iterdir():
        if path.name.startswith("."):  # e.g. the `hf download` cache
            continue
        if path.is_dir():
            # The DFlash drafter reads hidden states of fixed full-model layers, so it only
            # stays valid when every layer is kept.
            if path.name != "dflash" or layer_map is None:
                shutil.copytree(path, dst / path.name, dirs_exist_ok=True)
        elif path.suffix != ".safetensors" and path.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(path, dst / path.name)
    print(
        f"{counts}; {len(writer.weight_map)} tensors, {writer.total_bytes / 1e9:.2f} GB in {writer.num_shards} shards -> {dst}"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument(
        "--layers", type=str, default=None, help="Comma-separated source layer indices to keep, in order."
    )
    parser.add_argument(
        "--num-experts", type=int, default=None, help="Keep only the first N routed experts per MoE layer."
    )
    parser.add_argument("--keep-quant", action="store_true", help="Reindex layers but keep the source tensor formats.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    main(args.model_dir, args.save_dir, layers, args.num_experts, args.keep_quant, args.device)
