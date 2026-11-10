"""Prepare a residual HF base and an ordinary PEFT adapter for Miles single-LoRA training.

The decomposition follows Neil Kale's SkyRL PiSSA producer (SkyRL PR #2069):
W = W_residual + (alpha / rank) * B @ A. Both factors start nonzero.
Unlike a Megatron checkpoint, these artifacts load through Miles' existing
--hf-checkpoint and --lora-adapter-path options. Convert residual_base with the
normal convert_hf_to_torch_dist tool before Megatron training. Keep a KL reference
on the original model, not the residual base. This tool does not create optimizer
state or resume a training run.

Example (local, unquantized dense Qwen3 checkpoint):
    python -m tools.prepare_pissa /models/Qwen3-0.6B /models/Qwen3-0.6B-pissa-r32 --rank 32

For Qwen3.5, the default attn/mlp groups include GDN in_proj_qkv and in_proj_z.
Use --lora-type lora in Miles to keep the HF-fused projections' shared A factors.
MoE, quantized checkpoints, and output-head adaptation are deliberately unsupported.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import matches_lora_target


@torch.no_grad()
def pissa_decompose(weight: torch.Tensor, rank: int, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stored A, B and residual, accounting for LoRA scaling and storage precision."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("PiSSA requires an unquantized floating-point matrix")
    if not 0 < rank <= min(weight.shape) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("PiSSA requires 0 < rank <= min(weight.shape) and finite alpha > 0")
    scale = alpha / rank
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    root = (s[:rank] / scale).sqrt()
    a = (root[:, None] * vh[:rank]).to(weight.dtype).contiguous()
    b = (u[:, :rank] * root[None, :]).to(weight.dtype).contiguous()
    # Subtract the factors actually stored, so factor rounding is not counted twice.
    residual = (weight.float() - scale * (b.float() @ a.float())).to(weight.dtype).contiguous()
    return a, b, residual


def _weight_files(src: Path) -> list[Path]:
    index_path = src / "model.safetensors.index.json"
    names = (
        sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        if index_path.is_file()
        else ["model.safetensors"]
    )
    if not names or any(Path(name).name != name or not (src / name).is_file() for name in names):
        raise ValueError("Expected local safetensors shards named by the HF index, or model.safetensors")
    return [src / name for name in names]


def prepare(src: Path, dst: Path, *, rank: int, alpha: float | None = None, device: str = "cpu") -> tuple[Path, Path]:
    """Convert a local dense Qwen checkpoint without changing the source or overwriting output."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst == src or src in dst.parents or dst.exists():
        raise ValueError("Use a new output directory outside the source checkpoint")
    config = json.loads((src / "config.json").read_text())
    text_config = config.get("text_config") or config
    if (
        text_config["model_type"] not in ("qwen3", "qwen3_5", "qwen3_5_text")
        or text_config.get("quantization_config")
        or config.get("quantization_config")
    ):
        raise ValueError("PiSSA preparation supports unquantized dense Qwen3 and Qwen3.5 only")
    alpha = float(rank) if alpha is None else alpha
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("rank and alpha must be positive and finite")
    targets = resolve_hf_lora_targets(config, target_modules=["attn", "mlp"])
    files = _weight_files(src)
    # Validate every target before creating output. Never silently omit a projection.
    selected = []
    for path in files:
        with safe_open(path, framework="pt") as reader:
            for name in reader.keys():
                if name.endswith(".weight") and any(matches_lora_target(name[:-7], target) for target in targets):
                    shape = reader.get_slice(name).get_shape()
                    if len(shape) != 2 or rank > min(shape):
                        raise ValueError(f"Invalid rank {rank} for {name}: {shape}")
                    selected.append(name[:-7])
    missing = [target for target in targets if not any(matches_lora_target(name, target) for name in selected)]
    if missing:
        raise ValueError(f"Checkpoint is missing LoRA targets: {missing}")
    residual_dir, adapter_dir = dst / "residual_base", dst / "adapter"
    residual_dir.mkdir(parents=True)
    adapter_dir.mkdir()
    # Copy HF assets only, not caches or an unrelated training checkpoint.
    for path in src.iterdir():
        if path.is_file() and path.suffix in (".json", ".jinja", ".model", ".txt", ".tiktoken"):
            shutil.copy2(path, residual_dir / path.name)
    adapters = {}
    for path in files:
        tensors = {}
        with safe_open(path, framework="pt") as reader:
            metadata = reader.metadata()
            for name in reader.keys():
                weight = reader.get_tensor(name)
                module = name.removesuffix(".weight")
                if module in selected:
                    print(f"PiSSA: {name} {tuple(weight.shape)}", flush=True)
                    a, b, residual = pissa_decompose(weight.to(device), rank, alpha)
                    adapters[f"base_model.model.{module}.lora_A.weight"] = a.cpu()
                    adapters[f"base_model.model.{module}.lora_B.weight"] = b.cpu()
                    weight = residual.cpu()
                tensors[name] = weight
        save_file(tensors, residual_dir / path.name, metadata=metadata)
    save_file(adapters, adapter_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    # These are ordinary LoRA tensors on an already materialized residual base.
    # Do not mark them init_lora_weights='pissa': no consumer should rewrite the base again.
    adapter_config = dict(
        peft_type="LORA",
        task_type="CAUSAL_LM",
        base_model_name_or_path=str(residual_dir),
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
        target_modules=sorted(selected),
        inference_mode=True,
    )
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    return residual_dir, adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, help="defaults to rank, as in Neil's SkyRL example")
    parser.add_argument("--device", default="cpu", help="device used for each SVD, e.g. cpu or cuda:0")
    args = parser.parse_args()
    residual, adapter = prepare(args.src, args.dst, rank=args.rank, alpha=args.alpha, device=args.device)
    print(f"Residual base: {residual}\nAdapter: {adapter}")


if __name__ == "__main__":
    main()
