"""Resumable FSDP2 checkpoints and release-compatible backbone/head exports."""

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from examples.experimental.clef.model import TrainableClefModel
from examples.experimental.clef.state import ModelState, OptimizerState
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


def save_checkpoint(
    model: TrainableClefModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    metadata: dict[str, Any],
    processor: Any,
    head_config: dict[str, int],
    checkpoint_root: str | Path | None = None,
) -> None:
    checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else output_dir / "checkpoints"
    root = checkpoint_root / f"step_{step:07d}"
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    dcp.save(
        {"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, checkpoint_id=root / "native"
    )
    # Gather onto rank-zero CPU; no rank gathers a full FP32 model onto its GPU.
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        export = root / "hf"
        _export_model(model, state, export, processor, head_config)
        (root / "metadata.json").write_text(json.dumps({"step": step, **metadata}, indent=2))
    del state
    dist.barrier()
    if dist.get_rank() == 0:
        (root / "COMPLETE.json").write_text(json.dumps({"step": step, "world_size": dist.get_world_size()}))
        tracker = checkpoint_root / "latest.json"
        temporary = tracker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"step": step, "path": str(root)}))
        temporary.replace(tracker)
    dist.barrier()


def load_checkpoint(model: TrainableClefModel, optimizer: torch.optim.Optimizer, path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not (path / "COMPLETE.json").is_file():
        raise ValueError(f"checkpoint is incomplete: {path}")
    dcp.load(
        {"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, checkpoint_id=path / "native"
    )
    return json.loads((path / "metadata.json").read_text())


def _export_model(
    model: TrainableClefModel,
    state: dict[str, torch.Tensor],
    export: Path,
    processor: Any,
    head_config: dict[str, int],
) -> None:
    backbone = {
        key.removeprefix("language_model."): value.to(torch.bfloat16)
        for key, value in state.items()
        if key.startswith("language_model.")
    }
    model.language_model.save_pretrained(export, state_dict=backbone, max_shard_size="5GB")
    processor.save_pretrained(export)
    head = {
        key.removeprefix("head."): value.to(torch.bfloat16).contiguous()
        for key, value in state.items()
        if key.startswith("head.")
    }
    for name in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
        head[name] = head[name].reshape(())
    save_file(head, export / "joint_head.safetensors")
    (export / "joint_head_config.json").write_text(json.dumps(head_config, indent=2))
    shutil.copyfile(Path(__file__).with_name("joint_schema_model.py"), export / "joint_schema_model.py")
