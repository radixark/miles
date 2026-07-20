"""miles checkpoint conventions (iter_%07d/{model,optimizer,lr_scheduler} + rng.pt +
meta.json + latest_checkpointed_iteration.txt), adapted from fsdp_utils/checkpoint.py.

torchtitan's OptimizersContainer and LRSchedulersContainer already implement
``torch.distributed.checkpoint.stateful.Stateful`` with flat-FQN, reshardable state, so
they (and the titan model itself, whose ``state_dict()``/``load_state_dict()`` already
speak DTensor) plug directly into ``dcp.save``/``dcp.load`` without the get_state_dict/
set_state_dict wrapper the FSDP2 backend needs. This file is a twin of
fsdp_utils/checkpoint.py: bug fixes to the metadata/tracker conventions should mirror
into both until the two backends converge into a shared torch_native_utils module.

NOTE: this handles miles' resume/CI checkpoints only. The initial HF-checkpoint weight
load is separate (model.py's build_and_load_model), not routed through here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk (model always; optimizer/lr_scheduler if present)."""
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"[torchtitan] Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"[torchtitan] No tracker file at {tracker_file}; skipping load.")
            return None
        target_step = int(tracker_file.read_text().strip())

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"[torchtitan] Model checkpoint {model_dir} not found; skipping load.")
        return None

    try:
        dcp.load(state_dict={"model": actor.model.state_dict()}, checkpoint_id=str(model_dir))
        logger.info(f"[torchtitan] Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"[torchtitan] Failed to load model from {model_dir}: {e}")
        return None

    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        try:
            dcp.load(state_dict={"optim": actor.optimizer}, checkpoint_id=str(optimizer_dir))
            logger.info(f"[torchtitan] Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(f"[torchtitan] Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"[torchtitan] Optimizer checkpoint not found at {optimizer_dir}, skipping.")

    load_lr = hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    if load_lr:
        try:
            dcp.load(state_dict={"lr_scheduler": actor.lr_scheduler}, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"[torchtitan] Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            logger.warning(f"[torchtitan] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(f"[torchtitan] LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping.")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    return {
        "rng": rng_state,
        "metadata": _read_checkpoint_metadata(checkpoint_dir / "meta.json"),
        "iteration": target_step,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            actor.args.start_rollout_id = next_rollout
    elif iteration is not None:
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, iteration: int) -> None:
    """Save checkpoint (model always; optimizer/lr_scheduler unless --no-save-optim)."""
    torch.cuda.synchronize()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        for d in (checkpoint_dir, model_dir, optimizer_dir, lr_scheduler_dir):
            d.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dcp.save({"model": actor.model.state_dict()}, checkpoint_id=str(model_dir))

    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and getattr(actor, "optimizer", None) is not None:
        dcp.save({"optim": actor.optimizer}, checkpoint_id=str(optimizer_dir))
    if save_optimizer_state and getattr(actor, "lr_scheduler", None) is not None:
        dcp.save({"lr_scheduler": actor.lr_scheduler}, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        (base_dir / "latest_checkpointed_iteration.txt").write_text(str(step_id))
        logger.info(f"[torchtitan] Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
