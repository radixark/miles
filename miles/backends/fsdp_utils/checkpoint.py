"""Distributed-checkpoint save and resume for the FSDP backend.

Sharded state goes out through ``torch.distributed.checkpoint``, which speaks
DTensor directly. An actor names the pieces to persist by defining
``checkpoint_parts()``; otherwise the default below wraps its optimizer and LR
scheduler. On disk: one directory per part under ``iter_NNNNNNN``, alongside the
RNG state and a metadata file, with ``latest_checkpointed_iteration.txt`` at the
root naming the newest iteration.
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
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None)


class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _resolve_parts(actor: Any) -> dict[str, Stateful]:
    """The stateful pieces to write, keyed by the role each occupies on disk.

    Backends whose optimizer and LR scheduler are already ``Stateful`` -- as
    torchtitan's containers are -- override this and hand them over directly.
    The default wraps the plain torch objects an actor holds one of each of.
    """
    getter = getattr(actor, "checkpoint_parts", None)
    if getter is not None:
        return getter()

    parts: dict[str, Stateful] = {"model": ModelState(actor.model)}
    optimizer = getattr(actor, "optimizer", None)
    if optimizer is not None:
        parts["optimizer"] = OptimizerState(actor.model, optimizer)
    lr_scheduler = getattr(actor, "lr_scheduler", None)
    if lr_scheduler is not None:
        parts["lr_scheduler"] = LRSchedulerState(lr_scheduler)
    return parts


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
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"Model checkpoint {model_dir} not found; skipping load.")
        return None

    parts = _resolve_parts(actor)

    # Load model weights (always)
    state_dict = {"model_state": parts["model"]}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}")
        return None

    # Load optimizer state (optional)
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and parts.get("optimizer") is not None
    if load_optimizer and optimizer_dir.exists():
        optim_state_dict = {"optim_state": parts["optimizer"]}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(f"Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    has_lr_scheduler = parts.get("lr_scheduler") is not None
    load_lr_scheduler = has_lr_scheduler and lr_scheduler_dir.exists()
    if load_lr_scheduler:
        lr_scheduler_state_dict = {"lr_scheduler_state": parts["lr_scheduler"]}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            logger.warning(f"Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif has_lr_scheduler:
        logger.info(f"LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load.")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
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
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    parts = _resolve_parts(actor)

    dcp.save({"model_state": parts["model"]}, checkpoint_id=str(model_dir))

    # Optimizer and LR scheduler both ride on --no-save-optim.
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and parts.get("optimizer") is not None:
        dcp.save({"optim_state": parts["optimizer"]}, checkpoint_id=str(optimizer_dir))

    if save_optimizer_state and parts.get("lr_scheduler") is not None:
        dcp.save({"lr_scheduler_state": parts["lr_scheduler"]}, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
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

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
