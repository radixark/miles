"""Wraps torchtitan's own ``torchtitan.components.checkpoint.CheckpointManager`` —
exactly what torchtitan's own RL ``PolicyTrainer`` uses (experiments/rl/actors/trainer.py:
``config.checkpoint.build(...)``, ``.load()``, ``.save(step, last_step=)``) — instead of
hand-rolling ``dcp.save``/``dcp.load`` calls with our own tracker-file conventions.

CheckpointManager already unifies both halves of what used to be two separate hand-rolled
paths: the initial HF-safetensors load (``initial_load_in_hf`` + the state-dict adapter's
``hf_assets_path``, model.py used to do this itself via a 3-line dcp recipe) and the
native-DCP resume/save path (this file used to reimplement fsdp_utils' iter_%07d/meta.json
conventions by hand). One call to ``.load()`` picks the right one: HF load if
``checkpoint.folder`` doesn't exist yet, native DCP resume otherwise.
"""

import logging
from typing import Any

from torchtitan.components.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class _TrainState:
    """Rides inside the same CheckpointManager states dict as model/optimizer/
    lr_scheduler — mirrors torchtitan RL's own ``states={"train_state": self}``
    pattern (PolicyTrainer.state_dict/load_state_dict, trainer.py:217-223)."""

    def __init__(self, actor):
        self._actor = actor

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self._actor.global_step,
            "micro_step": self._actor.micro_step,
            "next_rollout_id": getattr(self._actor.args, "start_rollout_id", 0),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._actor.global_step = state_dict["global_step"]
        self._actor.micro_step = state_dict["micro_step"]
        self._actor.args.start_rollout_id = state_dict["next_rollout_id"]


def build(actor) -> CheckpointManager:
    args = actor.args
    base_folder = args.save or f"/tmp/torchtitan_ckpt/{args.wandb_group or 'run'}"
    config = CheckpointManager.Config(
        enable=True,  # must stay True even with no --save: this also gates the initial HF load
        folder="checkpoint",
        interval=1,  # miles' own save_interval (via should_run_periodic_action) is the real gate
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=False,
    )
    return config.build(
        dataloader=None,
        model_parts=[actor.model],
        optimizers=actor.optimizer,
        lr_schedulers=actor.lr_scheduler,
        states={"train_state": _TrainState(actor)},
        sd_adapter=actor.adapter,
        base_folder=base_folder,
    )


def load(actor) -> None:
    loaded = actor.checkpointer.load()
    logger.info(f"[torchtitan] checkpoint load: {'resumed from native DCP' if loaded else 'no checkpoint found'}")


def save(actor, rollout_id: int) -> None:
    if actor.args.save is None:
        return
    last_step = rollout_id + 1 == actor.args.num_rollout
    actor.checkpointer.save(rollout_id + 1, last_step=last_step)
