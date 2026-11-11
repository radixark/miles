import logging
from dataclasses import dataclass

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.components.checkpointer import MODEL, OPTIMIZER, CheckpointManager

from miles.backends.torchtitan_utils import lora_names

logger = logging.getLogger(__name__)


class TiedCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def _load_checkpoint(self, states, checkpoint_id, *, from_hf, from_quantized):
        if not from_hf:
            states = without_unsaved_optimizer_state(states, checkpoint_id)
            return super()._load_checkpoint(states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized)

        assert self.sd_adapter is not None
        state_dict = self._flattened_model_states_sd(states)
        # LoRA adapters have no HF checkpoint tensors; they keep their init
        hf_state = self.sd_adapter.to_hf(
            {k: v for k, v in state_dict.items() if not lora_names.is_adapter_key(k)},
            to_quantized_layout=from_quantized,
        )
        if self.sd_adapter.fqn_to_index_mapping:
            available = set(self.sd_adapter.fqn_to_index_mapping)
            dropped = sorted(k for k in hf_state if k not in available)
            if dropped:
                logger.info(
                    f"HF checkpoint lacks {len(dropped)} exported key(s) (e.g. {dropped[:3]}); "
                    "deferring to the adapter's from_hf reconstruction"
                )
                hf_state = {k: v for k, v in hf_state.items() if k in available}

        dcp.load(hf_state, storage_reader=self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized))
        states[MODEL].load_state_dict(self.sd_adapter.from_hf(hf_state, is_quantized_layout=from_quantized))


UNSAVED_STATE_NAMES_SHOWN = 8


class _OptimizerStateWithoutUnsaved(Stateful):
    """The optimizer's flat state dict minus the entries the checkpoint does not
    hold, so DCP asks the checkpoint only for what is there. The parameters
    behind the omitted entries end up without optimizer state, as after a
    fresh start: Adam creates it on their first gradient."""

    def __init__(self, optimizer: Stateful, unsaved: set[str]):
        self.optimizer = optimizer
        self.unsaved = unsaved

    def state_dict(self) -> dict:
        return {key: value for key, value in self.optimizer.state_dict().items() if key not in self.unsaved}

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)


def unsaved_optimizer_state(optimizer: Stateful, checkpoint_id: str) -> list[str]:
    """Flat optimizer-state keys (``state.<fqn>.<name>``) the optimizer expects
    but the checkpoint at ``checkpoint_id`` lacks. A parameter that never
    received a gradient has no optimizer state when the run is saved, while a
    fresh optimizer at load time expects state for every parameter; the
    difference is what this returns. Only ``state.`` entries count: a missing
    ``param_groups.`` entry is still a broken checkpoint."""
    saved = dcp.FileSystemReader(checkpoint_id).read_metadata().state_dict_metadata
    return sorted(
        key for key in optimizer.state_dict() if key.startswith("state.") and f"{OPTIMIZER}.{key}" not in saved
    )


def without_unsaved_optimizer_state(states: dict, checkpoint_id: str) -> dict:
    """``states`` for ``CheckpointManager._load_checkpoint`` with the optimizer
    narrowed to the state the checkpoint holds; the model and everything else
    stay strict. One log line names the parameters whose state starts fresh.

    torchtitan 0.4.0 saves state for every trainable parameter (its
    ``init_optim_state`` fills in the ones without state before each save), so
    its own saves never trip this. Saves written under torchtitan 0.3.0 do:
    its ``init_optim_state`` returned early once any state existed, so a
    parameter that never got a gradient was left out of the save."""
    optimizer = states.get(OPTIMIZER)
    if optimizer is None:
        return states
    unsaved = unsaved_optimizer_state(optimizer, checkpoint_id)
    if not unsaved:
        return states
    parameters = sorted({key.removeprefix("state.").rsplit(".", 1)[0] for key in unsaved})
    shown = ", ".join(parameters[:UNSAVED_STATE_NAMES_SHOWN])
    more = (
        f" and {len(parameters) - UNSAVED_STATE_NAMES_SHOWN} more"
        if len(parameters) > UNSAVED_STATE_NAMES_SHOWN
        else ""
    )
    logger.info(
        f"resume from {checkpoint_id}: {len(unsaved)} optimizer state entries of {len(parameters)} parameters are "
        f"not in the checkpoint (parameters that never received a gradient) and start fresh: {shown}{more}"
    )
    narrowed = dict(states)
    narrowed[OPTIMIZER] = _OptimizerStateWithoutUnsaved(optimizer, set(unsaved))
    return narrowed
