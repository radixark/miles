import logging
from dataclasses import dataclass

import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpointer import MODEL, CheckpointManager

from miles.backends.torchtitan_utils import lora_names

logger = logging.getLogger(__name__)


class TiedCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def _load_checkpoint(self, states, checkpoint_id, *, from_hf, from_quantized):
        if not from_hf:
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
