"""The two torchtitan components miles swaps into the trainer's config tree."""

import logging
from dataclasses import dataclass

import torch
from torchtitan.components import checkpoint as titan_checkpoint
from torchtitan.components.dataloader import BaseDataLoader

logger = logging.getLogger(__name__)


class EmptyDataLoader(BaseDataLoader):
    """The RL loop feeds microbatches directly; this is never iterated."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        pass

    def __init__(self, config: Config, **kwargs):
        self.config = config

    def __iter__(self):
        return iter(())

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class TiedCheckpointManager(titan_checkpoint.CheckpointManager):
    """CheckpointManager whose HF load tolerates keys the checkpoint does not ship.

    Upstream's ``dcp_load`` requests every exported key; a tied checkpoint has no
    ``lm_head.weight``. Missing keys are left to the adapter's ``from_hf``, and a
    rank that owns the lm_head but no embedding requests the embedding into an
    lm_head-shaped skeleton so the reconstruction has a source.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(titan_checkpoint.CheckpointManager.Config):
        pass

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        if not from_hf:
            return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)

        assert self.sd_adapter is not None
        hf_state = self.sd_adapter.to_hf(state_dict)
        index_mapping = getattr(self.sd_adapter, "fqn_to_index_mapping", None)
        if index_mapping:
            available = set(index_mapping)
            dropped = sorted(k for k in hf_state if k not in available)
            if dropped:
                logger.info(
                    f"HF checkpoint lacks {len(dropped)} exported key(s) (e.g. {dropped[:3]}); "
                    "deferring to the adapter's from_hf reconstruction"
                )
                lm_head_skeleton = hf_state.get("lm_head.weight")
                hf_state = {k: v for k, v in hf_state.items() if k in available}
                if "lm_head.weight" in dropped and lm_head_skeleton is not None:
                    embed_key = next((k for k in available if k.endswith("embed_tokens.weight")), None)
                    if embed_key is not None and embed_key not in hf_state:
                        hf_state[embed_key] = torch.empty_like(lm_head_skeleton)

        titan_checkpoint.dcp.load(
            hf_state,
            storage_reader=self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized),
        )
        self.states[titan_checkpoint.MODEL].load_state_dict(self.sd_adapter.from_hf(hf_state))
