"""The two torchtitan components miles swaps into the trainer's config tree.

Both exist because the RL loop, not the trainer, owns the data: the
dataloader is an empty stub since microbatches are fed directly, and the
checkpoint manager's HF load tolerates the tied checkpoints that torchtitan's
flavors do not model.
"""

import logging
from dataclasses import dataclass

import torch
from torchtitan.components import checkpoint as titan_checkpoint
from torchtitan.components.dataloader import BaseDataLoader

logger = logging.getLogger(__name__)


class EmptyDataLoader(BaseDataLoader):
    """The RL loop feeds microbatches directly; the trainer's own dataloader
    is never iterated and checkpoints no state."""

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
    """CheckpointManager whose HF load survives tied checkpoints.

    torchtitan flavors qwen3_5 with a separate ``lm_head`` while the HF
    checkpoint ties it to the embedding and ships no ``lm_head.weight``;
    upstream ``dcp_load`` requests every exported key and dies on the missing
    one. The from_hf branch below is upstream's, plus: keys the checkpoint
    does not ship are dropped from the request (the adapter's ``from_hf``
    reconstructs them), and when the dropped key is the tied lm_head on a rank
    that owns no embedding (a PP last stage), the checkpoint's embedding is
    requested into an lm_head-shaped skeleton so the reconstruction has a
    source.
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
