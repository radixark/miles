import logging
from dataclasses import dataclass

from torchtitan.components import checkpoint as titan_checkpoint
from torchtitan.components.dataloader import BaseDataLoader

logger = logging.getLogger(__name__)


class EmptyDataLoader(BaseDataLoader):
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
    @dataclass(kw_only=True, slots=True)
    class Config(titan_checkpoint.CheckpointManager.Config):
        pass

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        if not from_hf:
            return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)

        assert self.sd_adapter is not None
        hf_state = self.sd_adapter.to_hf(state_dict)
        if self.sd_adapter.fqn_to_index_mapping:
            available = set(self.sd_adapter.fqn_to_index_mapping)
            dropped = sorted(k for k in hf_state if k not in available)
            if dropped:
                logger.info(
                    f"HF checkpoint lacks {len(dropped)} exported key(s) (e.g. {dropped[:3]}); "
                    "deferring to the adapter's from_hf reconstruction"
                )
                hf_state = {k: v for k, v in hf_state.items() if k in available}

        titan_checkpoint.dcp.load(
            hf_state,
            storage_reader=self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized),
        )
        self.states[titan_checkpoint.MODEL].load_state_dict(self.sd_adapter.from_hf(hf_state))
