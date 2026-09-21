import json
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.multi_lora import AdapterSpec


class WeightPublisher:
    def __init__(self, iterator: HfWeightIteratorBase, adapter_config: dict) -> None:
        assert iterator.placement.is_full_gather, "publishing requires the full adapter on rank 0"
        self._iterator = iterator
        self._adapter_config = adapter_config

    @torch.no_grad()
    def publish_adapter(self, adapter: AdapterSpec, path: str, metadata: dict | None = None) -> None:
        is_writer = dist.get_rank() == 0

        def write_shards(tmp_dir: Path):
            tensors = {
                name: tensor.detach().contiguous().cpu()
                for name, tensor in self._iterator.materialize_adapter(adapter, materialize=is_writer).items()
            }
            data = safetensors.torch.save(tensors) if is_writer else None

            if is_writer:
                config = self._adapter_config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
                (tmp_dir / "adapter_config.json").write_text(json.dumps(config))
                (tmp_dir / "adapter_model.safetensors").write_bytes(data)

        write_checkpoint_dir(path, write_shards, metadata=metadata, overwrite=False)
