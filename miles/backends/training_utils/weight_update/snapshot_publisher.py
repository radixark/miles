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

    def publish_adapter(self, adapter: AdapterSpec | None, path: str, metadata: dict | None = None) -> None:
        write_checkpoint_dir(
            path, lambda tmp_dir: self.write_adapter(adapter, tmp_dir), metadata=metadata, overwrite=False
        )

    @torch.no_grad()
    def write_adapter(self, adapter: AdapterSpec | None, path: str | Path) -> None:
        """Write adapter files inside a caller-owned checkpoint directory transaction."""
        path = Path(path)
        is_writer = dist.get_rank() == 0
        tensors = {
            name: tensor.detach().contiguous().cpu()
            for name, tensor in self._iterator.materialize_adapter(adapter, materialize=is_writer).items()
        }
        adapter_bytes = safetensors.torch.save(tensors) if is_writer else None
        config = self._adapter_config
        if adapter is not None:
            config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}

        if is_writer:
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text(json.dumps(config))
            (path / "adapter_model.safetensors").write_bytes(adapter_bytes)
