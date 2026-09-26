import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from safetensors import safe_open

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.lora.utils import AdapterSpec, get_adapter_target_modules

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _is_hf_metadata_file(path: Path) -> bool:
    # The base checkpoint's weight index must not overwrite the exported shard mapping.
    return (
        path.is_file()
        and path.name != HF_EXPORT_COMPLETE_MARKER
        and path.suffix not in HF_WEIGHT_SUFFIXES
        and not path.name.endswith(".index.json")
    )


def _source_weight_map(base_checkpoint: Path) -> dict[str, str]:
    index_path = base_checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]

    checkpoint_path = base_checkpoint / "model.safetensors"
    if checkpoint_path.is_file():
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            return {name: checkpoint_path.name for name in checkpoint.keys()}

    raise ValueError(
        "--hf-export-source-tensor-prefixes requires a local safetensors "
        f"checkpoint, but none was found in {base_checkpoint}"
    )


def _copy_source_tensors(
    base_checkpoint: Path,
    output_path: Path,
    weight_map: dict[str, str],
    prefixes: Sequence[str],
) -> int:
    if isinstance(prefixes, (str, bytes)) or any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise ValueError("source_tensor_prefixes must be a sequence of nonempty strings")

    prefixes = tuple(prefixes)
    source_weight_map = _source_weight_map(base_checkpoint)
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in source_weight_map):
            raise ValueError(f"HF export source prefix has no matching weights: {prefix!r}")

    weights_by_shard: dict[str, list[str]] = {}
    for name, shard_name in source_weight_map.items():
        if name not in weight_map and name.startswith(prefixes):
            weights_by_shard.setdefault(shard_name, []).append(name)

    total_size = 0
    num_output_shards = len(weights_by_shard)
    for shard_index, (source_shard, names) in enumerate(weights_by_shard.items(), start=1):
        output_shard = f"model-source-{shard_index:05d}-of-{num_output_shards:05d}.safetensors"
        with safe_open(base_checkpoint / source_shard, framework="pt", device="cpu") as checkpoint:
            tensors = {name: checkpoint.get_tensor(name).contiguous() for name in names}
        safetensors.torch.save_file(tensors, output_path / output_shard)
        for name, tensor in tensors.items():
            weight_map[name] = output_shard
            total_size += tensor.numel() * tensor.element_size()

    return total_size


class SnapshotPublisher:
    def __init__(self, iterator: HfWeightIteratorBase, adapter_config: dict | None = None) -> None:
        assert iterator.placement.is_full_gather, "publishing requires full weights on rank 0"
        self._iterator = iterator
        self._adapter_config = adapter_config

    def publish_adapter(self, adapter: AdapterSpec | None, path: str, metadata: dict | None = None) -> None:
        write_checkpoint_dir(
            path,
            lambda checkpoint_dir: self.write_adapter(adapter, checkpoint_dir),
            metadata=metadata,
            overwrite=False,
        )

    @torch.no_grad()
    def write_adapter(self, adapter: AdapterSpec | None, path: str | Path) -> None:
        """Collectively write HF adapter files into the caller's checkpoint directory."""
        assert self._adapter_config is not None, "adapter export requires adapter_config"
        path = Path(path)
        is_writer = dist.get_rank() == 0
        adapter_tensors = {
            name: tensor.detach().contiguous().cpu()
            for name, tensor in self._iterator.materialize_adapter(adapter, materialize=is_writer).items()
        }
        adapter_bytes = safetensors.torch.save(adapter_tensors) if is_writer else None
        adapter_config = self._adapter_config
        if adapter is not None:
            adapter_config = adapter_config | {"r": adapter.rank, "lora_alpha": adapter.alpha}

        if is_writer:
            adapter_config = adapter_config | {"target_modules": get_adapter_target_modules(adapter_tensors)}
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text(json.dumps(adapter_config))
            (path / "adapter_model.safetensors").write_bytes(adapter_bytes)

    def write_model(
        self,
        path: str | Path,
        *,
        weights: Mapping[str, torch.Tensor],
        hf_checkpoint: str,
        source_tensor_prefixes: Sequence[str] = (),
    ) -> None:
        """Collectively write HF model shards into the caller's checkpoint directory."""
        path = Path(path)
        is_writer = dist.get_rank() == 0

        weight_map: dict[str, str] = {}
        total_size = 0
        shard_index = 0
        write_error = None
        for hf_named_tensors in self._iterator.iter_hf_weights(weights):
            if not is_writer or write_error is not None:
                continue
            shard_index += 1
            shard_name = f"model-{shard_index:05d}.safetensors"
            shard_tensors = {}
            for name, tensor in hf_named_tensors:
                shard_tensors[name] = tensor.detach().to("cpu").contiguous()
                weight_map[name] = shard_name
                total_size += shard_tensors[name].numel() * shard_tensors[name].element_size()
            try:
                safetensors.torch.save_file(shard_tensors, path / shard_name)
            except Exception as exc:
                # Peers must finish the remaining weight gathers before reporting a write failure.
                write_error = exc
            del shard_tensors

        if write_error is not None:
            raise write_error
        if is_writer:
            assert weight_map, f"HF export to {path} produced no weights"
            base_checkpoint = Path(hf_checkpoint)
            if base_checkpoint.is_dir():
                if source_tensor_prefixes:
                    total_size += _copy_source_tensors(
                        base_checkpoint,
                        path,
                        weight_map,
                        source_tensor_prefixes,
                    )
                for meta_file in base_checkpoint.iterdir():
                    if _is_hf_metadata_file(meta_file):
                        shutil.copy2(meta_file, path / meta_file.name)
            else:
                if source_tensor_prefixes:
                    raise ValueError(
                        "--hf-export-source-tensor-prefixes requires --hf-checkpoint to be a local directory"
                    )
                logger.warning(f"hf_checkpoint {hf_checkpoint} is not a local dir; metadata not copied to {path}")
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
