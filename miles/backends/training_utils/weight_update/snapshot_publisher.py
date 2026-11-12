import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from safetensors import safe_open

from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.hf_utils.weight_mapping import get_checkpoint_weight_map, get_param_name_remap

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _is_hf_metadata_file(path: Path) -> bool:
    # The base checkpoint's weight index must not overwrite the exported shard mapping.
    return path.is_file() and path.suffix not in HF_WEIGHT_SUFFIXES and not path.name.endswith(".index.json")


class SnapshotPublisher:
    def __init__(self, iterator: HfWeightIteratorBase) -> None:
        assert iterator.placement.is_full_gather, "publishing requires full weights on rank 0"
        self._iterator = iterator

    def write_model(self, path: str | Path, *, weights: Mapping[str, torch.Tensor], hf_checkpoint: str) -> None:
        """Collectively write HF model shards into the caller's checkpoint directory."""
        path = Path(path)
        is_writer = dist.get_rank() == 0
        base_checkpoint = Path(hf_checkpoint)
        base_weight_map = get_checkpoint_weight_map(base_checkpoint) if is_writer and base_checkpoint.is_dir() else {}
        # The export uses the engines' tensor names, which for DeepSeek-V4 differ from its native checkpoint's; the
        # snapshot keeps the checkpoint's.
        base_name_by_export_name = {}
        if base_weight_map:
            remap_name = get_param_name_remap(str(base_checkpoint / "config.json"), base_weight_map)
            base_name_by_export_name = {remap_name(name): name for name in base_weight_map}
        unmatched: list[str] = []

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
            for export_name, tensor in hf_named_tensors:
                if base_weight_map and export_name not in base_name_by_export_name:
                    unmatched.append(export_name)
                name = base_name_by_export_name.get(export_name, export_name)
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
            assert not unmatched, (
                f"HF export to {path} wrote {len(unmatched)} tensors with no counterpart in {hf_checkpoint}, "
                f"e.g. {unmatched[:3]}"
            )
            if base_weight_map:
                unexported_by_shard: dict[str, list[str]] = {}
                for name, shard in base_weight_map.items():
                    if name not in weight_map:
                        unexported_by_shard.setdefault(shard, []).append(name)
                for shard, names in sorted(unexported_by_shard.items()):
                    with safe_open(base_checkpoint / shard, framework="pt", device="cpu") as checkpoint:
                        unexported = {name: checkpoint.get_tensor(name) for name in names}
                    shard_index += 1
                    shard_name = f"model-{shard_index:05d}.safetensors"
                    safetensors.torch.save_file(unexported, path / shard_name)
                    for name, tensor in unexported.items():
                        weight_map[name] = shard_name
                        total_size += tensor.numel() * tensor.element_size()
                for meta_file in base_checkpoint.iterdir():
                    if _is_hf_metadata_file(meta_file):
                        shutil.copy2(meta_file, path / meta_file.name)
            else:
                logger.warning(
                    f"hf_checkpoint {hf_checkpoint} is not a local dir; its unexported weights and metadata are not "
                    f"copied to {path}"
                )
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
