import json
import logging
import shutil
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.hf_utils.weight_mapping import missing_base_tensors

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _is_hf_metadata_file(path: Path) -> bool:
    # The base checkpoint's weight index must not overwrite the exported shard mapping.
    return path.is_file() and path.suffix not in HF_WEIGHT_SUFFIXES and not path.name.endswith(".index.json")


# Keyed by the missing-tensor set (or the failure), so one run warns once, not per save and eval snapshot.
_partial_export_warnings_issued: set[Hashable] = set()


def _log_partial_export_warning(path: Path, exported_names: Collection[str], hf_checkpoint: str) -> None:
    """Warn once per process when the export holds fewer tensors than its base checkpoint.

    Advisory only: any failure is logged, never raised, so it cannot fail a finished export.
    """
    key: Hashable
    try:
        base_checkpoint = Path(hf_checkpoint)
        if not base_checkpoint.is_dir():
            return
        missing_by_name = missing_base_tensors(exported_names, base_checkpoint)
        if not missing_by_name:
            return
        missing = sorted(missing_by_name)
        key = frozenset(missing)
        examples = ", ".join(missing[:3])
        message = (
            f"The export to {path} holds {len(exported_names)} tensors, but {len(missing)} tensors of "
            f"{hf_checkpoint} were not exported (e.g. {examples}). The export loads correctly only on "
            f"top of the base checkpoint. To make it a standalone checkpoint, run: "
            f'python tools/complete_hf_export.py --export-dir "{path}" --hf-checkpoint "{hf_checkpoint}"'
        )
    except Exception as exc:
        key = repr(exc)
        message = f"Could not check the export to {path} for missing base tensors: {exc!r}"
    if key in _partial_export_warnings_issued:
        return
    _partial_export_warnings_issued.add(key)
    logger.warning(message)


class SnapshotPublisher:
    def __init__(self, iterator: HfWeightIteratorBase) -> None:
        assert iterator.placement.is_full_gather, "publishing requires full weights on rank 0"
        self._iterator = iterator

    def write_model(self, path: str | Path, *, weights: Mapping[str, torch.Tensor], hf_checkpoint: str) -> None:
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
                for meta_file in base_checkpoint.iterdir():
                    if _is_hf_metadata_file(meta_file):
                        shutil.copy2(meta_file, path / meta_file.name)
            else:
                logger.warning(f"hf_checkpoint {hf_checkpoint} is not a local dir; metadata not copied to {path}")
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
            _log_partial_export_warning(path, weight_map.keys(), hf_checkpoint)
