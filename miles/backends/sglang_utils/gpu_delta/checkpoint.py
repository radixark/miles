"""Canonical metadata for the GPU-delta receiver's immutable startup checkpoint.

Read only at first delta admission. The exclusive controller must keep the
original local checkpoint unchanged for the engine's lifetime; this is metadata
discovery, not proof of the current model's weight values. Ordinary loaders and
model methods are never modified.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from miles.utils.gpu_delta.publication import StaticPolicy, read_safetensors_headers, static_policy


@dataclass(frozen=True)
class CanonicalCheckpoint:
    """The startup checkpoint as admission reads it: safetensors headers and its own config.json."""

    # name -> {"shape", "dtype"} of every selected tensor.
    inventory: dict
    # The checkpoint's static-tensor policy, shared with the trainer's plan.
    policy: StaticPolicy
    # name -> (path, offset, nbytes, dtype, shape) of every selected tensor.
    locations: dict

    def read_rows(self, name, start, stop):
        """Bytes of rows ``[start, stop)`` of one canonical tensor; only canonical shadows read tensor data."""
        path, offset, nbytes, _, shape = self.locations[name]
        if not shape or not 0 <= start < stop <= shape[0] or nbytes % shape[0]:
            raise ValueError(f"canonical checkpoint rows [{start}, {stop}) are outside {name}{list(shape)}")
        row = nbytes // shape[0]
        with open(path, "rb") as source:
            source.seek(offset + start * row)
            data = source.read((stop - start) * row)
        if len(data) != (stop - start) * row:
            raise ValueError(f"canonical checkpoint tensor is truncated: {name}")
        return np.frombuffer(data, dtype=np.uint8)


def read_canonical_checkpoint(model_runner):
    """Read source names, shapes, dtypes and the static policy without materializing any tensors.

    The source inventory is independent of quantization. Runtime layout admission
    checks the physical buffers and model mappings separately. The policy comes from
    the checkpoint's own config.json, as in the trainer's plan, never from the
    configuration SGLang serves.
    """
    from sglang.srt.model_loader.loader import DefaultModelLoader, ModelOptModelLoader
    from sglang.srt.model_loader.weight_utils import filter_duplicate_safetensors_files, maybe_add_mtp_safetensors

    model = model_runner.model
    config = model_runner.load_config
    model_config = model_runner.model_config
    loader = model_runner.loader
    # Group offloading retains prefetched GPU copies independently of the CPU
    # parameter storage. Updating that storage would leave those copies stale.
    offload = model_runner.server_args
    if offload.cpu_offload_gb == 0 and offload.offload_group_size > 0:
        raise ValueError("GPU delta does not support prefetched group offloading")
    # ModelOpt's already-quantized branch delegates to DefaultModelLoader.
    # Subclasses may override the iterator or transform its names and values.
    if type(loader) not in (DefaultModelLoader, ModelOptModelLoader):
        raise ValueError("GPU delta requires the standard local checkpoint loader")
    if type(loader) is ModelOptModelLoader and not model_config._is_already_quantized():
        raise ValueError("GPU delta does not support load-time ModelOpt conversion")
    if config.load_format not in {"auto", "safetensors", "fastsafetensors"}:
        raise ValueError("GPU delta requires a standard safetensors load format")
    if model_runner.is_draft_worker or config.draft_model_idx is not None:
        raise ValueError("GPU delta checkpoint admission requires the target model")
    if (
        getattr(model, "secondary_weights", ()) or getattr(model, "allow_patterns_overrides", None) is not None
    ):  # config-access-exempt: Standard SGLang model classes may declare optional alternate checkpoint sources.
        raise ValueError("GPU delta does not support secondary or remapped sources")
    if config.decryption_key_file is not None:
        raise ValueError("GPU delta does not support encrypted checkpoint sources")

    folder = Path(model_config.model_path)
    if not folder.is_dir():
        raise ValueError("GPU delta requires the original immutable local checkpoint")
    policy = static_policy(json.loads((folder / "config.json").read_text()))
    # Match the default loader's primary source (empty prefix, no draft remap).
    # Reuse its index filtering and bundled-MTP rules without invoking downloads,
    # checksum verification, tensor loading, or checkpoint page-cache prefetch.
    files = filter_duplicate_safetensors_files(
        [str(path) for path in folder.glob("*.safetensors")],
        str(folder),
        "model.safetensors.index.json",
    )
    files = maybe_add_mtp_safetensors(files, str(folder), "model.safetensors.index.json", model_config.hf_config)
    if not files:
        raise ValueError("GPU delta checkpoint has no selected safetensors shards")
    locations = read_safetensors_headers(files)
    inventory = {name: {"shape": shape, "dtype": dtype} for name, (_, _, _, dtype, shape) in locations.items()}
    if not inventory:
        raise ValueError("canonical checkpoint metadata is empty")
    return CanonicalCheckpoint(inventory=inventory, policy=policy, locations=locations)
