"""HF-format export of the live Megatron model.

``export_hf_model_direct`` goes through miles' own megatron->HF converters (the
weight updater's machinery), so export coverage always matches weight-sync
coverage; ``save_hf_model`` picks between it and the Megatron-Bridge exporter
(LoRA needs the bridge for adapter merging) and writes a ``.complete`` marker.
Everything here is collective: all ranks must call it, global rank 0 writes.
"""

import json
import logging
import shutil
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import safetensors.torch
import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora_utils import is_lora_model, save_lora_checkpoint
from miles.backends.megatron_utils.update_weight.common import named_params_and_buffers
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER, load_hf_config
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _is_hf_metadata_file(path: Path) -> bool:
    """Tokenizer/config files worth copying into an export — not weights, and not the
    base checkpoint's weight index, which would clobber the one the export writes."""
    return path.is_file() and path.suffix not in HF_WEIGHT_SUFFIXES and not path.name.endswith(".index.json")


def export_hf_model_direct(
    args,
    model: Sequence[DDP],
    path: str | Path,
    *,
    model_name: str,
    quantization_config,
    megatron_local_weights,
) -> None:
    """Export current weights as an HF checkpoint via miles' own megatron->HF converters.

    Same conversion machinery as the weight updater, so export coverage matches
    weight-sync coverage (the bridge silently exports zero weights for specs it has
    no mapping for, e.g. qwen3.5). Collective — all ranks must call it; rank 0 writes.
    """
    path = Path(path)
    is_writer = torch.distributed.get_rank() == 0
    if is_writer:
        path.mkdir(parents=True, exist_ok=True)
        # A stale marker from an earlier run would vouch for this run's half-written shards.
        (path / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)

    iterator = HfWeightIteratorDirect(args, model, model_name=model_name, quantization_config=quantization_config)

    weight_map: dict[str, str] = {}
    total_size = 0
    shard_index = 0
    for hf_named_tensors in iterator.get_hf_weight_chunks(megatron_local_weights):
        if not is_writer:
            continue
        shard_index += 1
        shard_name = f"model-{shard_index:05d}.safetensors"
        shard_tensors = {}
        for name, tensor in hf_named_tensors:
            shard_tensors[name] = tensor.detach().to("cpu").contiguous()
            weight_map[name] = shard_name
            total_size += shard_tensors[name].numel() * shard_tensors[name].element_size()
        safetensors.torch.save_file(shard_tensors, path / shard_name)
        del shard_tensors

    try:
        if is_writer:
            assert weight_map, f"HF export to {path} produced no weights"
            base_checkpoint = Path(args.hf_checkpoint)
            if base_checkpoint.is_dir():
                for meta_file in base_checkpoint.iterdir():
                    if _is_hf_metadata_file(meta_file):
                        shutil.copy2(meta_file, path / meta_file.name)
            else:
                logger.warning(f"hf_checkpoint {args.hf_checkpoint} is not a local dir; metadata not copied to {path}")
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    finally:
        # In a finally: rank 0 is the only rank that can fail above.
        torch.distributed.barrier()


@cache
def _get_hf_bridge(hf_checkpoint: str):
    # Local: megatron.bridge is only needed on the bridge export path.
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


def save_hf_model(
    args,
    rollout_id: int,
    model: Sequence[DDP],
    *,
    path: str | Path | None = None,
    raise_on_error: bool = False,
) -> None:
    """Save Megatron model in HuggingFace format.

    For LoRA models this saves both:
    - A **merged** HF model (adapter weights folded into base) at ``{path}/``
      so it can be loaded directly with ``AutoModelForCausalLM.from_pretrained``.
    - An **adapter-only** HF PEFT checkpoint at ``{path}/adapter/``
      so it can be loaded with ``PeftModel.from_pretrained``.

    This function is collective — all ranks must call it. On success, global rank 0
    writes a ``.complete`` marker file.

    Args:
        args: Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
        path: Destination directory; defaults to ``args.save_hf.format(rollout_id)``.
        raise_on_error: Re-raise export failures instead of logging them.
    """
    should_log = get_parallel_state().effective_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0
    path = Path(path if path is not None else args.save_hf.format(rollout_id=rollout_id))

    try:
        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        if args.megatron_to_hf_mode == "raw" and not is_lora_model(model):
            # LoRA keeps the bridge (adapter merging).
            hf_config = load_hf_config(args.hf_checkpoint)
            export_hf_model_direct(
                args,
                model,
                path,
                model_name=type(hf_config).__name__.lower() if args.model_name is None else args.model_name,
                quantization_config=getattr(hf_config, "quantization_config", None),
                megatron_local_weights=dict(named_params_and_buffers(args, model, convert_to_global_name=True)),
            )
        else:
            bridge = _get_hf_bridge(args.hf_checkpoint)
            path.mkdir(parents=True, exist_ok=True)
            if torch.distributed.get_rank() == 0:
                (path / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)
            with patch_megatron_model(model):
                # For LoRA models, merge_adapter_weights=True (default) merges
                # adapter weights into base weights for a standalone HF model.
                bridge.save_hf_pretrained(model, path=path)

            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
                    raise RuntimeError(
                        f"HF export to {path} produced no weight files — the megatron "
                        f"bridge likely has no mapping for this model architecture."
                    )

        if should_log:
            logger.info(f"Successfully saved merged HuggingFace model to {path}")
    except Exception as e:
        if raise_on_error:
            raise
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")
        return

    # Additionally save adapter-only checkpoint for LoRA models
    if is_lora_model(model):
        try:
            adapter_path = path / "adapter"
            if should_log:
                logger.info(f"Saving LoRA adapter (HF PEFT format) to {adapter_path}")
            save_lora_checkpoint(model, args, str(adapter_path))
            if should_log:
                logger.info(f"Successfully saved LoRA adapter to {adapter_path}")
        except Exception as e:
            if raise_on_error:
                raise
            if should_log:
                logger.error(f"Failed to save LoRA adapter: {e}")
            return

    if torch.distributed.get_rank() == 0:
        (path / HF_EXPORT_COMPLETE_MARKER).touch()
