"""HF-format export of the live Megatron model.

``export_hf_model_direct`` goes through miles' own megatron->HF converters (the
weight updater's machinery), so export coverage always matches weight-sync
coverage; ``save_hf_model`` picks between it and the Megatron-Bridge exporter
(LoRA needs the bridge for adapter merging) and writes a ``.complete`` marker.
Everything here is collective: all ranks must call it. Direct exports may assign
distinct immutable shards to multiple ranks; global rank 0 commits the index.
"""

import json
import logging
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from pathlib import Path

import safetensors.torch
import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.utils import is_lora_model, save_lora_checkpoint
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER, load_hf_config
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


class _AsyncShardWriter:
    """Write one immutable shard at a time while the main thread gathers the next."""

    def __init__(self, path: Path):
        self._path = path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-export")
        self._pending: Future | None = None

    def submit(self, shard_name: str, tensors: dict[str, torch.Tensor]) -> None:
        self._wait()
        snapshot = {
            name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in tensors.items()
        }
        self._pending = self._executor.submit(safetensors.torch.save_file, snapshot, self._path / shard_name)

    def finish(self) -> None:
        try:
            self._wait()
        finally:
            self._executor.shutdown(wait=True)

    def _wait(self) -> None:
        if self._pending is None:
            return
        try:
            self._pending.result()
        finally:
            self._pending = None


def _raise_distributed_errors(error: Exception | None) -> None:
    message = None if error is None else f"{type(error).__name__}: {error}"
    group = get_gloo_group()
    errors = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.all_gather_object(errors, message, group=group)
    failures = [f"rank {rank}: {rank_error}" for rank, rank_error in enumerate(errors) if rank_error is not None]
    if failures:
        collective_error = RuntimeError("HF export failed:\n" + "\n".join(failures))
        if error is not None:
            raise collective_error from error
        raise collective_error


def _run_collectively(operation: Callable[[], None]) -> None:
    error = None
    try:
        operation()
    except Exception as exc:
        error = exc
    _raise_distributed_errors(error)


def _is_hf_metadata_file(path: Path) -> bool:
    """Tokenizer/config files worth copying into an export — not weights, and not the
    base checkpoint's weight index, which would clobber the one the export writes."""
    return (
        path.is_file()
        and path.name != HF_EXPORT_COMPLETE_MARKER
        and path.suffix not in HF_WEIGHT_SUFFIXES
        and not path.name.endswith(".index.json")
    )


def _prepare_export_directory(path: Path, *, is_writer: bool) -> None:
    if is_writer:
        path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        # A stale marker from an earlier attempt must never vouch for partial shards.
        (path / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)


def _write_direct_shards(
    iterator,
    megatron_local_weights,
    path: Path,
    *,
    replica_rank: int,
    writer_replica_ranks: list[int],
) -> tuple[dict[str, str], int]:
    writer = _AsyncShardWriter(path) if replica_rank in writer_replica_ranks else None
    writer_loads = dict.fromkeys(writer_replica_ranks, 0)
    weight_map: dict[str, str] = {}
    local_error: Exception | None = None

    for shard_index, named_tensors in enumerate(iterator.iter_hf_weights(megatron_local_weights), start=1):
        if local_error is not None:
            continue
        try:
            shard_name = f"model-{shard_index:05d}.safetensors"
            tensors = dict(named_tensors)
            if len(tensors) != len(named_tensors):
                raise ValueError(f"HF shard {shard_name} contains duplicate tensor names")
            duplicates = weight_map.keys() & tensors.keys()
            if duplicates:
                raise ValueError(f"duplicate HF tensor: {min(duplicates)}")

            shard_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            owner = min(writer_loads, key=writer_loads.__getitem__)
            writer_loads[owner] += shard_size
            weight_map.update(dict.fromkeys(tensors, shard_name))
            if replica_rank == owner:
                assert writer is not None
                writer.submit(shard_name, tensors)
        except Exception as exc:
            # Continue driving the collective iterator so a rank-local I/O failure
            # cannot strand peers in a later model-weight gather.
            local_error = exc

    if writer is not None:
        try:
            writer.finish()
        except Exception as exc:
            local_error = local_error or exc
    _raise_distributed_errors(local_error)

    if not weight_map:
        raise ValueError("HF export produced no weights")
    return weight_map, sum(writer_loads.values())


def _write_export_metadata(args, path: Path, weight_map: dict[str, str], total_size: int) -> None:
    if torch.distributed.get_rank() != 0:
        return
    base_checkpoint = Path(args.hf_checkpoint)
    if base_checkpoint.is_dir():
        for meta_file in base_checkpoint.iterdir():
            if _is_hf_metadata_file(meta_file):
                shutil.copy2(meta_file, path / meta_file.name)
    else:
        logger.warning(f"hf_checkpoint {args.hf_checkpoint} is not a local dir; metadata not copied to {path}")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


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
    placement = WeightUpdatePlacement(gather_pp=True)
    replica_rank, replica_size = get_data_replica_rank_and_size(get_parallel_state(), placement)
    if args.save_hf_writers > replica_size:
        raise ValueError(
            f"--save-hf-writers={args.save_hf_writers} exceeds the {replica_size} complete model replicas"
        )
    writer_replica_ranks = list(range(args.save_hf_writers))
    _run_collectively(
        lambda: _prepare_export_directory(path, is_writer=replica_rank in writer_replica_ranks)
    )
    iterator = HfWeightIteratorDirect(
        args,
        model,
        placement=placement,
        model_name=model_name,
        quantization_config=quantization_config,
    )

    weight_map, total_size = _write_direct_shards(
        iterator,
        megatron_local_weights,
        path,
        replica_rank=replica_rank,
        writer_replica_ranks=writer_replica_ranks,
    )
    _run_collectively(lambda: _write_export_metadata(args, path, weight_map, total_size))


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
