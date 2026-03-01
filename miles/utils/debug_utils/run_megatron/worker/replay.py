"""Routing replay stage management for standalone Megatron worker."""

from pathlib import Path
from typing import Any

import torch

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.replay_base import routing_replay_manager

_REPLAY_FORMAT_VERSION: int = 1


def load_replay_data(
    script: WorkerScriptArgs,
    *,
    rank: int,
    sequence_parallel: bool = False,
) -> None:
    """Load routing replay data from disk before forward pass.

    Supports two loading modes:
    1. Per-rank files (rank{N}_routing_replay.pt) — exact match, no slicing
    2. Single-rank fallback (rank0_routing_replay.pt) — with CP zigzag slicing
       and SP slicing for cross-config comparison
    """
    if not script.routing_replay_load_path:
        return

    per_rank_file: Path = _replay_file_path(base_dir=script.routing_replay_load_path, rank=rank)

    if per_rank_file.exists():
        _load_per_rank_file(per_rank_file, rank=rank)
    else:
        rank0_file: Path = _replay_file_path(base_dir=script.routing_replay_load_path, rank=0)
        if not rank0_file.exists():
            print(f"[worker rank={rank}] WARNING: replay file not found: {per_rank_file}", flush=True)
            return
        _load_with_slicing(rank0_file, rank=rank, sequence_parallel=sequence_parallel)


def setup_replay_stage(script: WorkerScriptArgs) -> None:
    """Set routing replay manager stage based on CLI args.

    The replay manager hooks are registered during model construction
    (when ``--use-routing-replay`` / ``--use-rollout-routing-replay`` is set).
    Here we only set the stage so the hooks know whether to record or replay.
    """
    if script.routing_replay_dump_path:
        routing_replay_manager.stage = "record"
        print(f"[worker] Routing replay stage=record (dump → {script.routing_replay_dump_path})", flush=True)
    elif script.routing_replay_load_path:
        routing_replay_manager.stage = "replay_forward"
        print(f"[worker] Routing replay stage=replay_forward (load ← {script.routing_replay_load_path})", flush=True)


def save_replay_data(script: WorkerScriptArgs, *, rank: int) -> None:
    """Save recorded routing replay data to disk."""
    if not script.routing_replay_dump_path:
        return

    script.routing_replay_dump_path.mkdir(parents=True, exist_ok=True)

    replays_data: list[list[torch.Tensor]] = [replay.top_indices_list for replay in routing_replay_manager.replays]
    total_entries: int = sum(len(d) for d in replays_data)

    if total_entries > 0:
        payload: dict[str, Any] = {
            "version": _REPLAY_FORMAT_VERSION,
            "replays": replays_data,
        }
        save_path: Path = _replay_file_path(base_dir=script.routing_replay_dump_path, rank=rank)
        torch.save(payload, save_path)
        if rank == 0:
            print(
                f"[worker] Saved routing replay ({total_entries} entries, {len(replays_data)} replays) → {save_path}",
                flush=True,
            )


def _load_per_rank_file(replay_file: Path, *, rank: int) -> None:
    """Load replay data from a per-rank file (exact match, no slicing)."""
    payload: dict[str, Any] = torch.load(replay_file, weights_only=False)

    saved_replays: list[list[torch.Tensor]] = payload["replays"]
    expected: int = len(routing_replay_manager.replays)
    if len(saved_replays) != expected:
        raise ValueError(f"Replay file has {len(saved_replays)} replays but model expects {expected}")

    total_entries: int = 0
    for replay, data in zip(routing_replay_manager.replays, saved_replays, strict=True):
        replay.top_indices_list = data
        total_entries += len(data)

    if rank == 0:
        print(
            f"[worker] Loaded routing replay ({total_entries} entries, {expected} replays) ← {replay_file}",
            flush=True,
        )


def _load_with_slicing(
    replay_file: Path,
    *,
    rank: int,
    sequence_parallel: bool,
) -> None:
    """Load replay from a single-rank file with CP zigzag slicing and SP slicing.

    Used for cross-config comparison where baseline has fewer ranks than target.
    """
    from megatron.core import mpu

    payload: dict[str, Any] = torch.load(replay_file, weights_only=False)
    saved_replays: list[list[torch.Tensor]] = payload["replays"]

    expected: int = len(routing_replay_manager.replays)
    if len(saved_replays) != expected:
        raise ValueError(f"Replay file has {len(saved_replays)} replays but model expects {expected}")

    cp_size: int = mpu.get_context_parallel_world_size() if mpu.is_initialized() else 1
    cp_rank: int = mpu.get_context_parallel_rank() if mpu.is_initialized() else 0
    tp_size: int = mpu.get_tensor_model_parallel_world_size() if mpu.is_initialized() else 1
    tp_rank: int = mpu.get_tensor_model_parallel_rank() if mpu.is_initialized() else 0

    do_sp_slice: bool = sequence_parallel and routing_replay_manager.if_sp_region and tp_size > 1

    total_entries: int = 0
    for replay_idx, (replay, indices_list) in enumerate(
        zip(routing_replay_manager.replays, saved_replays, strict=True)
    ):
        sliced: list[torch.Tensor] = indices_list

        if cp_size > 1:
            from megatron.core.transformer.deepseek_v4_cp_utils import natural_to_zigzag_slice

            sliced = [natural_to_zigzag_slice(t, dim=0, cp_size=cp_size, cp_rank=cp_rank) for t in sliced]

        if do_sp_slice:
            sliced = [_sp_slice(t, tp_size=tp_size, tp_rank=tp_rank) for t in sliced]

        replay.top_indices_list = sliced
        replay.forward_index = 0
        replay.backward_index = 0
        total_entries += len(sliced)

        if rank == 0:
            shapes_before: list[torch.Size] = [t.shape for t in indices_list]
            shapes_after: list[torch.Size] = [t.shape for t in sliced]
            print(
                f"[worker] replay[{replay_idx}]: cp={cp_size}/{cp_rank}, tp={tp_size}/{tp_rank}, "
                f"sp={sequence_parallel}, shapes {shapes_before} → {shapes_after}",
                flush=True,
            )

    if rank == 0:
        print(
            f"[worker] Loaded routing replay with slicing ({total_entries} entries, {expected} replays) "
            f"← {replay_file}",
            flush=True,
        )


def _sp_slice(tensor: torch.Tensor, *, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Slice tensor along dim=0 for sequence parallelism."""
    seqlen: int = tensor.size(0)
    assert seqlen % tp_size == 0, f"seqlen {seqlen} not divisible by tp_size {tp_size}"
    chunk_size: int = seqlen // tp_size
    start: int = chunk_size * tp_rank
    end: int = chunk_size * (tp_rank + 1)
    return tensor[start:end]


def _replay_file_path(*, base_dir: Path, rank: int) -> Path:
    return base_dir / f"rank{rank}_{routing_replay_manager.filename}"
