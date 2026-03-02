"""Routing replay stage management for standalone Megatron worker.

Only single-rank (nproc=1) baselines are supported: save always writes
a rank-0 file, and load always reads that file with CP/SP slicing.
"""

from pathlib import Path

import torch

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.replay_base import routing_replay_manager


def load_replay_data(
    script: WorkerScriptArgs,
    *,
    rank: int,
    sequence_parallel: bool = False,
) -> None:
    """Load routing replay data from rank 0's file with CP/SP slicing."""
    if not script.routing_replay_load_path:
        return

    load_dir: Path = script.routing_replay_load_path
    replay_files: list[Path] = sorted(load_dir.glob(f"*_{routing_replay_manager.filename}"))
    if len(replay_files) != 1:
        raise ValueError(
            f"Expected exactly 1 replay file in {load_dir}, "
            f"found {len(replay_files)}: {[f.name for f in replay_files]}"
        )

    replay_file: Path = replay_files[0]
    _load_replay(replay_file, rank=rank, sequence_parallel=sequence_parallel)


def setup_replay_before_model(script: WorkerScriptArgs) -> None:
    """Enable replay manager and set stage BEFORE model construction.

    Must be called before ``get_model()`` so that ``register_to_module``
    (called during model construction) sees ``enabled=True`` and creates
    Replay objects on MoE modules.
    """
    if script.routing_replay_dump_path:
        routing_replay_manager.enabled = True
        routing_replay_manager.stage = "record"
        print(f"[worker] Routing replay enabled, stage=record (dump → {script.routing_replay_dump_path})", flush=True)
    elif script.routing_replay_load_path:
        routing_replay_manager.enabled = True
        routing_replay_manager.stage = "replay_forward"
        print(
            f"[worker] Routing replay enabled, stage=replay_forward (load ← {script.routing_replay_load_path})",
            flush=True,
        )


def save_replay_data(script: WorkerScriptArgs, *, rank: int) -> None:
    """Save recorded routing replay data to disk (rank 0 only)."""
    if not script.routing_replay_dump_path:
        return
    if rank != 0:
        return

    script.routing_replay_dump_path.mkdir(parents=True, exist_ok=True)

    replays_data: list[list[torch.Tensor]] = [replay.top_indices_list for replay in routing_replay_manager.replays]
    total_entries: int = sum(len(d) for d in replays_data)

    if total_entries == 0:
        return

    save_path: Path = _replay_file_path(base_dir=script.routing_replay_dump_path)
    torch.save(replays_data, save_path)
    print(
        f"[worker] Saved routing replay ({total_entries} entries, {len(replays_data)} replays) → {save_path}",
        flush=True,
    )


def _replay_file_path(*, base_dir: Path) -> Path:
    return base_dir / f"rank0_{routing_replay_manager.filename}"


def _get_parallel_ranks() -> tuple[int, int, int, int]:
    """Return (cp_size, cp_rank, tp_size, tp_rank), defaulting to 1/0 if mpu is not initialized."""
    from megatron.core import mpu

    initialized: bool = mpu.is_initialized()
    cp_size: int = mpu.get_context_parallel_world_size() if initialized else 1
    cp_rank: int = mpu.get_context_parallel_rank() if initialized else 0
    tp_size: int = mpu.get_tensor_model_parallel_world_size() if initialized else 1
    tp_rank: int = mpu.get_tensor_model_parallel_rank() if initialized else 0
    return cp_size, cp_rank, tp_size, tp_rank


def _load_replay(
    replay_file: Path,
    *,
    rank: int,
    sequence_parallel: bool,
) -> None:
    """Load replay from rank 0's file with CP zigzag slicing and SP slicing."""
    saved_replays: list[list[torch.Tensor]] = torch.load(replay_file, weights_only=False)

    expected: int = len(routing_replay_manager.replays)
    if len(saved_replays) != expected:
        raise ValueError(f"Replay file has {len(saved_replays)} replays but model expects {expected}")

    cp_size, cp_rank, tp_size, tp_rank = _get_parallel_ranks()
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
            f"[worker] Loaded routing replay ({total_entries} entries, {expected} replays) ← {replay_file}",
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
