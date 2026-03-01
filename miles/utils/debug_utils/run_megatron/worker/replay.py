"""Routing replay stage management for standalone Megatron worker."""

from pathlib import Path

import torch

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.replay_base import routing_replay_manager


def load_replay_data(script: WorkerScriptArgs, *, rank: int) -> None:
    """Load routing replay data from disk before forward pass."""
    if not script.routing_replay_load_path:
        return

    replay_file: Path = _replay_file_path(base_dir=script.routing_replay_load_path, rank=rank)

    if not replay_file.exists():
        print(f"[worker rank={rank}] WARNING: replay file not found: {replay_file}", flush=True)
        return

    data: list[torch.Tensor] = torch.load(replay_file, weights_only=False)
    idx: int = 0
    for replay in routing_replay_manager.replays:
        chunk_size: int = len(replay.data) if replay.data else 1
        replay.data = data[idx : idx + chunk_size]
        idx += chunk_size

    if rank == 0:
        print(f"[worker] Loaded routing replay ({len(data)} entries) ← {replay_file}", flush=True)


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

    all_data: list[torch.Tensor] = []
    for replay in routing_replay_manager.replays:
        all_data.extend(replay.data)

    if all_data:
        save_path: Path = _replay_file_path(base_dir=script.routing_replay_dump_path, rank=rank)
        torch.save(all_data, save_path)
        if rank == 0:
            print(f"[worker] Saved routing replay ({len(all_data)} entries) → {save_path}", flush=True)


def _replay_file_path(*, base_dir: Path, rank: int) -> Path:
    return base_dir / f"rank{rank}_{routing_replay_manager.filename}"
