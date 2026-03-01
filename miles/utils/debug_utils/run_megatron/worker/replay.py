"""Routing replay stage management for standalone Megatron worker."""

from pathlib import Path
from typing import Any

import torch

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.replay_base import routing_replay_manager

_REPLAY_FORMAT_VERSION: int = 1


def load_replay_data(script: WorkerScriptArgs, *, rank: int) -> None:
    """Load routing replay data from disk before forward pass."""
    if not script.routing_replay_load_path:
        return

    replay_file: Path = _replay_file_path(base_dir=script.routing_replay_load_path, rank=rank)

    if not replay_file.exists():
        print(f"[worker rank={rank}] WARNING: replay file not found: {replay_file}", flush=True)
        return

    payload: dict[str, Any] = torch.load(replay_file, weights_only=False)

    saved_replays: list[list[torch.Tensor]] = payload["replays"]
    expected: int = len(routing_replay_manager.replays)
    if len(saved_replays) != expected:
        raise ValueError(
            f"Replay file has {len(saved_replays)} replays but model expects {expected}"
        )

    total_entries: int = 0
    for replay, data in zip(routing_replay_manager.replays, saved_replays):
        replay.top_indices_list = data
        total_entries += len(data)

    if rank == 0:
        print(
            f"[worker] Loaded routing replay ({total_entries} entries, {expected} replays) ← {replay_file}",
            flush=True,
        )


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

    replays_data: list[list[torch.Tensor]] = [
        replay.top_indices_list for replay in routing_replay_manager.replays
    ]
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


def _replay_file_path(*, base_dir: Path, rank: int) -> Path:
    return base_dir / f"rank{rank}_{routing_replay_manager.filename}"
