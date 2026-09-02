"""Engine-side RPCs for a weight-update session.

The session frame is: pause -> begin -> (transfer) -> set version -> end ->
resume. Callers gate driver-only calls (typically global rank 0) themselves.
"""

from argparse import Namespace
from collections.abc import Mapping, Sequence

import ray
from ray.actor import ActorHandle


def pause_engines(args: Namespace, rollout_engines: Sequence[ActorHandle]) -> None:
    """Quiesce the engines for a weight write.

    in_place pausing freezes requests and resumes them against their existing
    KV cache, so flushing would discard exactly what that mode preserves.
    """
    mode = args.pause_generation_mode
    ray.get([engine.pause_generation.remote(mode=mode) for engine in rollout_engines])
    if mode != "in_place":
        ray.get([engine.flush_cache.remote() for engine in rollout_engines])


def resume_engines(rollout_engines: Sequence[ActorHandle]) -> None:
    ray.get([engine.continue_generation.remote() for engine in rollout_engines])


def begin_weight_update(rollout_engines: Sequence[ActorHandle], selector: str = "all") -> None:
    """Open a weight-update session on the selected engines (restores packed weights)."""
    ray.get([engine.begin_weight_update.remote(selector=selector) for engine in rollout_engines])


def end_weight_update(rollout_engines: Sequence[ActorHandle]) -> None:
    """Close the session (post-load + quantization post-process on the full model)."""
    ray.get([engine.end_weight_update.remote() for engine in rollout_engines])


def set_weight_version(rollout_engines: Sequence[ActorHandle], weight_version: int) -> None:
    ray.get([engine.update_weight_version.remote(weight_version=str(weight_version)) for engine in rollout_engines])


def unload_lora_adapter(rollout_engines: Sequence[ActorHandle], lora_name: str) -> None:
    ray.get([engine.unload_lora_adapter.remote(lora_name=lora_name) for engine in rollout_engines])


def check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    """Raise if any engine reported a failed weight-sync RPC."""
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
