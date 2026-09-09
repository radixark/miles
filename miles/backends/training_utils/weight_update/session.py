"""Engine-side RPCs for a weight-update session.

The session frame is: pause -> begin -> (transfer) -> set version -> end ->
resume. Callers gate driver-only calls (typically global rank 0) themselves.
"""

from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils


def pause_engines(args: Namespace, rollout_engines: Sequence[SGLangApiClient]) -> None:
    """Quiesce the engines for a weight write.

    in_place pausing freezes requests and resumes them against their existing
    KV cache, so flushing would discard exactly what that mode preserves.
    """
    mode = args.pause_generation_mode
    async_utils.wait_futures([async_utils.submit(client.pause_generation(mode=mode)) for client in rollout_engines])
    if mode != "in_place":
        async_utils.wait_futures([async_utils.submit(client.flush_cache()) for client in rollout_engines])


def resume_engines(rollout_engines: Sequence[SGLangApiClient]) -> None:
    async_utils.wait_futures([async_utils.submit(client.continue_generation()) for client in rollout_engines])


def begin_weight_update(
    rollout_engines: Sequence[SGLangApiClient], selector: str = "all", *, sync_base: bool = True
) -> None:
    """Open a weight-update session on the selected engines. ``sync_base=False``
    declares an adapter-only session: no quant unpack, base tensors rejected."""
    async_utils.wait_futures(
        [
            async_utils.submit(client.begin_weight_update(selector=selector, sync_base=sync_base))
            for client in rollout_engines
        ]
    )


def end_weight_update(
    rollout_engines: Sequence[SGLangApiClient],
    *,
    expected_lora_checksums: Mapping | None = None,
    abort: bool = False,
) -> None:
    """Close the session: finalize base weights and apply the streamed LoRA
    stash under the manifest; ``abort`` discards both instead."""
    results = async_utils.wait_futures(
        [
            async_utils.submit(client.end_weight_update(expected_lora_checksums=expected_lora_checksums, abort=abort))
            for client in rollout_engines
        ]
    )
    for result in results:
        if isinstance(result, Mapping) and result.get("success") is False:
            raise RuntimeError(f"end_weight_update failed on a rollout engine: {result.get('message')}")


def register_lora_adapter(
    rollout_engines: Sequence[SGLangApiClient],
    *,
    lora_name: str,
    lora_config: Mapping,
    pinned: bool = False,
    lora_path: str | None = None,
    defer_publish: bool = False,
) -> None:
    """Create-or-refresh an adapter's identity on every engine; the bytes follow
    in the update stream. ``defer_publish`` keeps the name unservable until the
    session commits; ``lora_path`` makes it evictable (refill from disk)."""
    futures = [
        async_utils.submit(
            client.register_lora_adapter(
                lora_name=lora_name,
                config_dict=dict(lora_config),
                pinned=pinned,
                lora_path=lora_path,
                defer_publish=defer_publish,
            )
        )
        for client in rollout_engines
    ]
    results = async_utils.wait_futures(futures)
    check_weight_sync_results(results, is_lora=True)
    if defer_publish and any(not isinstance(result, Mapping) or not result.get("pending") for result in results):
        # an engine that ignored defer_publish would serve the name while its weights stream
        raise RuntimeError("the rollout engines must support deferred LoRA publication")


def set_weight_version(rollout_engines: Sequence[SGLangApiClient], weight_version: int) -> None:
    async_utils.wait_futures(
        [
            async_utils.submit(client.update_weight_version(weight_version=str(weight_version)))
            for client in rollout_engines
        ]
    )


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
