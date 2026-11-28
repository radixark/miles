"""Engine-side RPCs for a weight-update session.

The session frame is: pause -> begin -> (transfer) -> set version -> end ->
resume. Callers gate driver-only calls (typically global rank 0) themselves.

With --p2p-staged-update the engines are never paused: the transfer lands in
their staging region while they generate, and `end` installs it together with
the version, so the frame is begin -> (transfer) -> end.
"""

from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater
from miles.utils import async_utils
from miles.utils.arguments import driver_owns_generation_pause


def maybe_pause_engines(args: Namespace, cell_updaters: Sequence[_RolloutCellUpdater]) -> None:
    """Quiesce the engines for a weight write.

    in_place pausing freezes requests and resumes them against their existing
    KV cache, so flushing would discard exactly what that mode preserves.
    """
    if driver_owns_generation_pause(args) or args.p2p_staged_update:
        return
    mode = args.pause_generation_mode
    async_utils.wait_futures([updater.submit_client_call("pause_generation", mode=mode) for updater in cell_updaters])
    if mode != "in_place":
        async_utils.wait_futures([updater.submit_client_call("flush_cache") for updater in cell_updaters])


def maybe_resume_engines(args: Namespace, cell_updaters: Sequence[_RolloutCellUpdater]) -> None:
    if driver_owns_generation_pause(args) or args.p2p_staged_update:
        return
    async_utils.wait_futures([updater.submit_client_call("continue_generation") for updater in cell_updaters])


def weight_update_selector(*, speculative_algorithm: str | None, enable_mtp_training: bool) -> str:
    """Engine runners a weight sync covers. A speculative draft is synced only when the
    trainer trains its MTP layers (--enable-mtp-training); any other draft keeps its startup weights."""
    if speculative_algorithm and not enable_mtp_training:
        return "target"
    return "all"


def begin_weight_update(
    cell_updaters: Sequence[_RolloutCellUpdater],
    selector: str = "all",
    *,
    sync_base: bool = True,
    use_staging: bool = False,
) -> None:
    """Open a weight-update session on the selected engines. ``sync_base=False``
    declares an adapter-only session: no quant unpack, base tensors rejected.
    ``use_staging=True`` declares that the transfer writes the staging region."""
    results = async_utils.wait_futures(
        [
            updater.submit_client_call(
                "begin_weight_update", selector=selector, sync_base=sync_base, use_staging=use_staging
            )
            for updater in cell_updaters
        ]
    )
    if use_staging:
        # an engine that refused keeps generating: nothing may be written to it
        _mark_errored_on_failure(cell_updaters, results, "begin_weight_update(use_staging=True)")


def end_weight_update(
    cell_updaters: Sequence[_RolloutCellUpdater],
    *,
    expected_lora_checksums: Mapping | None = None,
    weight_version: int | None = None,
) -> None:
    """Close the session: re-finalize base weights (sync_base sessions) and apply
    the streamed LoRA stash (optionally verified against a sha256 manifest).
    ``weight_version`` is for a staged session, whose weights this call installs."""
    results = async_utils.wait_futures(
        [
            updater.submit_client_call(
                "end_weight_update",
                expected_lora_checksums=expected_lora_checksums,
                weight_version=None if weight_version is None else str(weight_version),
            )
            for updater in cell_updaters
        ]
    )
    _mark_errored_on_failure(cell_updaters, results, "end_weight_update")


def _mark_errored_on_failure(cell_updaters: Sequence[_RolloutCellUpdater], results: Sequence, call: str) -> None:
    for updater, result in zip(cell_updaters, results, strict=True):
        if isinstance(result, Mapping) and result.get("success") is False:
            updater.mark_errored(RuntimeError(f"{call} failed on a rollout engine: {result.get('message')}"))


def register_lora_adapter(
    rollout_engines: Sequence[SGLangApiClient], *, lora_name: str, lora_config: Mapping, pinned: bool = False
) -> None:
    """Create-or-refresh an adapter's identity and config on every engine
    (weights zeroed; the bytes follow in the update stream)."""
    futures = [
        async_utils.submit(
            client.register_lora_adapter(lora_name=lora_name, config_dict=dict(lora_config), pinned=pinned)
        )
        for client in rollout_engines
    ]
    check_weight_sync_results(async_utils.wait_futures(futures), is_lora=True)


def set_weight_version(cell_updaters: Sequence[_RolloutCellUpdater], weight_version: int) -> None:
    async_utils.wait_futures(
        [
            updater.submit_client_call("update_weight_version", weight_version=str(weight_version))
            for updater in cell_updaters
        ]
    )


def check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    """Raise if any engine reported a failed weight-sync RPC."""
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(
            result, "success"
        ):  # config-access-exempt: weight-update transports return different result schemas
            success = result.success
            error_msg = getattr(
                result, "error_message", "unknown error"
            )  # config-access-exempt: weight-update transports return different result schemas
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
