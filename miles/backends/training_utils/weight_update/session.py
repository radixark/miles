"""Own the trainer side of the engine weight-update session."""

import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


class EngineWeightUpdateSession:
    """Broadcast engine RPC outcomes to every training rank.

    Staged sessions skip pausing and abort on failure; successful scopes must call `commit`.
    """

    def __init__(
        self,
        protocol,
        args: Namespace,
        *,
        staged: bool,
        sync_base: bool,
        selector: str,
        registrations: Sequence[tuple[str, Mapping]] = (),
        lora_path: str | None = None,
    ) -> None:
        self._protocol = protocol
        self._args = args
        self._staged = staged
        self._sync_base = sync_base
        self._selector = selector
        self._registrations = registrations
        self._lora_path = lora_path
        self._committed = False

    def __enter__(self) -> "EngineWeightUpdateSession":
        try:
            self._rpcs_from_rank0(self._open)
        except Exception:
            # __exit__ never runs when __enter__ raises: _open may have paused
            # engines or staged a registration before failing partway
            self._discard_open()
            raise
        return self

    def commit(self, expected_lora_checksums: Mapping | None, weight_version: int | None) -> None:
        self._rpcs_from_rank0(lambda: self._close(expected_lora_checksums, weight_version))
        self._committed = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            assert self._committed, "the session scope exited without commit()"
            return
        if self._staged and dist.get_rank() == 0:
            try:
                end_weight_update(self._protocol.rollout_engines, abort=True)
            except Exception:
                # the abort usually shares the failure's root cause; it must not mask it
                logger.exception("Failed to discard the staged adapter session")

    def _discard_open(self) -> None:
        """Best-effort rollback of _open's engine state: abort a staged session
        (also dropping any pending publication) or resume paused engines. Safe
        only before any weight bytes moved — a failed open, not a failed stream."""
        if not (self._protocol.use_weight_update_session and dist.get_rank() == 0):
            return
        try:
            if self._staged:
                end_weight_update(self._protocol.rollout_engines, abort=True)
            else:
                resume_engines(self._protocol.rollout_engines)
        except Exception:
            # the cleanup usually shares the failure's root cause; it must not mask it
            logger.exception("Failed to roll back the engines after a failed session open")

    def _open(self) -> None:
        engines = self._protocol.rollout_engines
        if not self._staged:
            pause_engines(self._args, engines)
        # eager registration: the engine validates the rank before any bytes move
        for lora_name, lora_config in self._registrations:
            register_lora_adapter(
                engines,
                lora_name=lora_name,
                lora_config=lora_config,
                lora_path=self._lora_path,
                defer_publish=self._staged,
            )
        begin_weight_update(engines, self._selector, sync_base=self._sync_base)

    def _close(self, checksums: Mapping | None, weight_version: int | None) -> None:
        engines = self._protocol.rollout_engines
        end_weight_update(engines, expected_lora_checksums=checksums)
        if weight_version is not None:
            set_weight_version(engines, weight_version)
        if not self._staged:
            resume_engines(engines)

    def _rpcs_from_rank0(self, rpcs: Callable[[], None]) -> None:
        failure = [None]
        if self._protocol.use_weight_update_session and dist.get_rank() == 0:
            try:
                rpcs()
            except Exception as exc:
                logger.exception("engine weight-update RPCs failed")
                failure[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(failure, src=0, group=get_gloo_group())
        if failure[0] is not None:
            raise RuntimeError(f"engine weight-update RPCs failed: {failure[0]}")


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
    stash after verifying ``expected_lora_checksums``; ``abort`` discards both."""
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
