import logging
from argparse import Namespace
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Future
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils

from .p2p_transfer_utils import P2PTransferManager, RemoteWeightInfo

logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater:
    def __init__(
        self,
        args: Namespace,
        cell_id: str,
        api_client: SGLangApiClient,
        transfer_engine: Any,
        transfer_manager: P2PTransferManager,
        targets_by_rollout_engine_rank: dict[int, RemoteWeightInfo],
    ) -> None:
        self.cell_id = cell_id
        self.error: BaseException | None = None
        self._args = args
        self._api_client = api_client
        self._transfer_engine = transfer_engine
        self._transfer_manager = transfer_manager
        self._disposed = False
        self._target_by_rollout_engine_rank = targets_by_rollout_engine_rank
        self._pending_op = ""
        self._pending_writes: list[Future[None]] = []

    @property
    def is_errored(self) -> bool:
        return self.error is not None

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    @property
    def accepts_writes(self) -> bool:
        return not self._disposed and not self.is_errored

    def submit_pause(self) -> Future[Any] | None:
        mode = self._args.pause_generation_mode
        return self._submit("pause_generation", lambda client: client.pause_generation(mode=mode))

    def submit_flush_cache(self) -> Future[Any] | None:
        if self._args.pause_generation_mode == "in_place":
            return None
        return self._submit("flush_cache", lambda client: client.flush_cache())

    def submit_begin(self, *, selector: str, sync_base: bool) -> Future[Any] | None:
        return self._submit(
            "begin_weight_update",
            lambda client: client.begin_weight_update(selector=selector, sync_base=sync_base),
        )

    def submit_end(self) -> Future[Any] | None:
        return self._submit("end_weight_update", lambda client: client.end_weight_update())

    def submit_set_weight_version(self, weight_version: int) -> Future[Any] | None:
        return self._submit(
            "update_weight_version",
            lambda client: client.update_weight_version(weight_version=str(weight_version)),
        )

    def submit_resume(self) -> Future[Any] | None:
        return self._submit("continue_generation", lambda client: client.continue_generation())

    def collect(self, future: Future[Any]) -> None:
        op = self._pending_op
        try:
            _raise_if_unsuccessful(op, future.result())
        except Exception as error:
            logger.exception(f"[weight-update] {op} failed on inference cell {self.cell_id}")
            self.mark_errored(error)

    def mark_errored(self, error: BaseException) -> None:
        if self.error is not None:
            logger.warning(f"inference cell {self.cell_id} failed again, keeping the first error", exc_info=error)
            return
        self.error = error
        logger.error(f"inference cell {self.cell_id} can no longer be updated", exc_info=error)

    def dispose(self) -> None:
        self._disposed = True

    def submit_write(
        self, rollout_engine_rank: int, names: list[str], weight_memory_registry: dict[str, tuple[int, int, int]]
    ) -> None:
        if not self.accepts_writes:
            return
        self._pending_writes.append(
            self._transfer_manager.submit(
                self._write_if_active,
                self._target_by_rollout_engine_rank[rollout_engine_rank],
                names,
                weight_memory_registry,
            )
        )

    def wait_for_pending_writes(self) -> None:
        pending, self._pending_writes = self._pending_writes, []
        for future in pending:
            future.result()

    def _submit(
        self, op: str, make_request: Callable[[SGLangApiClient], Coroutine[Any, Any, Any]]
    ) -> Future[Any] | None:
        if self.is_errored:
            return None
        self._pending_op = op
        return async_utils.submit(make_request(self._api_client))

    def _write_if_active(
        self,
        target: RemoteWeightInfo,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
    ) -> None:
        if not self.accepts_writes:
            logger.warning(f"[P2P-Shared] skipping a queued write to cell {self.cell_id}")
            return
        _do_p2p_write_one_session(self._transfer_engine, target, names, weight_memory_registry)


def _raise_if_unsuccessful(op: str, result: object) -> None:
    if not isinstance(result, Mapping) or result.get("success") is not False:
        return

    message = result.get("error_message") or result.get("error") or result.get("message") or "unknown error"
    raise RuntimeError(f"{op} was rejected by the rollout engine: {message}")


def _do_p2p_write_one_session(
    transfer_engine: Any,
    remote_session: RemoteWeightInfo,
    names: list[str],
    weight_memory_registry: dict[str, tuple[int, int, int]],
) -> None:
    """P2P write from shared CPU pinned buffers to a single remote session.

    Used by the parallelized submission path where each session within an
    rollout engine rank is submitted as a separate task to P2PTransferManager.
    """
    source_ptrs, source_lens = [], []
    valid_names = []

    for name in names:
        cpu_reg = weight_memory_registry.get(name)
        assert cpu_reg, f"the _weight_memory_registry of {name} failed"

        data_ptr, numel, ele_size = cpu_reg
        source_ptrs.append(data_ptr)
        source_lens.append(numel * ele_size)
        valid_names.append(name)

    if not source_ptrs:
        return

    session_id = remote_session.session_id
    target_ptrs = []
    for name, source_len in zip(valid_names, source_lens, strict=True):
        if name in remote_session.weights_info:
            location = remote_session.weights_info[name]
            target_len = location.numel * location.element_size
            assert target_len == source_len, (
                f"[P2P-Shared] {name} spans {source_len} bytes here and {target_len} bytes on session "
                f"{session_id}, so writing it would run past the target buffer"
            )
            target_ptrs.append(location.address)

    assert len(target_ptrs) == len(source_ptrs), (
        f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
        f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
    )

    ret = transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
    if ret < 0:
        raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")
