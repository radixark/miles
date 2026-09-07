import logging
from concurrent.futures import Future
from typing import Any

from .p2p_transfer_utils import P2PTransferManager, RemoteWeightInfo

logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater:
    def __init__(
        self,
        cell_id: str,
        transfer_engine: Any,
        transfer_manager: P2PTransferManager,
        targets_by_rollout_engine_rank: dict[int, RemoteWeightInfo],
    ) -> None:
        self.cell_id = cell_id
        self.error: BaseException | None = None
        self._transfer_engine = transfer_engine
        self._transfer_manager = transfer_manager
        self._disposed = False
        self._target_by_rollout_engine_rank = targets_by_rollout_engine_rank
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
