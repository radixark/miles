import logging
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, NamedTuple

import torch

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.backends.training_utils.weight_update.protocols.p2p_cell_executor import _CellWriteExecutor
from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import RemoteWeightInfo

logger = logging.getLogger(__name__)


class P2PInferenceCellUpdater:
    def __init__(
        self,
        cell_id: str,
        transfer_engine: Any,
        health: InferenceCellHealth,
        targets_by_engine_rank: dict[int, RemoteWeightInfo],
        transfer_timeout: float,
    ) -> None:
        self.cell_id = cell_id
        self._transfer_engine = transfer_engine
        self._health = health
        self._transfer_timeout = transfer_timeout
        self._executor = _CellWriteExecutor(cell_id)
        self._disposed = False
        self._target_by_engine_rank = targets_by_engine_rank
        self._pending_writes: list[Future] = []

    @property
    def is_errored(self) -> bool:
        return self._health.is_errored(self.cell_id)

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    @property
    def accepts_writes(self) -> bool:
        return not self._disposed and not self.is_errored

    def mark_errored(self, error: BaseException) -> None:
        self._health.mark_errored(self.cell_id, error)

    def submit_write(
        self, engine_rank: int, names: list[str], weight_memory_registry: dict[str, tuple[int, int, int]]
    ) -> Future | None:
        if not self.accepts_writes:
            return None
        future = self._executor.submit(
            self._write_if_active,
            self._target_by_engine_rank[engine_rank],
            names,
            weight_memory_registry,
        )
        self._pending_writes.append(future)
        return future

    def wait_for_write(self, future: Future | None) -> None:
        if future is None:
            return
        self._collect_write(future)

    def wait_for_pending_writes(self) -> None:
        for future in list(self._pending_writes):
            self._collect_write(future)

    def take_unfinished_writes(self) -> list[Future]:
        unfinished, self._pending_writes = self._pending_writes, []
        return unfinished

    def dispose(self) -> _CellWriteExecutor | None:
        self._disposed = True
        return None if self._executor.close() else self._executor

    def _collect_write(self, future: Future) -> None:
        try:
            future.result(timeout=0.0 if self.is_errored else self._transfer_timeout)
        except FutureTimeoutError as error:
            self._abandon_write(future, error)
            return
        except CancelledError:
            self._forget_write(future)
            return
        except Exception as error:
            logger.exception(f"[P2P-Shared] a write to cell {self.cell_id} failed")
            self.mark_errored(error)
        self._forget_write(future)

    def _abandon_write(self, future: Future, error: BaseException) -> None:
        self.mark_errored(error)
        if future.cancel():
            self._forget_write(future)
            return
        logger.error(f"[P2P-Shared] a write to cell {self.cell_id} is still running after the transfer timeout")

    def _forget_write(self, future: Future) -> None:
        if future in self._pending_writes:
            self._pending_writes.remove(future)

    def _write_if_active(
        self,
        target: RemoteWeightInfo,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
    ) -> None:
        if not self.accepts_writes:
            logger.warning(f"[P2P-Shared] skipping a queued write to cell {self.cell_id}")
            return
        _write_one_target(self._transfer_engine, target, names, weight_memory_registry)


class TransferEngineMeta(NamedTuple):
    engine_rank: int
    model_replica: torch.nn.Module
    cell_updaters: list[P2PInferenceCellUpdater]


def _write_one_target(
    transfer_engine: Any,
    target: RemoteWeightInfo,
    names: list[str],
    weight_memory_registry: dict[str, tuple[int, int, int]],
) -> None:
    """P2P write from shared CPU pinned buffers to a single remote session.

    Used by the parallelized submission path where each session within an
    engine rank is submitted as a separate task to this cell's write executor.
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

    session_id = target.session_id
    target_ptrs = []
    for name, source_len in zip(valid_names, source_lens, strict=True):
        if name in target.weights_info:
            location = target.weights_info[name]
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
