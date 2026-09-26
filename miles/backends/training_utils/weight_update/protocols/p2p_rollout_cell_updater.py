from concurrent.futures import Future
from typing import Any

from .p2p_transfer_utils import P2PTransferManager, RemoteWeightInfo


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater:
    def __init__(
        self,
        rollout_engine_ind: int,
    ) -> None:
        self.rollout_engine_ind = rollout_engine_ind
        self.targets_by_rollout_engine_rank: dict[int, RemoteWeightInfo] = {}
        self._pending_writes: list[Future[None]] = []

    def submit_write(
        self,
        rollout_engine_rank: int,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        transfer_engine: Any,
        transfer_manager: P2PTransferManager,
    ) -> None:
        self._pending_writes.append(
            transfer_manager.submit(
                _do_p2p_write_one_session,
                transfer_engine,
                self.targets_by_rollout_engine_rank[rollout_engine_rank],
                names,
                weight_memory_registry,
            )
        )

    def wait_for_pending_writes(self) -> None:
        pending, self._pending_writes = self._pending_writes, []
        for future in pending:
            future.result()


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
    for name in valid_names:
        if name in remote_session.weights_info:
            target_ptrs.append(remote_session.weights_info[name].address)

    assert len(target_ptrs) == len(source_ptrs), (
        f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
        f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
    )

    ret = transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
    if ret < 0:
        raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")
