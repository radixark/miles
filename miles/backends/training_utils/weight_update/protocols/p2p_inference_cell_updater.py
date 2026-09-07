import logging
from concurrent.futures import Future
from typing import Any, NamedTuple

import torch

from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import (
    P2PTransferManager,
    RemoteWeightInfo,
)

logger = logging.getLogger(__name__)


class P2PInferenceCellUpdater:
    def __init__(
        self,
        cell_id: str,
        transfer_engine: Any,
        transfer_manager: P2PTransferManager,
        targets_by_engine_rank: dict[int, RemoteWeightInfo],
    ) -> None:
        self.cell_id = cell_id
        self._transfer_engine = transfer_engine
        self._transfer_manager = transfer_manager
        self._target_by_engine_rank = targets_by_engine_rank

    def submit_write(
        self, engine_rank: int, names: list[str], weight_memory_registry: dict[str, tuple[int, int, int]]
    ) -> Future:
        return self._transfer_manager.submit(
            _write_one_target,
            self._transfer_engine,
            self._target_by_engine_rank[engine_rank],
            names,
            weight_memory_registry,
        )


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
    engine rank is submitted as a separate task to P2PTransferManager.
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
