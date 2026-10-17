import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, NamedTuple

import ray
import torch

from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import WeightTransferFailedEvent
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller, reach_fault_hook
from miles.utils.test_utils.fault_injector.models import FaultHookName

logger = logging.getLogger(__name__)


class RemoteWeightLocation(NamedTuple):
    address: int
    numel: int
    element_size: int


@dataclass(frozen=True)
class RemoteShard:
    """Mooncake session and published weight addresses for one engine rank's target or draft runner."""

    cell_id: str
    rollout_engine_ind: int
    rollout_engine_rank: int
    runner_role: str
    session_id: str
    weight_locations_by_name: dict[str, RemoteWeightLocation]

    @property
    def published_nbytes_by_name(self) -> dict[str, int]:
        return {
            name: location.numel * location.element_size for name, location in self.weight_locations_by_name.items()
        }


class MooncakeTransport:
    """Writes tensors from this trainer process's registered memory into rollout engines over Mooncake.

    The p2p protocol keeps one for the whole trainer process: it registers its transfer buffers once, calls
    `connect` with each new set of rollout engines, then `write` for each rollout engine rank. Each rollout engine
    has its own write thread, so a stuck engine holds up only its own writes.
    """

    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self) -> None:
        self._transfer_engine = _create_transfer_engine()
        self._write_executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}

    def connect(
        self,
        cell_updaters_of_cell_id: dict[str, _RolloutCellUpdater],
        engine_cell_ids: Sequence[str],
        assignments: Sequence[RolloutEngineRankAssignment],
        runner_roles: Sequence[str],
    ) -> dict[str, dict[int, list[RemoteShard]]]:
        """Return assigned shards by runner role and rollout rank.

        Starts new write threads: a rollout engine index now names a new engine, which must not queue behind the
        writes of the one it replaced.
        """
        for write_executor in self._write_executors_by_rollout_engine_ind.values():
            write_executor.shutdown(wait=False)
        self._write_executors_by_rollout_engine_ind = {}
        return {
            runner_role: {
                assignment.rollout_engine_rank: [
                    remote_shard
                    for rollout_engine_ind in assignment.rollout_engine_indices
                    if (
                        remote_shard := _query_remote_shard(
                            cell_updaters_of_cell_id[engine_cell_ids[rollout_engine_ind]],
                            rollout_engine_ind,
                            assignment.rollout_engine_rank,
                            runner_role,
                        )
                    )
                    is not None
                ]
                for assignment in assignments
            }
            for runner_role in runner_roles
        }

    def register_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.register_memory(tensor.data_ptr(), _nbytes(tensor))
        if ret != 0:
            raise RuntimeError(
                f"Mooncake could not register {_nbytes(tensor)} bytes at {tensor.data_ptr():#x}: error {ret}"
            )

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.unregister_memory(tensor.data_ptr())
        if ret != 0:
            raise RuntimeError(f"Mooncake could not unregister the memory at {tensor.data_ptr():#x}: error {ret}")

    def write(
        self,
        remote_shards: Sequence[RemoteShard],
        tensors_by_name: dict[str, torch.Tensor],
        *,
        run_write: Callable[[RemoteShard, Callable[[], None]], None],
    ) -> list[Future[None]]:
        """Writes each tensor into the weight of the same name on every shard; returns one future per shard.

        Allocations must stay registered while native writes may still read them. Healthy receivers require
        unchanged bytes until their futures finish. Raises before writing anything if a shard lacks one of the
        weights or holds it in a different number of bytes.
        """
        for remote_shard in remote_shards:
            _assert_tensors_fit(remote_shard, tensors_by_name)
        context = fault_hook_controller.current_context()
        return [
            self._write_executor(remote_shard.rollout_engine_ind).submit(
                run_write, remote_shard, partial(self._write_shard, remote_shard, tensors_by_name, context=context)
            )
            for remote_shard in remote_shards
        ]

    def _write_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._write_executors_by_rollout_engine_ind:
            self._write_executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mooncake-write-rollout-engine-{rollout_engine_ind}"
            )
        return self._write_executors_by_rollout_engine_ind[rollout_engine_ind]

    def _write_shard(
        self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor], *, context: FaultHookContext
    ) -> None:
        names = list(tensors_by_name)
        if not names:
            return
        reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
        started_at = datetime.now(timezone.utc)
        try:
            ret = self._transfer_engine.batch_transfer_sync_write(
                remote_shard.session_id,
                [tensors_by_name[name].data_ptr() for name in names],
                [remote_shard.weight_locations_by_name[name].address for name in names],
                [_nbytes(tensors_by_name[name]) for name in names],
            )
            if ret < 0:
                raise RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")
        except Exception as error:
            if is_event_logger_initialized() and context.debug_weight_update_id is not None:
                try:
                    get_event_logger().log(
                        WeightTransferFailedEvent,
                        dict(
                            debug_weight_update_id=context.debug_weight_update_id,
                            cell_id=remote_shard.cell_id,
                            workers_hash=context.snapshot_cell_id_to_hashes[remote_shard.cell_id],
                            started_at=started_at,
                            error=repr(error),
                        ),
                        include_context=False,
                    )
                except Exception:
                    logger.exception("Could not record P2P transport failure for session %s", remote_shard.session_id)
            logger.exception("P2P transport write failed for session %s", remote_shard.session_id)
            raise


def _query_remote_shard(
    cell_updater: _RolloutCellUpdater, rollout_engine_ind: int, rollout_engine_rank: int, runner_role: str
) -> RemoteShard | None:
    info = cell_updater.submit_client_call(
        "get_remote_instance_transfer_engine_info", rank=rollout_engine_rank, role=runner_role
    ).result()
    if info is None:
        return None
    session_id, weights_info = info
    assert (
        session_id is not None
    ), f"the {runner_role} of rollout engine {rollout_engine_ind} rank {rollout_engine_rank} has no Mooncake session"
    return RemoteShard(
        cell_id=cell_updater.cell_id,
        rollout_engine_ind=rollout_engine_ind,
        rollout_engine_rank=rollout_engine_rank,
        runner_role=runner_role,
        session_id=session_id,
        weight_locations_by_name={name: RemoteWeightLocation(*location) for name, location in weights_info.items()},
    )


def _create_transfer_engine() -> Any:
    # not in miles' requirements
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    transfer_engine.initialize(ray._private.services.get_node_ip_address(), "P2PHANDSHAKE", "rdma", "")
    return transfer_engine


def _assert_tensors_fit(remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
    target = (
        f"the {remote_shard.runner_role} of rollout engine {remote_shard.rollout_engine_ind} "
        f"rank {remote_shard.rollout_engine_rank}"
    )
    for name, tensor in tensors_by_name.items():
        assert name in remote_shard.weight_locations_by_name, f"{target} publishes no {name}"
        location = remote_shard.weight_locations_by_name[name]
        target_nbytes = location.numel * location.element_size
        assert target_nbytes == _nbytes(tensor), (
            f"{name} is {_nbytes(tensor)} bytes here but {target_nbytes} bytes on {target}; "
            "writing it would run past the target"
        )


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
