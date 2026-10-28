import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, ClassVar, Protocol

import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    assert_tensors_fit,
    query_remote_shards,
)
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


class NativeTransport(Protocol):
    """What one transport library, such as Mooncake or NIXL, supplies to `P2PTransport`."""

    # the --p2p-transport value, and the transport sglang tags its published transfer engine info with
    name: ClassVar[str]
    transfer_buffer_device: ClassVar[torch.device]

    def connect_peer(self, transfer_engine_info: dict) -> str:
        """Returns the transport peer of the runner that published `transfer_engine_info`."""

    def remove_peers(self) -> None:
        """Forgets every peer `connect_peer` returned."""

    def register_tensor(self, tensor: torch.Tensor) -> Any:
        """Registers the tensor's memory with the library; returns what `unregister_tensor` takes."""

    def unregister_tensor(self, registration: Any) -> None:
        """Raises while a receiver may still read the memory, such as after a write that failed with unknown
        completion."""

    def write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        """Writes each tensor into its weight on the shard; returns once the write completes, raises if it fails."""


class P2PTransport:
    """Writes tensors from this trainer process's registered memory into rollout engines over a `NativeTransport`.

    The p2p protocol keeps one for the whole trainer process: it registers its transfer buffers once, calls `connect`
    with each new set of rollout engines, then `write` for each rollout engine rank. Each rollout engine has its own
    write thread, so a stuck engine holds up only its own writes. The native transport decides when registered memory
    may be released, since only it knows whether a failed write can still read its source.
    """

    def __init__(self, native_transport: NativeTransport) -> None:
        self._native_transport = native_transport
        self._executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}
        # the registered tensor stays referenced with its registration, so its allocation outlives it
        self._registrations_by_address: dict[int, tuple[torch.Tensor, Any]] = {}

    @property
    def transfer_buffer_device(self) -> torch.device:
        return self._native_transport.transfer_buffer_device

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
        for executor in self._executors_by_rollout_engine_ind.values():
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors_by_rollout_engine_ind = {}
        self._native_transport.remove_peers()
        return query_remote_shards(
            cell_updaters_of_cell_id,
            engine_cell_ids,
            assignments,
            runner_roles,
            self._native_transport.name,
            self._native_transport.connect_peer,
        )

    def register_memory(self, tensor: torch.Tensor) -> None:
        self._registrations_by_address[tensor.data_ptr()] = (tensor, self._native_transport.register_tensor(tensor))

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        _, registration = self._registrations_by_address[tensor.data_ptr()]
        self._native_transport.unregister_tensor(registration)
        del self._registrations_by_address[tensor.data_ptr()]

    def write(
        self,
        remote_shards: Sequence[RemoteShard],
        tensors_by_name: dict[str, torch.Tensor],
        *,
        run_write: Callable[[RemoteShard, Callable[[], None]], None],
    ) -> list[Future[None]]:
        """Writes each tensor into the weight of the same name on every shard, through `run_write` on the shard's
        rollout engine thread; returns one future per shard.

        Allocations must stay registered while native writes may still read them. Healthy receivers require
        unchanged bytes until their futures finish. Raises before writing anything if a shard lacks one of the
        weights or holds it in a different number of bytes.
        """
        for remote_shard in remote_shards:
            assert_tensors_fit(remote_shard, tensors_by_name)
        context = fault_hook_controller.current_context()
        return [
            self._get_or_create_executor(remote_shard.rollout_engine_ind).submit(
                run_write, remote_shard, partial(self._write_shard, remote_shard, tensors_by_name, context=context)
            )
            for remote_shard in remote_shards
        ]

    def _write_shard(
        self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor], *, context: FaultHookContext
    ) -> None:
        if not tensors_by_name:
            return
        reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
        started_at = datetime.now(timezone.utc)
        try:
            self._native_transport.write_shard(remote_shard, tensors_by_name)
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
                    logger.exception("Could not record P2P transport failure for %s", remote_shard.transport_peer)
            logger.exception("P2P transport write failed for %s", remote_shard.transport_peer)
            raise

    def _get_or_create_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._executors_by_rollout_engine_ind:
            self._executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{self._native_transport.name}-write-rollout-engine-{rollout_engine_ind}",
            )
        return self._executors_by_rollout_engine_ind[rollout_engine_ind]
