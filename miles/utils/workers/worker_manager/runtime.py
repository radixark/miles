"""What the backend-neutral worker manager needs from the cluster backend that runs its workers.

The manager owns pools, cells, generations, ports and liveness; a runtime only knows how to start a cell's
processes on the gpus the manager picked for them, talk to them, and take them down again.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.backend_capability.base import BackendCapability


@dataclass(frozen=True, kw_only=True)
class GpuSlotRef:
    """A run of consecutive logical gpu slots of one named layout ("actor", "rollout", ...)."""

    layout_name: str
    first_slot_index: int
    num_slots: int


@dataclass(frozen=True, kw_only=True)
class WorkerLaunchRequest:
    pool_id: str
    name: str
    actor_class: type
    ctor_kwargs: dict[str, Any]
    env_vars: dict[str, str]
    num_cpus: float
    num_gpus: float
    gpu_slots: GpuSlotRef | None
    pin_to_head: bool
    # Only for a worker served natively: concurrency group -> threads.
    concurrency_groups: dict[str, int] | None = None


class WorkerProcess(abc.ABC):
    """One launched worker, whatever runs it."""

    @abc.abstractmethod
    async def call(self, method: str, /, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def call_without_result(self, method: str, /, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    async def probe_is_dead(self) -> bool: ...

    @property
    @abc.abstractmethod
    def native_handle(self) -> Any:
        """What a driver of this backend calls the worker through natively, e.g. a ray ActorHandle."""


class WorkerRuntime(abc.ABC):
    @abc.abstractmethod
    async def launch_cell(self, *, cell_id: str, launches: list[WorkerLaunchRequest]) -> list[WorkerProcess]:
        """Start every worker of one cell; the result is ordered like ``launches``."""

    @abc.abstractmethod
    async def kill_worker(self, process: WorkerProcess) -> None:
        """Take one worker down as soon as it is done shutting down, without waiting for its peers; never raises.

        A runtime whose workers can only stop together, as a whole cell, may leave it to ``release_cell``."""

    @abc.abstractmethod
    async def release_cell(self, *, processes: list[WorkerProcess]) -> None:
        """Free what the cell held beyond its workers, once every one of them is killed; never raises."""

    @abc.abstractmethod
    def physical_gpu_ids(self, slots: GpuSlotRef) -> list[int]:
        """The physical device index, on its node, of each slot of ``slots``."""

    @abc.abstractmethod
    def worker_capability_factory(self) -> Callable[[], BackendCapability]:
        """A picklable factory a launched worker calls to reach the run's backend capability."""
