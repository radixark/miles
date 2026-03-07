from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from miles.utils.ft.models.diagnostic import DiagnosticPipelineResult
    from miles.utils.ft.models.fault import TriggerType


class JobStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"


@runtime_checkable
class NodeManagerProtocol(Protocol):
    async def mark_node_bad(self, node_id: str, reason: str) -> None: ...

    async def unmark_node_bad(self, node_id: str) -> None: ...

    async def get_bad_nodes(self) -> list[str]: ...


@runtime_checkable
class TrainingJobProtocol(Protocol):
    async def stop_training(self, timeout_seconds: int = 300) -> None: ...

    async def submit_training(
        self, excluded_node_ids: list[str] | None = None,
    ) -> str: ...

    async def get_training_status(self) -> JobStatus: ...


@runtime_checkable
class NotificationProtocol(Protocol):
    async def send(self, title: str, content: str, severity: str) -> None: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class DiagnosticOrchestratorProtocol(Protocol):
    async def run_diagnostic_pipeline(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> DiagnosticPipelineResult: ...


@runtime_checkable
class ControllerClientProtocol(Protocol):
    """Agent-side interface for communicating with the FtController.

    Implementations hide the transport (Ray, in-process, stub) so that
    agent code never imports ray or calls .remote().
    """

    def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int,
        timeout: float = 10,
    ) -> None: ...

    def log_step(
        self, run_id: str, step: int, metrics: dict[str, float],
    ) -> None: ...


def ft_controller_actor_name(ft_id: str) -> str:
    if not ft_id:
        return "ft_controller"
    return f"ft_controller_{ft_id}"


def ft_node_agent_actor_name(ft_id: str, node_id: str) -> str:
    prefix = f"ft_node_agent_{ft_id}" if ft_id else "ft_node_agent"
    return f"{prefix}_{node_id}"
