"""Cross-layer boundary contracts: Protocols and constants shared across layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from miles.utils.ft.agents.types import DiagnosticResult


# ---------------------------------------------------------------------------
# Agent metadata provider — agents collect and expose arbitrary metadata
# ---------------------------------------------------------------------------


class AgentMetadataProvider(ABC):
    @abstractmethod
    def get_metadata(self) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# protocols/agents.py — controller calls agents
# ---------------------------------------------------------------------------

DIAGNOSTIC_TIMEOUT_SECONDS: int = 120


class NodeAgentProtocol(ABC):
    @abstractmethod
    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult: ...


class NodeExecutorProtocol(ABC):
    diagnostic_type: str

    @abstractmethod
    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult: ...


class ClusterExecutorProtocol(ABC):
    """Strategy for executing one diagnostic step within the pipeline.

    Returns bad_node_ids (empty if all healthy).
    """

    @abstractmethod
    async def execute(
        self,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]: ...


# ---------------------------------------------------------------------------
# protocols/controller.py — agents call controller
# ---------------------------------------------------------------------------

REGISTER_TIMEOUT_SECONDS: float = 10


class ControllerClientProtocol(ABC):
    """Agent-side interface for communicating with the FtController.

    Implementations hide the transport (Ray, in-process, stub) so that
    agent code never imports ray or calls .remote().
    """

    @abstractmethod
    def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int,
        timeout_seconds: float = REGISTER_TIMEOUT_SECONDS,
    ) -> None: ...

    @abstractmethod
    def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
    ) -> None: ...


def ft_controller_actor_name(ft_id: str) -> str:
    if not ft_id:
        return "ft_controller"
    return f"ft_controller_{ft_id}"


def ft_node_agent_actor_name(ft_id: str, node_id: str) -> str:
    prefix = f"ft_node_agent_{ft_id}" if ft_id else "ft_node_agent"
    return f"{prefix}_{node_id}"


# ---------------------------------------------------------------------------
# protocols/platform.py — controller calls platform
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"


STOP_TRAINING_TIMEOUT_SECONDS: int = 300


class NodeManagerProtocol(ABC):
    @abstractmethod
    async def mark_node_bad(self, node_id: str, reason: str, node_metadata: dict[str, str] | None = None) -> None: ...

    @abstractmethod
    async def unmark_node_bad(self, node_id: str) -> None: ...

    @abstractmethod
    async def get_bad_nodes(self) -> list[str]: ...


class MainJobProtocol(ABC):
    @abstractmethod
    async def stop_job(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None: ...

    @abstractmethod
    async def submit_job(self) -> str: ...

    @abstractmethod
    async def get_job_status(self) -> JobStatus: ...


class NotifierProtocol(ABC):
    @abstractmethod
    async def send(self, title: str, content: str, severity: str) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...
