from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from miles.utils.ft.models import Decision, TriggerType


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
class DiagnosticSchedulerProtocol(Protocol):
    async def run_diagnostic_pipeline(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
    ) -> Decision: ...
