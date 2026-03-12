from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import NotifierProtocol
from miles.utils.ft.controller.state_machines.restart.models import (
    RestartContext,
    RestartState,
)
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel


class RecoveryState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class RealtimeChecksSt(RecoveryState):
    pre_identified_bad_nodes: tuple[str, ...] = ()


class EvictingAndRestartingSt(RecoveryState):
    restart: RestartState
    failed_next_state: RecoveryState


class StopTimeDiagnosticsSt(RecoveryState):
    pass


class NotifyHumansSt(RecoveryState):
    state_before: str
    reason: str = ""


class RecoveryDoneSt(RecoveryState):
    pass


class RecoveryContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # per-call
    trigger: TriggerType
    recovery_start_time: datetime

    # deps
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    restart_stepper: Callable[[RestartState, RestartContext], AsyncGenerator[RestartState, None]]
    restart_context: RestartContext
    notifier: NotifierProtocol | None
    timeout_seconds: int
    rank_pids_provider: Callable[[str], dict[int, int]] | None
