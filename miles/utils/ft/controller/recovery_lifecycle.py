from __future__ import annotations

import logging
import time
from collections.abc import Callable

from miles.utils.ft.controller.actions import PlatformDeps, handle_enter_recovery
from miles.utils.ft.controller.recovery_cooldown import RecoveryCooldown
from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator
from miles.utils.ft.models._fault import Decision
from miles.utils.ft.models._recovery import RecoveryPhase, RecoverySnapshot, _BAD_NODES_CONFIRMED_PHASES

logger = logging.getLogger(__name__)


class RecoveryLifecycleManager:
    """Owns the full recovery lifecycle: cooldown gating, orchestrator creation,
    stepping, completion detection, and cleanup.
    """

    def __init__(
        self,
        cooldown: RecoveryCooldown,
        on_recovery_duration: Callable[[float], None] | None = None,
    ) -> None:
        self._cooldown = cooldown
        self._on_recovery_duration = on_recovery_duration

        self._orchestrator: RecoveryOrchestrator | None = None
        self._start_time: float | None = None
        self._diagnosing_nodes: set[str] = set()
        self._last_phase_history: list[RecoveryPhase] | None = None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def in_progress(self) -> bool:
        return self._orchestrator is not None

    @property
    def orchestrator(self) -> RecoveryOrchestrator | None:
        return self._orchestrator

    @property
    def phase(self) -> RecoveryPhase | None:
        if self._orchestrator is None:
            return None
        return self._orchestrator.phase

    @property
    def bad_node_ids(self) -> list[str]:
        if self._orchestrator is None:
            return []
        return self._orchestrator.bad_node_ids

    @property
    def diagnosing_nodes(self) -> set[str]:
        return set(self._diagnosing_nodes)

    @property
    def last_phase_history(self) -> list[RecoveryPhase] | None:
        return self._last_phase_history

    @property
    def cooldown(self) -> RecoveryCooldown:
        return self._cooldown

    def snapshot(self) -> RecoverySnapshot:
        if self._orchestrator is not None:
            phase = self._orchestrator.phase
            phase_history = list(self._orchestrator.phase_history)
            bad_nodes_confirmed = phase in _BAD_NODES_CONFIRMED_PHASES
        else:
            phase = None
            phase_history = self._last_phase_history
            bad_nodes_confirmed = False

        return RecoverySnapshot(
            in_progress=self.in_progress,
            phase=phase,
            phase_history=phase_history,
            diagnosing_nodes=sorted(self._diagnosing_nodes),
            bad_nodes_confirmed=bad_nodes_confirmed,
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def start(self, decision: Decision, deps: PlatformDeps) -> bool:
        """Attempt to start a recovery cycle.

        Records the trigger in the cooldown tracker.  Returns False (without
        creating an orchestrator) if the cooldown limit has been reached.
        """
        self._cooldown.record(decision.trigger)

        if self._cooldown.is_throttled(decision.trigger):
            logger.warning(
                "recovery_cooldown_throttled trigger=%s",
                decision.trigger,
            )
            return False

        self._start_time = time.monotonic()
        self._orchestrator = await handle_enter_recovery(
            decision=decision, deps=deps,
        )
        return True

    async def step(self) -> None:
        """Advance recovery one tick.  Auto-cleans up when the orchestrator
        reports DONE and invokes the ``on_recovery_duration`` callback."""
        if self._orchestrator is None:
            return

        try:
            await self._orchestrator.step()
            self._diagnosing_nodes = set(self._orchestrator.bad_node_ids)
        except Exception:
            logger.error("recovery_step_failed, forcing NOTIFY", exc_info=True)
            self._orchestrator.force_notify("recovery step exception")

        if self._orchestrator.is_done():
            self._complete_recovery()

    def add_bad_nodes(self, node_ids: list[str]) -> None:
        if self._orchestrator is not None:
            self._orchestrator.add_bad_nodes(node_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _complete_recovery(self) -> None:
        assert self._orchestrator is not None
        recovery_elapsed = (
            time.monotonic() - self._start_time
            if self._start_time is not None
            else 0.0
        )
        logger.info(
            "recovery_complete trigger=%s duration_seconds=%.1f",
            self._orchestrator.trigger, recovery_elapsed,
        )

        if self._on_recovery_duration is not None:
            self._on_recovery_duration(recovery_elapsed)

        self._last_phase_history = list(self._orchestrator.phase_history)
        self._orchestrator = None
        self._start_time = None
        self._diagnosing_nodes.clear()
