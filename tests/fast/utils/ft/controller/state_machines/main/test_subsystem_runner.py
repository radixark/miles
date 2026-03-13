"""Tests for advance_subsystems() in subsystem_runner.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from miles.utils.ft.controller.state_machines.main.models import NormalSt
from miles.utils.ft.controller.state_machines.main.restart_coordinator import (
    has_pending_main_job_restart,
)
from miles.utils.ft.controller.state_machines.main.subsystem_runner import advance_subsystems
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    RecoveringSt,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalRestartingMainJobSt,
)
from miles.utils.ft.controller.types import TriggerType


# ---------------------------------------------------------------------------
# has_pending_main_job_restart
# ---------------------------------------------------------------------------


class TestHasPendingMainJobRestart:
    def test_returns_true_when_unfulfilled_request_exists(self) -> None:
        subsystems = {
            "gpu": RecoveringSt(
                recovery=EvictingAndRestartingSt(
                    restart=ExternalRestartingMainJobSt(external_execution_result=None),
                    failed_next_state=StopTimeDiagnosticsSt(),
                ),
                trigger=TriggerType.CRASH,
                recovery_start_time=datetime.now(timezone.utc),
            ),
        }
        assert has_pending_main_job_restart(subsystems) is True

    def test_returns_false_when_no_request(self) -> None:
        subsystems = {
            "gpu": DetectingAnomalySt(),
            "net": DetectingAnomalySt(),
        }
        assert has_pending_main_job_restart(subsystems) is False

    def test_returns_false_for_empty_subsystems(self) -> None:
        assert has_pending_main_job_restart({}) is False


# ---------------------------------------------------------------------------
# advance_subsystems early-stop
# ---------------------------------------------------------------------------


def _make_requestor_state() -> RecoveringSt:
    return RecoveringSt(
        recovery=EvictingAndRestartingSt(
            restart=ExternalRestartingMainJobSt(external_execution_result=None),
            failed_next_state=StopTimeDiagnosticsSt(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


class TestAdvanceSubsystemsEarlyStop:
    @pytest.mark.asyncio
    async def test_skips_later_subsystems_when_earlier_one_requests_main_job_restart(
        self,
    ) -> None:
        """Previously advance_subsystems() iterated all subsystems even when
        an earlier subsystem had already produced a pending MAIN_JOB restart
        request. This caused unnecessary side effects in later subsystems
        that would be discarded by trigger_main_job_restart() anyway. Now
        the loop breaks early once a pending request is detected."""
        stepper_calls: list[str] = []

        # Subsystem "a_requestor" already has a pending MAIN_JOB restart.
        # Subsystem "b_later" should NOT be stepped.
        requestor_state = _make_requestor_state()
        state = NormalSt(subsystems={
            "a_requestor": requestor_state,
            "b_later": DetectingAnomalySt(),
        })

        # The subsystem stepper records which subsystems are stepped.
        # For a_requestor: already in ExternalRestartingMainJobSt, no transition.
        # For b_later: would do something, but should be skipped.
        async def mock_stepper(sub_state, ctx):
            return
            yield  # make it an async generator

        async def mock_run_stepper_to_convergence(stepper, sub_state, ctx, *, on_convergence_failure=None):
            return
            yield  # make it an async generator

        # We use the real advance_subsystems but with a dummy context that
        # tracks which subsystems get processed. Since "a_requestor" already
        # has ExternalRestartingMainJobSt, the loop should stop after it.
        from unittest.mock import MagicMock

        context = MagicMock()
        context.shared.subsystem_specs = {
            "a_requestor": MagicMock(),
            "b_later": MagicMock(),
        }

        # Patch context_factories.build_subsystem_context and run_stepper_to_convergence
        import miles.utils.ft.controller.state_machines.main.subsystem_runner as runner_mod
        from miles.utils.ft.utils.state_machine import StateMachineStepper

        original_build_ctx = runner_mod.build_subsystem_context
        original_run_convergence = runner_mod.run_stepper_to_convergence

        subsystems_stepped: list[str] = []

        def tracking_build_ctx(*, spec, context, recovery_stepper, restart_stepper):
            return MagicMock()

        async def tracking_run_convergence(stepper, sub_state, ctx, *, on_convergence_failure=None):
            # Determine which subsystem this is by checking the current state
            subsystems_stepped.append(type(sub_state).__name__)
            return
            yield  # async generator

        runner_mod.build_subsystem_context = tracking_build_ctx
        runner_mod.run_stepper_to_convergence = tracking_run_convergence
        try:
            results = []
            async for s in advance_subsystems(
                state,
                context,
                subsystem_stepper=MagicMock(),
                recovery_stepper=MagicMock(),
                restart_stepper=MagicMock(),
                on_convergence_failure=None,
            ):
                results.append(s)

            # "a_requestor" is sorted first, its stepper produces no transition
            # (no yield from tracking_run_convergence), then has_pending_main_job_restart
            # returns True. "b_later" should NOT be stepped.
            assert "RecoveringSt" in subsystems_stepped
            assert "DetectingAnomalySt" not in subsystems_stepped
        finally:
            runner_mod.build_subsystem_context = original_build_ctx
            runner_mod.run_stepper_to_convergence = original_run_convergence

    @pytest.mark.asyncio
    async def test_no_early_stop_when_no_restart_pending(self) -> None:
        """When no subsystem has a pending MAIN_JOB restart, all subsystems
        are stepped normally."""
        state = NormalSt(subsystems={
            "a_first": DetectingAnomalySt(),
            "b_second": DetectingAnomalySt(),
        })

        from unittest.mock import MagicMock

        context = MagicMock()
        context.shared.subsystem_specs = {
            "a_first": MagicMock(),
            "b_second": MagicMock(),
        }

        import miles.utils.ft.controller.state_machines.main.subsystem_runner as runner_mod

        original_build_ctx = runner_mod.build_subsystem_context
        original_run_convergence = runner_mod.run_stepper_to_convergence

        subsystems_stepped: list[str] = []

        def tracking_build_ctx(*, spec, context, recovery_stepper, restart_stepper):
            return MagicMock()

        async def tracking_run_convergence(stepper, sub_state, ctx, *, on_convergence_failure=None):
            subsystems_stepped.append(type(sub_state).__name__)
            return
            yield

        runner_mod.build_subsystem_context = tracking_build_ctx
        runner_mod.run_stepper_to_convergence = tracking_run_convergence
        try:
            async for _ in advance_subsystems(
                state,
                context,
                subsystem_stepper=MagicMock(),
                recovery_stepper=MagicMock(),
                restart_stepper=MagicMock(),
                on_convergence_failure=None,
            ):
                pass

            assert len(subsystems_stepped) == 2
        finally:
            runner_mod.build_subsystem_context = original_build_ctx
            runner_mod.run_stepper_to_convergence = original_run_convergence
