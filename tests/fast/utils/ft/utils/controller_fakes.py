from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import NamedTuple

from prometheus_client import CollectorRegistry

from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.factory import create_ft_controller
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.metric_names import AGENT_HEARTBEAT
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

# ---------------------------------------------------------------------------
# Platform fakes
# ---------------------------------------------------------------------------


class FakeNodeManager(NodeManagerProtocol):
    """In-memory implementation of NodeManagerProtocol for testing."""

    def __init__(self) -> None:
        self._bad_nodes: set[str] = set()
        self._ever_marked_bad: set[str] = set()

    async def mark_node_bad(
        self,
        node_id: str,
        reason: str = "",
        node_metadata: dict[str, str] | None = None,
    ) -> None:
        self._bad_nodes.add(node_id)
        self._ever_marked_bad.add(node_id)
        self.last_node_metadata = node_metadata

    async def unmark_node_bad(self, node_id: str) -> None:
        self._bad_nodes.discard(node_id)

    def is_node_bad(self, node_id: str) -> bool:
        return node_id in self._bad_nodes

    def was_ever_marked_bad(self, node_id: str) -> bool:
        return node_id in self._ever_marked_bad

    async def get_bad_nodes(self) -> list[str]:
        return sorted(self._bad_nodes)


class FakeNotifier(NotifierProtocol):
    """Records all send() calls for assertion in tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def send(self, title: str, content: str, severity: str) -> None:
        self.calls.append((title, content, severity))

    async def aclose(self) -> None:
        pass


class FakeMainJob(MainJobProtocol):
    """Programmable implementation of MainJobProtocol for testing."""

    def __init__(self, status_sequence: list[JobStatus] | None = None) -> None:
        self._status_sequence = status_sequence or [JobStatus.RUNNING]
        self._call_count: int = 0
        self._stopped: bool = False
        self._submitted: bool = False
        self._submit_call_count: int = 0
        self._run_id: str = "fake-initial"

    async def get_job_status(self) -> JobStatus:
        index = min(self._call_count, len(self._status_sequence) - 1)
        status = self._status_sequence[index]
        self._call_count += 1
        return status

    async def stop_job(self, timeout_seconds: int = 300) -> None:
        self._stopped = True

    async def submit_job(self) -> str:
        self._submitted = True
        self._submit_call_count += 1
        self._call_count = 0
        self._run_id = f"fake-run-{self._submit_call_count}"
        return self._run_id


# ---------------------------------------------------------------------------
# Test detectors
# ---------------------------------------------------------------------------


class FixedDecisionDetector(BaseFaultDetector):
    """Detector that always returns a fixed Decision. Tracks call count."""

    def __init__(self, decision: Decision) -> None:
        self.call_count = 0
        self._decision = decision

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        self.call_count += 1
        return self._decision


_ALWAYS_NONE_DECISION = Decision(action=ActionType.NONE, reason="always none")
_ALWAYS_MARK_BAD_DECISION = Decision(
    action=ActionType.ENTER_RECOVERY,
    bad_node_ids=["node-1"],
    reason="test fault detected",
    trigger=TriggerType.CRASH,
)


def AlwaysNoneDetector() -> FixedDecisionDetector:
    return FixedDecisionDetector(decision=_ALWAYS_NONE_DECISION)


def AlwaysMarkBadDetector() -> FixedDecisionDetector:
    return FixedDecisionDetector(decision=_ALWAYS_MARK_BAD_DECISION)


def AlwaysEnterRecoveryDetector(
    trigger: TriggerType = TriggerType.CRASH,
    reason: str = "test recovery",
) -> FixedDecisionDetector:
    return FixedDecisionDetector(
        decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger=trigger,
            reason=reason,
        )
    )


class CrashingDetector(BaseFaultDetector):
    """Detector that raises an exception on evaluate(). For testing fault isolation."""

    def __init__(self) -> None:
        self.call_count = 0

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        self.call_count += 1
        raise RuntimeError("detector internal error")


class FastHangDetector(BaseFaultDetector):
    """HangDetector with sub-minute timeout for fast testing."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self._timeout = timedelta(seconds=timeout_seconds)

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.RUNNING:
            return Decision(action=ActionType.NONE, reason="not running")

        df = ctx.metric_store.changes(
            AGENT_HEARTBEAT,
            window=self._timeout,
            label_filters={"rank": "0"},
        )
        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="no iteration data")

        if df["value"][0] == 0:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"iteration stalled for {self._timeout.total_seconds()}s",
                trigger=TriggerType.HANG,
            )
        return Decision(action=ActionType.NONE, reason="progressing")


class OneShotCrashDetector(BaseFaultDetector):
    """Fires ENTER_RECOVERY once, then returns NONE forever after."""

    def __init__(self) -> None:
        self._fired = False

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        if not self._fired:
            self._fired = True
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason="one-shot crash for test",
                trigger=TriggerType.CRASH,
            )
        return Decision(action=ActionType.NONE, reason="no fault")


# ---------------------------------------------------------------------------
# Controller test harness
# ---------------------------------------------------------------------------


class ControllerTestHarness(NamedTuple):
    controller: FtController
    node_manager: FakeNodeManager
    main_job: FakeMainJob
    metric_store: MiniPrometheus
    mini_wandb: MiniWandb
    controller_exporter: ControllerExporter
    notifier: FakeNotifier | None


def make_test_controller(
    detectors: list[BaseFaultDetector] | None = None,
    status_sequence: list[JobStatus] | None = None,
    notifier: FakeNotifier | None = FakeNotifier,
    tick_interval: float = 0.01,
    controller_exporter: ControllerExporter | None = None,
    diagnostic_orchestrator: object | None = None,
    recovery_cooldown_minutes: float = 30.0,
    recovery_cooldown_max_count: int = 3,
    registration_grace_ticks: int = 0,
    register_dummy_rank: bool = True,
    monitoring_success_iterations: int = 10,
) -> ControllerTestHarness:
    """Construct a Controller and all its dependencies for testing.

    ``notifier`` defaults to a fresh FakeNotifier instance. Pass ``None``
    explicitly to create a Controller without a notifier.

    ``diagnostic_orchestrator`` defaults to a real DiagnosticOrchestrator with
    empty pipeline (same behavior as old stub). Pass a FakeDiagnosticOrchestrator
    for recovery-specific tests.
    """
    real_notifier: FakeNotifier | None = FakeNotifier() if notifier is FakeNotifier else notifier

    node_manager = FakeNodeManager()
    main_job = FakeMainJob(status_sequence=status_sequence)
    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()

    if controller_exporter is None:
        controller_exporter = ControllerExporter(registry=CollectorRegistry())

    recovery_cooldown = SlidingWindowThrottle(
        window_minutes=recovery_cooldown_minutes,
        max_count=recovery_cooldown_max_count,
    )

    controller = create_ft_controller(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        scrape_target_manager=metric_store,
        notifier=real_notifier,
        detectors=detectors,
        tick_interval=tick_interval,
        controller_exporter=controller_exporter,
        diagnostic_orchestrator=diagnostic_orchestrator,
        recovery_cooldown=recovery_cooldown,
        registration_grace_ticks=registration_grace_ticks,
        monitoring_success_iterations=monitoring_success_iterations,
    )

    if register_dummy_rank:
        controller._activate_run("dummy-run")
        controller.training_rank_roster.rank_placement[0] = "node-0"
        controller.training_rank_roster.rank_placement[1] = "node-1"

    return ControllerTestHarness(
        controller=controller,
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        controller_exporter=controller_exporter,
        notifier=real_notifier,
    )


# ---------------------------------------------------------------------------
# Standalone failing callables (for monkey-patching existing fakes)
# ---------------------------------------------------------------------------


async def failing_stop_job(timeout_seconds: int = 300) -> None:
    raise RuntimeError("stop failed")


async def failing_submit_job() -> str:
    raise RuntimeError("submit failed")


async def failing_mark_node_bad(
    node_id: str,
    reason: str = "",
    node_metadata: dict[str, str] | None = None,
) -> None:
    raise RuntimeError("mark_node_bad failed")


# ---------------------------------------------------------------------------
# Factory helpers (create fakes pre-wired with failures)
# ---------------------------------------------------------------------------


def make_failing_main_job(
    *,
    fail_stop: bool = False,
    fail_submit: bool = False,
    status_sequence: list[JobStatus] | None = None,
) -> FakeMainJob:
    """FakeMainJob with configurable method failures."""
    job = FakeMainJob(status_sequence=status_sequence)
    if fail_stop:
        job.stop_job = failing_stop_job  # type: ignore[assignment]
    if fail_submit:
        job.submit_job = failing_submit_job  # type: ignore[assignment]
    return job


def make_failing_node_manager() -> FakeNodeManager:
    """FakeNodeManager whose mark_node_bad always raises."""
    mgr = FakeNodeManager()
    mgr.mark_node_bad = failing_mark_node_bad  # type: ignore[assignment]
    return mgr


# ---------------------------------------------------------------------------
# Async test-flow helpers
# ---------------------------------------------------------------------------


async def run_controller_briefly(harness: ControllerTestHarness, delay: float = 0.03) -> None:
    """Run the controller loop and shut it down after a short delay."""

    async def _shutdown_soon() -> None:
        await asyncio.sleep(delay)
        await harness.controller.shutdown()

    task = asyncio.create_task(_shutdown_soon())
    await harness.controller.run()
    await task


async def advance_until_recovery_complete(harness: ControllerTestHarness, max_ticks: int = 10) -> None:
    """Tick the controller until recovery is no longer in progress."""
    from miles.utils.ft.controller.state_machines.main import Recovering

    for _ in range(max_ticks):
        if not isinstance(harness.controller._state_machine.state, Recovering):
            return
        await harness.controller._tick()
    assert not isinstance(harness.controller._state_machine.state, Recovering)
