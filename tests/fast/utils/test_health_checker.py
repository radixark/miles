import asyncio

import pytest
from pydantic import ValidationError

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    ActivenessTracker,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.test_utils.clock import FakeClock


async def _settle(clock: FakeClock) -> None:
    for _ in range(1000):
        if clock.pending_count >= 1:
            return
        await asyncio.sleep(0)


def _make_checker(
    *,
    check_fn=None,
    on_result=None,
    interval: float = 10.0,
    timeout: float = 5.0,
    first_wait: float = 0.0,
    failure_threshold: int = 1,
    name: str = "test",
    clock: FakeClock | None = None,
    activeness: "_Activeness | None" = None,
) -> tuple[SimpleHealthChecker, FakeClock]:
    from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig

    if check_fn is None:

        async def check_fn() -> None:
            pass

    c = clock or FakeClock()
    checker = SimpleHealthChecker(
        name=name,
        check_fn=check_fn,
        get_activeness=activeness or _Activeness(),
        on_result=on_result,
        config=SimpleHealthCheckerConfig(
            interval=interval, timeout=timeout, first_wait=first_wait, failure_threshold=failure_threshold
        ),
        clock=c,
    )
    return checker, c


class _Activeness:
    def __init__(self, active: bool = True) -> None:
        self._tracker = ActivenessTracker(active=active)

    @property
    def active(self) -> bool:
        return self._tracker.get().active

    @active.setter
    def active(self, value: bool) -> None:
        self._tracker.bump_active(value)

    def __call__(self) -> ActiveAndEpoch:
        return self._tracker.get()


class TestActivenessTracker:
    def test_repeating_the_same_activeness_does_not_bump_the_epoch(self):
        """A redundant state publication must not invalidate in-flight probes or re-arm the grace period."""
        tracker = ActivenessTracker(active=True)
        tracker.bump_active(False)
        after_transition = tracker.get()

        tracker.bump_active(False)

        assert after_transition == ActiveAndEpoch(active=False, epoch=1)
        assert tracker.get() == after_transition


class TestConfig:
    @pytest.mark.parametrize("interval", [0.0, 0.999])
    def test_subsecond_health_check_intervals_are_rejected(self, interval: float) -> None:
        """The supported heartbeat cadence keeps the twelve-hour RPC identity budget finite."""
        with pytest.raises(ValidationError):
            SimpleHealthCheckerConfig(
                interval=interval,
                timeout=10.0,
                first_wait=0.0,
                failure_threshold=3,
            )


class TestStartStop:
    async def test_start_creates_task(self):
        checker, _ = _make_checker()
        assert checker._task is None

        checker.start()
        assert checker._task is not None

        checker.stop()
        assert checker._task is None

    async def test_start_is_idempotent(self):
        checker, _ = _make_checker()
        checker.start()
        task = checker._task

        checker.start()
        assert checker._task is task

        checker.stop()

    async def test_stop_without_start_is_noop(self):
        checker, _ = _make_checker()
        checker.stop()


class TestCheckFnCalled:
    async def test_check_fn_called_after_first_interval(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, interval=10.0)
        checker.start()

        # Step 1: first_wait=0, so first check runs immediately after task starts
        await _settle(clock)
        assert call_count == 1

        # Step 2: Elapse less than interval — no second check
        await clock.elapse(5.0)
        assert call_count == 1

        # Step 3: Elapse to interval — second check
        await clock.elapse(5.0)
        assert call_count == 2

        checker.stop()

    async def test_first_wait_delays_first_check(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, first_wait=300.0, interval=10.0)
        checker.start()
        await _settle(clock)

        # Step 1: Elapse 100s — still in first_wait
        await clock.elapse(100.0)
        assert call_count == 0

        # Step 2: Elapse to 300s — first_wait completes, first check runs
        await clock.elapse(200.0)
        assert call_count == 1

        # Step 3: Elapse interval — second check
        await clock.elapse(10.0)
        assert call_count == 2

        checker.stop()


class TestOnResult:
    async def test_on_result_true_on_success(self):
        results: list[bool] = []

        checker, clock = _make_checker(on_result=lambda s: results.append(s))
        checker.start()
        await _settle(clock)
        checker.stop()

        assert results == [True]

    async def test_on_result_false_on_failure(self):
        results: list[bool] = []

        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s))
        checker.start()
        await _settle(clock)
        checker.stop()

        assert results == [False]

    async def test_loop_survives_on_result_raising(self, caplog):
        """A raising on_result callback is logged and does not kill the check loop."""
        results: list[bool] = []

        def on_result(success: bool) -> None:
            results.append(success)
            raise RuntimeError("callback boom")

        checker, clock = _make_checker(on_result=on_result, interval=5.0)
        checker.start()

        await _settle(clock)
        await clock.elapse(5.0)
        checker.stop()

        assert results == [True, True]
        assert any("on_result_failed" in r.message for r in caplog.records)

    async def test_loop_continues_after_failure(self):
        results: list[bool] = []

        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0)
        checker.start()

        await _settle(clock)
        await clock.elapse(5.0)
        checker.stop()

        assert results == [False, False]

    async def test_intermittent_failure(self):
        call_count = 0
        results: list[bool] = []

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("intermittent")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0)
        checker.start()
        await _settle(clock)
        # first_wait=0 so first check runs immediately on start
        assert results == [True]

        for _ in range(3):
            await clock.elapse(5.0)
        checker.stop()

        assert results == [True, False, True, False]


class TestActiveness:
    async def test_an_inactive_checker_does_not_call_check_fn(self):
        """Probing a cell that is offloaded or not yet serving would report a false failure."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        activeness = _Activeness(active=False)
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)

        checker.start()
        await _settle(clock)
        await clock.elapse(20.0)
        checker.stop()

        assert call_count == 0

    async def test_becoming_active_resumes_checking(self):
        """Nothing calls a resume method any more, so the loop must pick the change up itself."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        activeness = _Activeness(active=False)
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)

        checker.start()
        await _settle(clock)
        await clock.elapse(20.0)
        assert call_count == 0

        activeness.active = True
        await clock.elapse(5.0)
        checker.stop()

        assert call_count >= 1

    async def test_becoming_inactive_stops_checking_without_one_last_probe(self):
        """Activeness is read before deciding to probe, so an offloaded engine is never hit."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)

        checker.start()
        await _settle(clock)
        calls_while_active = call_count

        activeness.active = False
        await clock.elapse(20.0)
        checker.stop()

        assert call_count == calls_while_active


class TestNeedFirstWait:
    async def test_becoming_active_again_triggers_first_wait(self):
        """After a pause the engine may have been replaced, so the grace period applies again."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, first_wait=100.0, interval=5.0, activeness=activeness)
        checker.start()
        await _settle(clock)

        # Step 1: Initial first_wait (100s)
        await clock.elapse(50.0)
        assert call_count == 0
        await clock.elapse(50.0)
        assert call_count == 1

        # Step 2: Normal interval (5s)
        await clock.elapse(5.0)
        assert call_count == 2

        # Step 3: a full off/on cycle observed by the loop resets first_wait
        activeness.active = False
        await clock.elapse(5.0)
        activeness.active = True

        # Step 4: the new first_wait (100s) must elapse before the next check
        await clock.elapse(5.0)
        assert call_count == 2

        await clock.elapse(50.0)
        assert call_count == 2

        await clock.elapse(50.0)
        assert call_count == 3

        checker.stop()

    async def test_going_inactive_does_not_re_arm_first_wait(self):
        """Only the transition back to active restarts the grace period."""
        activeness = _Activeness()
        checker, clock = _make_checker(first_wait=300.0, activeness=activeness)
        checker.start()
        await _settle(clock)
        await clock.elapse(300.0)
        assert checker._need_first_wait is False

        activeness.active = False
        await clock.elapse(10.0)
        assert checker._need_first_wait is False

        checker.stop()

    async def test_becoming_inactive_during_first_wait_skips_the_due_probe(self):
        """The probe that becomes due at the end of the grace period must not hit an engine offloaded meanwhile."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, first_wait=100.0, interval=5.0, activeness=activeness)
        checker.start()
        await _settle(clock)

        await clock.elapse(50.0)
        assert call_count == 0

        activeness.active = False
        await clock.elapse(50.0)
        assert call_count == 0
        assert checker.status == TriState.UNKNOWN

        await clock.elapse(10.0)
        assert call_count == 0

        checker.stop()


class TestProbeTimeout:
    async def test_a_timed_out_probe_is_reported_failed_and_polling_continues(self):
        """A check hanging past the timeout is cancelled, published as a failure, and followed by later polls."""
        results: list[bool] = []
        published = asyncio.Event()
        cancelled = asyncio.Event()

        async def check_fn() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def on_result(success: bool) -> None:
            results.append(success)
            published.set()

        checker, clock = _make_checker(check_fn=check_fn, on_result=on_result, timeout=0.05, interval=5.0)
        checker.start()
        await asyncio.wait_for(published.wait(), timeout=10)

        assert results == [False]
        assert cancelled.is_set()
        assert checker.status == TriState.FALSE

        published.clear()
        await _settle(clock)
        await clock.elapse(5.0)
        await asyncio.wait_for(published.wait(), timeout=10)

        assert results == [False, False]
        checker.stop()


class TestTriState:
    async def test_initial_status_is_unknown(self):
        checker, _ = _make_checker()
        assert checker.status == TriState.UNKNOWN

    async def test_healthy_after_successful_check(self):
        checker, clock = _make_checker()
        checker.start()
        await _settle(clock)

        assert checker.status == TriState.TRUE
        checker.stop()

    async def test_unhealthy_after_failed_check(self):
        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn)
        checker.start()
        await _settle(clock)

        assert checker.status == TriState.FALSE
        checker.stop()

    async def test_stop_resets_to_unknown(self):
        checker, clock = _make_checker()
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        checker.stop()
        assert checker.status == TriState.UNKNOWN

    async def test_going_inactive_resets_to_unknown(self):
        """A stale Healthy verdict about a sleeping engine would be read as a live replica."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False
        await clock.elapse(10.0)
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_recovers_from_unhealthy_to_healthy(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")

        checker, clock = _make_checker(check_fn=check_fn, interval=5.0)
        checker.start()

        await _settle(clock)
        assert checker.status == TriState.FALSE

        await clock.elapse(5.0)
        assert checker.status == TriState.TRUE

        checker.stop()


class TestFailureThresholdDebounce:
    """With failure_threshold > 1, transient failures must not flip the status to FALSE
    until that many consecutive checks have failed; any success resets the counter."""

    def _flaky_check_fn(self, outcomes: list[bool]):
        idx = 0

        async def check_fn() -> None:
            nonlocal idx
            ok = outcomes[idx]
            idx += 1
            if not ok:
                raise RuntimeError("boom")

        return check_fn

    async def test_below_threshold_keeps_previous_status(self):
        # success, then 2 failures (< threshold 3): status stays TRUE.
        check_fn = self._flaky_check_fn([True, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        await clock.elapse(5.0)
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 1

        await clock.elapse(5.0)
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 2

        checker.stop()

    async def test_status_flips_false_only_at_threshold(self):
        check_fn = self._flaky_check_fn([False, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        checker.start()

        await _settle(clock)
        assert checker.status == TriState.UNKNOWN  # 1st failure: below threshold, keep initial UNKNOWN

        await clock.elapse(5.0)
        assert checker.status == TriState.UNKNOWN  # 2nd failure: still below threshold

        await clock.elapse(5.0)
        assert checker.status == TriState.FALSE  # 3rd consecutive failure: threshold reached

        checker.stop()

    async def test_success_resets_failure_counter(self):
        # 2 failures, a success, then 2 more failures: never reaches 3 consecutive, stays TRUE.
        check_fn = self._flaky_check_fn([False, False, True, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        checker.start()

        await _settle(clock)  # fail 1
        await clock.elapse(5.0)  # fail 2
        assert checker._consecutive_failures == 2

        await clock.elapse(5.0)  # success -> reset
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 0

        await clock.elapse(5.0)  # fail 1
        await clock.elapse(5.0)  # fail 2
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 2

        checker.stop()

    async def test_on_result_reports_raw_per_check_not_debounced(self):
        results: list[bool] = []
        check_fn = self._flaky_check_fn([True, False, False, False])
        checker, clock = _make_checker(
            check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0, failure_threshold=3
        )
        checker.start()
        await _settle(clock)
        for _ in range(3):
            await clock.elapse(5.0)
        checker.stop()

        assert results == [True, False, False, False]

    async def test_becoming_active_again_resets_the_failure_counter(self):
        """Failures against the old engine must not count toward recycling its replacement."""
        check_fn = self._flaky_check_fn([False, False, False])
        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3, activeness=activeness)
        checker.start()
        await _settle(clock)
        await clock.elapse(5.0)
        assert checker._consecutive_failures == 2

        activeness.active = False
        await clock.elapse(5.0)
        activeness.active = True
        await clock.elapse(5.0)

        assert checker._consecutive_failures <= 1
        checker.stop()

    async def test_a_pause_resume_completed_between_two_polls_resets_the_failure_counter(self):
        """A reconfigure that flips activeness off and on while the loop sleeps must still clear stale failures."""
        check_fn = self._flaky_check_fn([False, False, False])
        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=check_fn, first_wait=100.0, interval=10.0, failure_threshold=3, activeness=activeness
        )
        checker.start()
        await _settle(clock)

        await clock.elapse(100.0)
        await clock.elapse(10.0)
        assert checker._consecutive_failures == 2

        activeness.active = False
        activeness.active = True
        await clock.elapse(10.0)

        assert checker._consecutive_failures == 0
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_pause_resume_completed_between_two_polls_re_arms_the_first_wait(self):
        """The unobserved reconfigure also restarts the grace period, so the replacement is not probed at once."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=check_fn, first_wait=100.0, interval=10.0, failure_threshold=3, activeness=activeness
        )
        checker.start()
        await _settle(clock)

        await clock.elapse(100.0)
        assert call_count == 1

        activeness.active = False
        activeness.active = True
        await clock.elapse(10.0)
        assert call_count == 1

        await clock.elapse(100.0)
        assert call_count == 2
        checker.stop()


class TestDiscardingStaleProbeResults:
    def _hanging_check_fn(self, started: asyncio.Event):
        async def check_fn() -> None:
            started.set()
            await asyncio.sleep(3600)

        return check_fn

    def _gated_check_fn(self, started: asyncio.Event, release: asyncio.Event):
        async def check_fn() -> None:
            started.set()
            await release.wait()
            raise RuntimeError("boom")

        return check_fn

    async def test_a_probe_that_lands_after_a_pause_publishes_no_result(self):
        """A probe that outlives its window would report a failure about an engine nobody was watching."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()
        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=self._gated_check_fn(started, release),
            on_result=lambda s: results.append(s),
            interval=5.0,
            activeness=activeness,
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        release.set()
        await _settle(clock)

        assert results == []
        assert checker._consecutive_failures == 0
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_probe_that_spans_a_whole_pause_resume_window_publishes_no_result(self):
        """Activeness is back to True when the probe lands, so only the epoch tells the result is stale."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()
        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=self._gated_check_fn(started, release),
            on_result=lambda s: results.append(s),
            interval=5.0,
            activeness=activeness,
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        activeness.active = True
        release.set()
        await _settle(clock)

        assert results == []
        assert checker._consecutive_failures == 0
        checker.stop()

    async def test_the_loop_keeps_polling_after_a_discarded_result(self):
        """Discarding one result must not silently kill the checker for the rest of the run."""
        call_count = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                started.set()
                await release.wait()

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        activeness.active = True
        release.set()
        await _settle(clock)

        await clock.elapse(5.0)

        assert call_count == 2
        checker.stop()

    async def test_a_successful_probe_spanning_pause_resume_is_discarded(self):
        """An old engine's success must not mark its replacement healthy without probing it."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def check_fn() -> None:
            started.set()
            await release.wait()

        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0, activeness=activeness
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        activeness.active = True
        release.set()
        await _settle(clock)

        assert results == []
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_probe_that_lands_inside_its_own_window_is_published(self):
        """Discarding is for stale results only; an undisturbed window must still publish its verdict."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()
        checker, clock = _make_checker(
            check_fn=self._gated_check_fn(started, release), on_result=lambda s: results.append(s), interval=5.0
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        release.set()
        await _settle(clock)

        assert results == [False]
        checker.stop()

    async def test_stopping_also_kills_a_probe_still_in_flight(self):
        """A probe left running after stop() outlives the cell and keeps dialing a dead engine."""
        started = asyncio.Event()
        checker, _ = _make_checker(check_fn=self._hanging_check_fn(started), interval=5.0)
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)
        probe_task = checker._probe_task

        checker.stop()

        with pytest.raises(asyncio.CancelledError):
            await probe_task
        assert probe_task.cancelled()


class TestNoopHealthChecker:
    def test_noop_status_is_always_unknown(self):
        checker = NoopHealthChecker()
        assert checker.status == TriState.UNKNOWN
