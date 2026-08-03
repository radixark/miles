import asyncio

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import (
    ActivenessState,
    ActivenessTracker,
    NoopHealthChecker,
    SimpleHealthChecker,
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

    def __call__(self) -> ActivenessState:
        return self._tracker.get()


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

    async def test_going_inactive_is_reported_before_the_loop_wakes_up(self):
        """The pause is decided between two probes, and a reader in that window must not see the old verdict."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness, interval=30.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False

        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_pause_and_resume_between_two_probes_hides_the_old_verdict_until_the_loop_catches_up(self):
        """A verdict from before the pause describes an engine that may have been replaced meanwhile."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness, interval=30.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False
        activeness.active = True

        assert checker.status == TriState.UNKNOWN
        await clock.elapse(30.0)
        assert checker.status == TriState.TRUE
        checker.stop()

    async def test_a_failed_verdict_does_not_outlive_the_pause_that_follows_it(self):
        """A cell suspended right after a failed probe would otherwise be healed a second time."""

        async def check_fn() -> None:
            raise RuntimeError("boom")

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, activeness=activeness, interval=30.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.FALSE

        activeness.active = False

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


class TestStatusWhileProbingIsPaused:
    async def test_a_pause_is_reported_while_a_probe_is_still_in_flight(self):
        """The loop is stuck inside a probe, so only a live read can hide the verdict published before the pause."""
        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                started.set()
                await release.wait()

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, activeness=activeness, interval=30.0, timeout=3600.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        await clock.elapse(30.0)
        await asyncio.wait_for(started.wait(), timeout=1)
        activeness.active = False

        assert checker.status == TriState.UNKNOWN
        assert checker._status == TriState.TRUE

        release.set()
        checker.stop()

    async def test_a_resumed_checker_serves_a_fresh_verdict_again(self):
        """Hiding the verdict lasts exactly as long as the pause instead of latching the checker at unknown."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness, interval=5.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False
        assert checker.status == TriState.UNKNOWN

        activeness.active = True
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


class TestNoopHealthChecker:
    def test_noop_status_is_always_unknown(self):
        checker = NoopHealthChecker()
        assert checker.status == TriState.UNKNOWN
