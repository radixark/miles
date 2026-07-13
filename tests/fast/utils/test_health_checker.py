import asyncio

from miles.utils.ft_utils.control_server.models import TriState
from miles.utils.ft_utils.health_checker import NoopHealthChecker, SimpleHealthChecker
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
) -> tuple[SimpleHealthChecker, FakeClock]:
    from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig

    if check_fn is None:

        async def check_fn() -> None:
            pass

    c = clock or FakeClock()
    checker = SimpleHealthChecker(
        name=name,
        check_fn=check_fn,
        on_result=on_result,
        config=SimpleHealthCheckerConfig(
            interval=interval, timeout=timeout, first_wait=first_wait, failure_threshold=failure_threshold
        ),
        clock=c,
    )
    return checker, c


class TestStartStop:
    async def test_start_creates_task(self):
        checker, _ = _make_checker()
        assert checker._task is None

        await checker.start()
        assert checker._task is not None

        checker.stop()
        assert checker._task is None

    async def test_start_is_idempotent(self):
        checker, _ = _make_checker()
        await checker.start()
        task = checker._task

        await checker.start()
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
        await checker.start()

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
        await checker.start()

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
        await checker.start()
        await _settle(clock)
        checker.stop()

        assert results == [True]

    async def test_on_result_false_on_failure(self):
        results: list[bool] = []

        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s))
        await checker.start()
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
        await checker.start()

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
        await checker.start()

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
        await checker.start()
        await _settle(clock)
        # first_wait=0 so first check runs immediately on start
        assert results == [True]

        for _ in range(3):
            await clock.elapse(5.0)
        checker.stop()

        assert results == [True, False, True, False]


class TestPauseResume:
    async def test_paused_checker_does_not_call_check_fn(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, interval=5.0)
        checker.pause()

        await checker.start()
        await clock.elapse(20.0)
        checker.stop()

        assert call_count == 0

    async def test_resume_after_pause_resumes_checking(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, interval=5.0)
        checker.pause()

        await checker.start()
        await clock.elapse(20.0)
        assert call_count == 0

        checker.resume()
        await clock.elapse(5.0)
        checker.stop()

        assert call_count >= 1

    async def test_pause_resume_flags(self):
        checker, _ = _make_checker()
        assert not checker._paused

        checker.pause()
        assert checker._paused

        checker.resume()
        assert not checker._paused


class TestNeedFirstWait:
    async def test_resume_triggers_first_wait_again(self):
        """After resume, the loop waits first_wait before the next check."""
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, first_wait=100.0, interval=5.0)
        await checker.start()

        # Step 1: Initial first_wait (100s)
        await clock.elapse(50.0)
        assert call_count == 0
        await clock.elapse(50.0)
        assert call_count == 1

        # Step 2: Normal interval (5s)
        await clock.elapse(5.0)
        assert call_count == 2

        # Step 3: Pause + resume resets first_wait
        checker.pause()
        checker.resume()

        # Step 4: Need to elapse past the pending interval sleep first,
        # then the new first_wait (100s) before next check
        await clock.elapse(5.0)
        assert call_count == 2

        await clock.elapse(50.0)
        assert call_count == 2

        await clock.elapse(50.0)
        assert call_count == 3

        checker.stop()

    async def test_pause_without_resume_no_first_wait(self):
        checker, clock = _make_checker(first_wait=300.0)
        await checker.start()
        await clock.elapse(300.0)
        assert checker._need_first_wait is False

        checker.pause()
        assert checker._need_first_wait is False

        checker.stop()


class TestTriState:
    async def test_initial_status_is_unknown(self):
        checker, _ = _make_checker()
        assert checker.status == TriState.UNKNOWN

    async def test_healthy_after_successful_check(self):
        checker, clock = _make_checker()
        await checker.start()
        await _settle(clock)

        assert checker.status == TriState.TRUE
        checker.stop()

    async def test_unhealthy_after_failed_check(self):
        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn)
        await checker.start()
        await _settle(clock)

        assert checker.status == TriState.FALSE
        checker.stop()

    async def test_stop_resets_to_unknown(self):
        checker, clock = _make_checker()
        await checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        checker.stop()
        assert checker.status == TriState.UNKNOWN

    async def test_pause_resets_to_unknown(self):
        checker, clock = _make_checker()
        await checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        checker.pause()
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_resume_resets_to_unknown(self):
        checker, clock = _make_checker()
        await checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        checker.pause()
        checker.resume()
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
        await checker.start()

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
        await checker.start()
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
        await checker.start()

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
        await checker.start()

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
        await checker.start()
        await _settle(clock)
        for _ in range(3):
            await clock.elapse(5.0)
        checker.stop()

        assert results == [True, False, False, False]

    async def test_resume_resets_failure_counter(self):
        check_fn = self._flaky_check_fn([False, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        await checker.start()
        await _settle(clock)
        await clock.elapse(5.0)
        assert checker._consecutive_failures == 2

        checker.resume()
        assert checker._consecutive_failures == 0
        checker.stop()


class TestRolloutHealthCheckerPreservesImmediateFailure:
    def test_rollout_checker_forces_failure_threshold_one(self):
        """Rollout health checker keeps pre-debounce semantics (single failure -> unhealthy)
        even when handed a config with a larger threshold."""
        from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig, create_rollout_cell_health_checker

        checker = create_rollout_cell_health_checker(
            cell_id="c0",
            get_engines=lambda: [object()],
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=5),
        )

        assert checker._config.failure_threshold == 1


class TestNoopHealthChecker:
    def test_noop_status_is_always_unknown(self):
        checker = NoopHealthChecker()
        assert checker.status == TriState.UNKNOWN
