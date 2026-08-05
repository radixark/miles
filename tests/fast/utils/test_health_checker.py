import argparse
import asyncio

import pytest
from _pytest.recwarn import WarningsRecorder

from miles.utils.arguments import get_miles_extra_args_provider
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

    async def test_stopping_before_a_probe_starts_does_not_abandon_its_coroutine(
        self, recwarn: WarningsRecorder
    ) -> None:
        """Stopping a newly scheduled probe must not leave its coroutine unawaited."""
        checker, _ = _make_checker(interval=5.0)
        run_probe = asyncio.create_task(checker._run_probe())
        await asyncio.sleep(0)

        checker.stop()

        with pytest.raises(asyncio.CancelledError):
            await run_probe
        assert not [warning for warning in recwarn if "was never awaited" in str(warning.message)]


class TestNoopHealthChecker:
    def test_noop_status_is_always_unknown(self):
        checker = NoopHealthChecker()
        assert checker.status == TriState.UNKNOWN


class TestFailureThresholdArgument:
    def _parse(self, extra: list[str], **add_arguments_kwargs: int) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        SimpleHealthCheckerConfig.add_arguments(parser, prefix="demo-check", **add_arguments_kwargs)
        return parser.parse_args(extra)

    def test_a_caller_can_ask_for_a_debounce_of_its_own(self):
        """A checker polling on a long interval needs to report the first failure, not the third."""
        args = self._parse([], failure_threshold_default=1)

        assert args.demo_check_failure_threshold == 1

    def test_a_caller_that_asks_for_nothing_keeps_the_shared_debounce(self):
        """Giving one caller a tighter default must not tighten it for every other checker."""
        args = self._parse([])

        assert args.demo_check_failure_threshold == 3

    def test_an_explicit_flag_still_beats_a_caller_supplied_default(self):
        """A caller default that is nailed into the parser would make the flag unusable."""
        args = self._parse(["--demo-check-failure-threshold", "5"], failure_threshold_default=1)

        assert args.demo_check_failure_threshold == 5


class TestShippedRolloutConfig:
    async def test_a_dead_engine_is_reported_unhealthy_after_a_single_probe_under_the_shipped_defaults(self):
        """A cell whose engine is already gone must not read Healthy for two more 30s probe intervals."""
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        config = SimpleHealthCheckerConfig.from_args(
            parser.parse_args(["--rollout-batch-size", "64"]), prefix="rollout_health_check"
        )

        async def check_fn() -> None:
            raise RuntimeError("engine down")

        clock = FakeClock()
        checker = SimpleHealthChecker(
            name="rollout-cell",
            check_fn=check_fn,
            get_activeness=_Activeness(),
            config=config,
            clock=clock,
        )
        checker.start()
        await _settle(clock)

        assert checker.status == TriState.FALSE
        assert checker._consecutive_failures == 1
        checker.stop()
