from __future__ import annotations

import abc
import argparse
import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, NamedTuple

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)


class SimpleHealthCheckerConfig(StrictBaseModel):
    interval: float
    timeout: float
    first_wait: float
    failure_threshold: int

    @staticmethod
    def add_arguments(
        parser: argparse.ArgumentParser,
        *,
        prefix: str,
        interval_default: float = 10.0,
        timeout_default: float = 10.0,
        first_wait_default: float = 300.0,
        failure_threshold_default: int = 3,
    ) -> None:
        parser.add_argument(
            f"--{prefix}-interval",
            type=float,
            default=interval_default,
            help=f"Interval in seconds between {prefix} health checks.",
        )
        parser.add_argument(
            f"--{prefix}-timeout",
            type=float,
            default=timeout_default,
            help=f"Timeout in seconds for a single {prefix} health check RPC.",
        )
        parser.add_argument(
            f"--{prefix}-first-wait",
            type=float,
            default=first_wait_default,
            help=(
                f"Initial grace period (seconds) before starting {prefix} health checks, re-armed on every "
                "resume. This allows time for model compilation and initialization; increase it "
                "significantly when using deepgemm."
            ),
        )
        parser.add_argument(
            f"--{prefix}-failure-threshold",
            type=int,
            default=failure_threshold_default,
            help=(
                f"Number of consecutive failed {prefix} checks before reporting unhealthy. "
                "Debounces transient RPC blips so a single hiccup does not recycle a live cell."
            ),
        )

    @staticmethod
    def from_args(args: object, *, prefix: str) -> SimpleHealthCheckerConfig:
        attr_prefix = prefix.replace("-", "_")
        return SimpleHealthCheckerConfig(
            interval=getattr(args, f"{attr_prefix}_interval"),
            timeout=getattr(args, f"{attr_prefix}_timeout"),
            first_wait=getattr(args, f"{attr_prefix}_first_wait"),
            failure_threshold=getattr(args, f"{attr_prefix}_failure_threshold"),
        )


class ActiveAndEpoch(NamedTuple):
    active: bool
    epoch: int


class ActivenessTracker:
    def __init__(self, *, active: bool) -> None:
        self._state = ActiveAndEpoch(active=active, epoch=0)

    def get(self) -> ActiveAndEpoch:
        return self._state

    def bump_active(self, active: bool) -> None:
        if active == self._state.active:
            return
        self._state = ActiveAndEpoch(active=active, epoch=self._state.epoch + 1)


class BaseHealthChecker(abc.ABC):
    @property
    @abc.abstractmethod
    def status(self) -> TriState: ...

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...


class SimpleHealthChecker(BaseHealthChecker):
    """Periodic async health checker. Calls *check_fn*; reports result via *on_result*.

    Probing is driven by *get_activeness*, read once per loop before deciding to probe.
    After each transition back to active, waits ``first_wait`` seconds before the first check.
    """

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        get_activeness: Callable[[], ActiveAndEpoch],
        on_result: Callable[[bool], None] | None = None,
        config: SimpleHealthCheckerConfig,
        clock: Clock | None = None,
    ) -> None:
        self._name = name
        self._check_fn = check_fn
        self._get_activeness = get_activeness
        self._on_result = on_result
        self._config = config
        self._clock = clock or RealClock()

        self._status = TriState.UNKNOWN
        self._active_and_epoch = ActiveAndEpoch(active=False, epoch=0)
        self._need_first_wait: bool = True
        self._consecutive_failures: int = 0
        self._task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None

    @property
    def status(self) -> TriState:
        active_and_epoch = self._get_activeness()
        if not active_and_epoch.active or active_and_epoch != self._active_and_epoch:
            return TriState.UNKNOWN
        return self._status

    def start(self) -> None:
        if self._task is not None:
            return
        log_structured(logger.info, tag="ft", op="health", phase="start", name=self._name)
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            log_structured(logger.info, tag="ft", op="health", phase="stop", name=self._name)
            self._task.cancel()
            self._task = None
        if self._probe_task is not None:
            self._probe_task.cancel()
            self._probe_task = None
        self._status = TriState.UNKNOWN

    def _on_paused(self) -> None:
        log_structured(logger.info, tag="ft", op="health", phase="pause", name=self._name)
        self._status = TriState.UNKNOWN

    def _on_resumed(self) -> None:
        log_structured(logger.info, tag="ft", op="health", phase="resume", name=self._name)
        self._need_first_wait = True
        self._status = TriState.UNKNOWN
        self._consecutive_failures = 0

    async def _loop(self) -> None:
        while True:
            active_and_epoch = self._get_activeness()
            active = active_and_epoch.active
            if active_and_epoch != self._active_and_epoch:
                self._active_and_epoch = active_and_epoch
                if active:
                    self._on_resumed()
                else:
                    self._on_paused()

            if active and self._need_first_wait:
                self._need_first_wait = False
                log_structured(
                    logger.info,
                    tag="ft",
                    op="health",
                    phase="first_wait",
                    name=self._name,
                    wait_s=self._config.first_wait,
                )
                await self._clock.sleep(self._config.first_wait)
                continue

            if active:
                success = await self._run_probe()
                active_and_epoch_now = self._get_activeness()
                if not active_and_epoch_now.active or active_and_epoch_now.epoch != active_and_epoch.epoch:
                    log_structured(logger.info, tag="ft", op="health", phase="probe_discarded", name=self._name)
                else:
                    self._publish_result(success=success)

            await self._clock.sleep(self._config.interval)

    async def _run_probe(self) -> bool:
        self._probe_task = asyncio.create_task(self._check_fn())

        try:
            await asyncio.wait_for(self._probe_task, timeout=self._config.timeout)
            return True
        except Exception:
            log_structured(logger.error, tag="ft", op="health", phase="check_failed", name=self._name, exc_info=True)
            return False
        finally:
            self._probe_task = None

    def _publish_result(self, *, success: bool) -> None:
        prev_status = self._status
        if success:
            self._consecutive_failures = 0
            self._status = TriState.TRUE
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.failure_threshold:
                self._status = TriState.FALSE

        log_structured(
            logger.info,
            tag="ft",
            op="health",
            phase="poll",
            name=self._name,
            ok=success,
            status=self._status.value,
            consecutive_failures=self._consecutive_failures,
        )

        if prev_status != self._status:
            log_structured(
                logger.info,
                tag="ft",
                op="health",
                phase="status_change",
                name=self._name,
                from_status=prev_status.value,
                to_status=self._status.value,
                consecutive_failures=self._consecutive_failures,
            )

        if self._on_result is not None:
            try:
                self._on_result(success)
            except Exception:
                log_structured(
                    logger.error,
                    tag="ft",
                    op="health",
                    phase="on_result_failed",
                    name=self._name,
                    exc_info=True,
                )


class NoopHealthChecker(BaseHealthChecker):
    def __init__(self) -> None:
        self.stopped: bool = False

    @property
    def status(self) -> TriState:
        return TriState.UNKNOWN

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True
