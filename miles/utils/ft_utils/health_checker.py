from __future__ import annotations

import abc
import argparse
import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from miles.utils.ft_utils.control_server.models import TriState
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
    def add_arguments(parser: argparse.ArgumentParser, *, prefix: str) -> None:
        parser.add_argument(
            f"--{prefix}-interval",
            type=float,
            default=10.0,
            help=f"Interval in seconds between {prefix} health checks.",
        )
        parser.add_argument(
            f"--{prefix}-timeout",
            type=float,
            default=10.0,
            help=f"Timeout in seconds for a single {prefix} health check RPC.",
        )
        parser.add_argument(
            f"--{prefix}-first-wait",
            type=float,
            default=300.0,
            help=f"Initial grace period (seconds) before starting {prefix} health checks.",
        )
        parser.add_argument(
            f"--{prefix}-failure-threshold",
            type=int,
            default=3,
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


class BaseHealthChecker(abc.ABC):
    @property
    @abc.abstractmethod
    def status(self) -> TriState: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def pause(self) -> None: ...

    @abc.abstractmethod
    def resume(self) -> None: ...


class SimpleHealthChecker(BaseHealthChecker):
    """Periodic async health checker. Calls *check_fn*; reports result via *on_result*.

    After each ``resume()``, waits ``first_wait`` seconds before the first check
    (matching ``RolloutHealthMonitor._need_first_wait`` semantics).
    """

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        on_result: Callable[[bool], None] | None = None,
        config: SimpleHealthCheckerConfig,
        clock: Clock | None = None,
    ) -> None:
        self._name = name
        self._check_fn = check_fn
        self._on_result = on_result
        self._config = config
        self._clock = clock or RealClock()

        self._status = TriState.UNKNOWN
        self._paused: bool = False
        self._need_first_wait: bool = True
        self._consecutive_failures: int = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def status(self) -> TriState:
        return self._status

    async def start(self) -> None:
        if self._task is not None:
            return
        log_structured(logger.info, op="health", phase="start", name=self._name)
        self._task = asyncio.create_task(self._loop())
        await asyncio.sleep(0)

    def stop(self) -> None:
        if self._task is not None:
            log_structured(logger.info, op="health", phase="stop", name=self._name)
            self._task.cancel()
            self._task = None
        self._status = TriState.UNKNOWN

    def pause(self) -> None:
        log_structured(logger.info, op="health", phase="pause", name=self._name)
        self._paused = True
        self._status = TriState.UNKNOWN

    def resume(self) -> None:
        log_structured(logger.info, op="health", phase="resume", name=self._name)
        self._paused = False
        self._need_first_wait = True
        self._status = TriState.UNKNOWN
        self._consecutive_failures = 0

    async def _loop(self) -> None:
        while True:
            if self._need_first_wait:
                self._need_first_wait = False
                log_structured(
                    logger.info, op="health", phase="first_wait", name=self._name, wait_s=self._config.first_wait
                )
                await self._clock.sleep(self._config.first_wait)

            if not self._paused:
                success = False
                try:
                    await asyncio.wait_for(self._check_fn(), timeout=self._config.timeout)
                    success = True
                except Exception:
                    log_structured(logger.error, op="health", phase="check_failed", name=self._name, exc_info=True)

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
                            logger.error, op="health", phase="on_result_failed", name=self._name, exc_info=True
                        )

            await self._clock.sleep(self._config.interval)


class NoopHealthChecker(BaseHealthChecker):
    @property
    def status(self) -> TriState:
        return TriState.UNKNOWN

    async def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


# TODO: should move when Rollout FT is implemented
def create_rollout_cell_health_checker(
    *,
    cell_id: str,
    get_engines: Callable[[], list[Any]],
    config: SimpleHealthCheckerConfig,
    on_result: Callable[[bool], None] | None = None,
) -> SimpleHealthChecker:

    async def _check() -> None:
        engines = get_engines()
        if not engines:
            raise RuntimeError("No engines")

        lead_engine = engines[0]
        if lead_engine is None:
            raise RuntimeError("Lead engine is None")

        await lead_engine.health_generate.remote()

    # Preserve the pre-debounce rollout semantics for now: a single failed check
    # reports unhealthy immediately. The trainer cell checker uses the default
    # failure_threshold; tuning rollout debouncing is left to Rollout FT work.
    config = config.model_copy(update={"failure_threshold": 1})

    return SimpleHealthChecker(
        name=f"rollout-cell-{cell_id}",
        check_fn=_check,
        on_result=on_result,
        config=config,
    )
