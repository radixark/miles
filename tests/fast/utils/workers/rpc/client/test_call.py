from __future__ import annotations

import math
import time
from typing import Any

import httpx
import pytest

from miles.utils.workers.rpc.client.call import RpcCall, _CallStillPendingError
from miles.utils.workers.rpc.client.handle import DEFAULT_CALL_TIMEOUT_SECONDS
from miles.utils.workers.rpc.client.misc import RETRY_INITIAL_DELAY_SECONDS, RetryableResponseError
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec, collect_rpc_method_specs


class _Worker:
    def demo(self, a: int) -> int:
        return a


class _FailingTransport:
    async def request(self, method: str, path: str, *, seconds: float, response_model: type, **kwargs: Any) -> Any:
        raise httpx.ConnectError("injected transport failure")


class _RaisingTransport:
    def __init__(self, error: Exception) -> None:
        self.requests = 0
        self._error = error

    async def request(self, method: str, path: str, *, seconds: float, response_model: type, **kwargs: Any) -> Any:
        self.requests += 1
        raise self._error


def _spec() -> RpcMethodSpec:
    return collect_rpc_method_specs(_Worker)["demo"]


def _make_call(*, call_timeout_seconds: float, transport: Any | None = None) -> RpcCall:
    return RpcCall(
        spec=_spec(),
        kwargs={"a": 1},
        worker_cls_name="_Worker",
        transport=transport if transport is not None else _FailingTransport(),
        call_timeout_seconds=call_timeout_seconds,
    )


class TestCallTimeoutValidation:
    @pytest.mark.parametrize("call_timeout_seconds", [float("inf"), float("nan"), 0.0, -1.0])
    def test_a_non_finite_or_non_positive_call_timeout_is_rejected(self, call_timeout_seconds: float) -> None:
        """The retry loop compares the remaining budget against the deadline, so such a value would never expire."""
        with pytest.raises(AssertionError, match="call_timeout_seconds"):
            _make_call(call_timeout_seconds=call_timeout_seconds)

    def test_a_finite_call_timeout_is_accepted(self) -> None:
        """A normal deadline is the whole point of the parameter and must stay usable."""
        assert _make_call(call_timeout_seconds=30.0) is not None

    def test_a_negative_infinite_call_timeout_is_rejected(self) -> None:
        """Minus infinity is neither finite nor positive, and would make every remaining budget comparison nonsense."""
        with pytest.raises(AssertionError, match="call_timeout_seconds"):
            _make_call(call_timeout_seconds=float("-inf"))

    def test_the_rejection_message_names_the_offending_value(self) -> None:
        """An operator reading the crash needs to see which configured timeout was refused."""
        with pytest.raises(AssertionError, match="inf"):
            _make_call(call_timeout_seconds=float("inf"))

    def test_the_default_handle_call_timeout_is_a_usable_finite_deadline(self) -> None:
        """The shipped default must satisfy the finite-deadline check, or every rpc call would assert at runtime."""
        assert math.isfinite(DEFAULT_CALL_TIMEOUT_SECONDS)
        assert DEFAULT_CALL_TIMEOUT_SECONDS > 0
        assert _make_call(call_timeout_seconds=DEFAULT_CALL_TIMEOUT_SECONDS) is not None


class TestPollBackoffAgainstTheRemainingBudget:
    async def test_a_transport_failure_sleeps_no_longer_than_the_remaining_budget(
        self, recorded_sleeps: list[float]
    ) -> None:
        """Sleeping the full backoff past the deadline burns the caller's budget on a retry that can never run."""
        call = _make_call(call_timeout_seconds=30.0)

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(0.05)

        assert recorded_sleeps == [0.05]

    async def test_a_transport_failure_sleeps_the_full_backoff_while_budget_remains(
        self, recorded_sleeps: list[float]
    ) -> None:
        """With plenty of budget left the backoff must not be shortened, or the retry loop would spin."""
        call = _make_call(call_timeout_seconds=30.0)

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(30.0)

        assert recorded_sleeps == [RETRY_INITIAL_DELAY_SECONDS]

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("connect refused"),
            httpx.ReadError("connection reset"),
            httpx.PoolTimeout("no free connection"),
            RetryableResponseError("GET /v1/calls/x returned 503"),
        ],
        ids=["connect_error", "read_error", "pool_timeout", "retryable_response"],
    )
    async def test_every_retryable_poll_failure_caps_its_backoff_at_the_remaining_budget(
        self, error: Exception, recorded_sleeps: list[float]
    ) -> None:
        """Each retryable poll failure shares one backoff, so all of them must respect the remaining budget."""
        call = _make_call(call_timeout_seconds=30.0, transport=_RaisingTransport(error))

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(0.02)

        assert recorded_sleeps == [0.02]

    async def test_an_exhausted_budget_makes_the_backoff_sleep_zero(self, recorded_sleeps: list[float]) -> None:
        """Once the budget is gone the backoff must collapse to nothing instead of a whole extra second."""
        call = _make_call(call_timeout_seconds=30.0)

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(0.0)

        assert recorded_sleeps == [0.0]

    async def test_a_long_poll_timeout_does_not_sleep_at_all(self, recorded_sleeps: list[float]) -> None:
        """A long poll already consumed its wait, so retrying it must not add any backoff on top."""
        call = _make_call(call_timeout_seconds=30.0, transport=_RaisingTransport(TimeoutError("long poll expired")))

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(30.0)

        assert recorded_sleeps == []


class TestPollLoopDeadline:
    async def test_the_poll_loop_gives_up_close_to_the_call_timeout_when_every_poll_fails(self) -> None:
        """A fixed one-second backoff overruns a short deadline, so the caller waits far past the budget it set."""
        transport = _RaisingTransport(httpx.ConnectError("injected transport failure"))
        call = _make_call(call_timeout_seconds=0.1, transport=transport)

        started_at = time.monotonic()
        with pytest.raises(TimeoutError):
            await call._poll_until_done()
        elapsed = time.monotonic() - started_at

        assert transport.requests >= 1
        assert elapsed < RETRY_INITIAL_DELAY_SECONDS / 2
