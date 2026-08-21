"""Sync retry helper used by rollout engine bringup.

Stdlib-only: the bringup call site supplies ``should_retry`` as
``isinstance(..., ActorUnavailableError)``. These tests drive the same
predicate shape with stand-in exception types.
"""

from __future__ import annotations

import pytest

from miles.utils.retry_utils import retry_sync


class Transient(Exception):
    """Retryable (stands in for ray.exceptions.ActorUnavailableError)."""


class Permanent(Exception):
    """Not retryable (stands in for ray.exceptions.ActorDiedError)."""


def _retry_transient(exc: Exception) -> bool:
    return isinstance(exc, Transient)


def test_recovers_after_retryable_failures():
    sentinel = object()
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient("temporarily unavailable")
        return sentinel

    result = retry_sync(thunk, should_retry=_retry_transient, what="test", sleep_fn=lambda _s: None)

    assert result is sentinel
    assert calls["n"] == 3


def test_exhaustion_reraises_last_error():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise Transient("still unavailable")

    with pytest.raises(Transient):
        retry_sync(thunk, should_retry=_retry_transient, what="test", sleep_fn=lambda _s: None, max_attempts=3)

    assert calls["n"] == 3


def test_non_retryable_error_propagates_immediately():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise Permanent("the actor died unexpectedly")

    with pytest.raises(Permanent):
        retry_sync(thunk, should_retry=_retry_transient, what="test", sleep_fn=lambda _s: None, max_attempts=3)

    assert calls["n"] == 1


def test_success_on_first_attempt_calls_thunk_once():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        return "ok"

    assert retry_sync(thunk, should_retry=_retry_transient, what="test") == "ok"
    assert calls["n"] == 1


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_never_intercepted(exc_type):
    calls = {"n": 0}
    seen_by_predicate: list[Exception] = []

    def retry_everything(exc: Exception) -> bool:
        seen_by_predicate.append(exc)
        return True

    def thunk():
        calls["n"] += 1
        raise exc_type()

    with pytest.raises(exc_type):
        retry_sync(thunk, should_retry=retry_everything, what="test", sleep_fn=lambda _s: None, max_attempts=3)
    assert calls["n"] == 1
    assert seen_by_predicate == []
