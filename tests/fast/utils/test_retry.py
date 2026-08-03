"""Unit tests for ``miles.utils.retry`` (stdlib-only — no Ray, no network).

They exercise the generic retry mechanism with a caller-supplied
``should_retry`` predicate, driven the same way the rollout-bringup call site
does (isinstance against the Ray exception hierarchy).
"""

import types

import pytest

from miles.utils.retry import retry_with_backoff


class Transient(Exception):
    """Retryable (stands in for ray.exceptions.ActorUnavailableError)."""


class Permanent(Exception):
    """Not retryable (stands in for ray.exceptions.ActorDiedError)."""


def _retry_transient(exc: Exception) -> bool:
    return isinstance(exc, Transient)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("miles.utils.retry.time", types.SimpleNamespace(sleep=lambda _s: None))


def test_recovers_after_retryable_failures():
    """A thunk that fails retryably N times then succeeds is retried to success."""
    sentinel = object()
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] < 3:  # fail on attempts 1 and 2, succeed on 3
            raise Transient("temporarily unavailable")
        return sentinel

    result = retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert result is sentinel
    assert calls["n"] == 3


def test_exhaustion_reraises_last_error():
    """A thunk that always fails retryably re-raises after exactly max_retries attempts."""
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise Transient("still unavailable")

    with pytest.raises(Transient):
        retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert calls["n"] == 3  # no swallow: exhausted then re-raised


def test_non_retryable_error_propagates_immediately():
    """An error the predicate rejects propagates on the FIRST attempt — never masked."""
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise Permanent("the actor died unexpectedly")

    with pytest.raises(Permanent):
        retry_with_backoff(thunk, should_retry=_retry_transient, what="test", max_retries=3)

    assert calls["n"] == 1


def test_success_on_first_attempt_calls_thunk_once():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        return "ok"

    assert retry_with_backoff(thunk, should_retry=_retry_transient, what="test") == "ok"
    assert calls["n"] == 1


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_never_intercepted(exc_type):
    """KeyboardInterrupt/SystemExit propagate at once, without consulting the predicate.

    The handler catches ``Exception``, so control-flow exceptions cannot be
    retried no matter what ``should_retry`` returns — pinned here with a
    predicate that retries everything it is shown.
    """
    calls = {"n": 0}
    seen_by_predicate: list[Exception] = []

    def retry_everything(exc: Exception) -> bool:
        seen_by_predicate.append(exc)
        return True

    def thunk():
        calls["n"] += 1
        raise exc_type()

    with pytest.raises(exc_type):
        retry_with_backoff(thunk, should_retry=retry_everything, what="test", max_retries=3)
    assert calls["n"] == 1
    assert seen_by_predicate == []
