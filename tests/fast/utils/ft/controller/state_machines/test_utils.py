"""Tests for state_machines/utils.py (safe_notify)."""

from __future__ import annotations

import logging

import pytest

from miles.utils.ft.controller.state_machines.utils import safe_notify


class _FakeNotifier:
    def __init__(self, *, fail: bool = False, fail_count: int = 0) -> None:
        self.calls: list[str] = []
        self._fail = fail
        self._fail_count = fail_count
        self._attempt = 0

    async def send(self, *, title: str, content: str, severity: str = "critical") -> None:
        self._attempt += 1
        self.calls.append(title)
        if self._fail:
            raise ConnectionError("notifier unavailable")
        if self._fail_count > 0 and self._attempt <= self._fail_count:
            raise ConnectionError("transient failure")


class TestSafeNotify:
    @pytest.mark.anyio
    async def test_success(self) -> None:
        notifier = _FakeNotifier()
        await safe_notify(notifier, title="Alert", content="body")
        assert notifier.calls == ["Alert"]

    @pytest.mark.anyio
    async def test_none_notifier_is_noop(self) -> None:
        await safe_notify(None, title="Alert", content="body")

    @pytest.mark.anyio
    async def test_does_not_propagate_exception(self) -> None:
        notifier = _FakeNotifier(fail=True)
        await safe_notify(notifier, title="Alert", content="body")

    @pytest.mark.anyio
    async def test_logs_error_on_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        notifier = _FakeNotifier(fail=True)
        with caplog.at_level(logging.ERROR, logger="miles.utils.ft.controller.state_machines.utils"):
            await safe_notify(notifier, title="Alert", content="body")
        assert any("safe_notify_failed" in msg for msg in caplog.messages)


class TestSafeNotifyRetry:
    """safe_notify previously had no retry — a single transient failure
    (webhook timeout, 429, network blip) was immediately swallowed and
    the state machine would proceed as if notification succeeded."""

    @pytest.mark.anyio
    async def test_transient_failure_then_success_delivers(self) -> None:
        notifier = _FakeNotifier(fail_count=1)
        await safe_notify(notifier, title="Alert", content="body")

        assert len(notifier.calls) == 2
        assert notifier.calls[-1] == "Alert"

    @pytest.mark.anyio
    async def test_all_retries_fail_still_swallows_exception(self) -> None:
        notifier = _FakeNotifier(fail=True)
        await safe_notify(notifier, title="Alert", content="body")

    @pytest.mark.anyio
    async def test_all_retries_fail_logs_final_error(self, caplog: pytest.LogCaptureFixture) -> None:
        notifier = _FakeNotifier(fail=True)
        with caplog.at_level(logging.ERROR, logger="miles.utils.ft.controller.state_machines.utils"):
            await safe_notify(notifier, title="Alert", content="body")
        assert any("safe_notify_failed" in msg for msg in caplog.messages)

    @pytest.mark.anyio
    async def test_retries_exhaust_with_correct_attempt_count(self) -> None:
        notifier = _FakeNotifier(fail=True)
        await safe_notify(notifier, title="Alert", content="body")

        assert len(notifier.calls) == 3
