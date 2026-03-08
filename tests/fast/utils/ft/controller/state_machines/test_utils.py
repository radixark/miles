"""Tests for state_machines/utils.py (safe_notify)."""

from __future__ import annotations

import logging

import pytest

from miles.utils.ft.controller.state_machines.utils import safe_notify


class _FakeNotifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    async def send(self, *, title: str, content: str, severity: str = "critical") -> None:
        self.calls.append(title)
        if self._fail:
            raise ConnectionError("notifier unavailable")


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
