"""Sandbox reports made inside a ``subproc`` agent call reach the calling process."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.dashboard import backend
from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils import observers
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils import sandbox_report_stubs


class _SandboxRecorder(observers.Observer):
    def __init__(self):
        self.reports: list[observers.SandboxReport] = []

    def sandbox_report(self, report):
        self.reports.append(report)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _SandboxRecorder:
    recorder = _SandboxRecorder()
    installed = [SimpleNamespace(name="recorder", load=lambda: lambda: recorder)]
    monkeypatch.setattr(observers, "entry_points", lambda group: installed)
    SingletonMeta._instances.pop(observers.Observers, None)
    return recorder


def test_each_report_reaches_the_receiver_of_the_report_sandbox_that_made_it(
    recorder: _SandboxRecorder, monkeypatch: pytest.MonkeyPatch
):
    """Agent code reports through the dashboard's or the observers' ``report_sandbox``; each must reach its own."""
    pushed: list[list] = []
    monkeypatch.setattr(
        backend, "_handle", SimpleNamespace(push_sandbox_reports=SimpleNamespace(remote=pushed.append))
    )

    result = asyncio.run(
        call_agent_function(
            sandbox_report_stubs.report_to_the_dashboard_and_the_observers,
            mode="subproc",
            on_child_reports=agentic_tool_call._route_sandbox_reports,
            sandbox_key="a",
        )
    )

    assert result == "reported"
    assert [report.provider for batch in pushed for report in batch] == ["dashboard"]
    assert [report["provider"] for report in recorder.reports] == ["observers"]
