"""Agent functions that report their sandbox, kept apart from ``agent_function_stubs``
so the other ``subproc`` tests do not import the dashboard or the run observers in every child process."""

import asyncio

from miles.dashboard.hooks import report_sandbox
from miles.rollout.agentic import sandbox_reports


async def report_sandbox_health(sandbox_key: str, **kwargs) -> str:
    report_sandbox(sandbox_key, provider="local")
    report_sandbox(sandbox_key, status="finished")
    return "reported"


async def report_then_hang(sandbox_key: str, **kwargs) -> None:
    report_sandbox(sandbox_key, provider="local")
    try:
        await asyncio.sleep(3600)
    finally:
        report_sandbox(sandbox_key, status="finished")


async def report_to_the_dashboard_and_the_observers(sandbox_key: str, **kwargs) -> str:
    report_sandbox(sandbox_key, provider="dashboard")
    sandbox_reports.report_sandbox(sandbox_key, provider="observers")
    return "reported"
