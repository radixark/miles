"""An agent function that reports its sandbox, kept apart from ``agent_function_stubs``
so the other ``subproc`` tests do not import the dashboard in every child process."""

import asyncio

from miles.dashboard.hooks import report_sandbox


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
