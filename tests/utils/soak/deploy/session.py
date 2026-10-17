import asyncio
from collections.abc import Awaitable

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.core.utils import note_launch_outcome

LauncherChain = asyncio.Queue[asyncio.Task[LaunchOutcome]]


def start_launch(
    launching: Awaitable[None], *, event_log: EventLog, request_id: str, chain: LauncherChain
) -> asyncio.Task[LaunchOutcome]:
    launcher = asyncio.create_task(
        note_launch_outcome(event_log=event_log, request_id=request_id, launching=launching)
    )
    chain.put_nowait(launcher)
    return launcher


async def follow_launchers(first: Awaitable[LaunchOutcome], *, chain: LauncherChain) -> None:
    outcome = await first
    while outcome == LaunchOutcome.REPLACED:
        assert not chain.empty(), "A launcher was replaced, and no take-over launcher succeeded it"
        outcome = await chain.get_nowait()
    while not chain.empty():
        await chain.get_nowait()
