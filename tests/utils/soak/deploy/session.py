import asyncio

from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.recipes.gsm8k import Gsm8kRun, execute_gsm8k_session

LauncherChain = asyncio.Queue[asyncio.Task[LaunchOutcome]]


async def execute_hot_restart_session(run: Gsm8kRun, *, chain: LauncherChain) -> None:
    outcome = await execute_gsm8k_session(run)
    while outcome == LaunchOutcome.REPLACED:
        assert not chain.empty(), "A launcher was replaced, and no take-over launcher succeeded it"
        outcome = await chain.get_nowait()
