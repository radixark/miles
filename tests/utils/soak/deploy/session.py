import asyncio

from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.recipes.gsm8k import Gsm8kRun

LauncherChain = asyncio.Queue[asyncio.Task[LaunchOutcome]]


async def execute_hot_restart_session(run: Gsm8kRun, *, chain: LauncherChain) -> None:
    raise NotImplementedError
