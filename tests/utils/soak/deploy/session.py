import asyncio
from dataclasses import dataclass, field

from tests.utils.soak.recipes.gsm8k import Gsm8kRun, execute_gsm8k_session


@dataclass
class LauncherChain:
    tasks: asyncio.Queue[asyncio.Task[str]] = field(default_factory=asyncio.Queue)

    def append(self, task: asyncio.Task[str]) -> None:
        self.tasks.put_nowait(task)

    def pop(self) -> asyncio.Task[str]:
        assert not self.tasks.empty(), "A launcher was replaced, and no take-over launcher succeeded it"
        return self.tasks.get_nowait()


async def execute_hot_restart_session(run: Gsm8kRun, *, chain: LauncherChain) -> None:
    launcher = asyncio.create_task(execute_gsm8k_session(run))
    try:
        while await launcher == "replaced":
            launcher = chain.pop()
    finally:
        if not launcher.done():
            launcher.cancel()
