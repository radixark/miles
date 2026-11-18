import asyncio
import os
from pathlib import Path

import harbor_agent_function as haf
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trial.config import EnvironmentConfig


def _mark(name: str) -> None:
    Path(os.environ["BLOCKING_ENV_MARKERS"], name).touch()


def _is_marked(name: str) -> bool:
    return Path(os.environ["BLOCKING_ENV_MARKERS"], name).exists()


class BlockingEnvironment(BaseEnvironment):
    @staticmethod
    def type() -> str:
        return "blocking"

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool) -> None:
        _mark("started")

    async def stop(self, delete: bool) -> None:
        _mark(f"stopped_delete_{delete}")

    async def upload_file(self, source_path, target_path) -> None:
        pass

    async def upload_dir(self, source_dir, target_dir) -> None:
        pass

    async def download_file(self, source_path, target_path) -> None:
        pass

    async def download_dir(self, source_dir, target_dir) -> None:
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        if "test.sh" in command and not _is_marked("verifier_running"):
            _mark("verifier_running")
            await asyncio.sleep(3600)
        return ExecResult(stdout="", stderr="", return_code=0)


async def run_trial(**kwargs) -> dict:
    haf._environment_config = lambda: EnvironmentConfig(import_path=f"{__name__}:BlockingEnvironment", delete=True)
    return await haf.run(**kwargs)
