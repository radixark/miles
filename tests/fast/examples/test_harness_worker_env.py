import asyncio
import fnmatch
import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestServer

from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, provider_env_names
from miles.rollout.agentic.worker import create_app
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.test_utils import agent_function_stubs as stubs

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
_ENV_READ = re.compile(r'(?:getenv|environ\.get|_env_flag|_env_int|_allowed_hosts)\(\s*"(\w+)"|environ\["(\w+)"\]')
_CREDENTIALS = {name for spec in PROVIDER_CREDENTIALS.values() for name in spec["key_env_vars"]}
_ROLLOUT_SIDE_ONLY = {"HARBOR_ADMIN_SECRET"}
_OPENENV_LOOP = ("openenv_agent_function.py", "openenv_sandbox_common.py", "tb2_sandbox_recipe.py")


def _env_reads(directory: Path, sources: tuple[str, ...]) -> set[str]:
    return {a or b for source in sources for a, b in _ENV_READ.findall((directory / source).read_text())}


_HARNESSES = [
    ("experimental/harbor", "harbor_agent_function", ("harbor_agent_function.py",)),
    ("experimental/nemo-gym", "nemogym_agent_function", ("nemogym_agent_function.py",)),
    ("experimental/nemo-gym-workspace-assistant", "workplace_agent", ("workplace_agent.py",)),
    ("swe-agent-harbor-docker", "swe_agent_function", ("swe_agent_function.py",)),
    ("experimental/openenv", "openenv_agent_function", ("openenv_agent_function.py",)),
    (
        "experimental/openenv",
        "openenv_e2b_agent_function",
        (*_OPENENV_LOOP, "openenv_e2b_agent_function.py", "tb2_sandbox_e2b.py"),
    ),
    (
        "experimental/openenv",
        "openenv_daytona_agent_function",
        (*_OPENENV_LOOP, "openenv_daytona_agent_function.py", "tb2_sandbox_daytona.py"),
    ),
    (
        "experimental/openenv",
        "openenv_modal_agent_function",
        (*_OPENENV_LOOP, "openenv_modal_agent_function.py", "tb2_sandbox_modal.py"),
    ),
]


@pytest.mark.parametrize(("directory", "module", "sources"), _HARNESSES)
def test_a_harness_declares_every_env_var_it_reads(monkeypatch, directory, module, sources):
    monkeypatch.syspath_prepend(_EXAMPLES / directory)
    patterns = (*provider_env_names(), *importlib.import_module(module).WORKER_ENV)

    reads = _env_reads(_EXAMPLES / directory, sources) - _CREDENTIALS - _ROLLOUT_SIDE_ONLY

    assert reads
    assert [name for name in reads if not any(fnmatch.fnmatchcase(name, p) for p in patterns)] == []


@pytest.mark.parametrize(("directory", "module", "sources"), _HARNESSES)
def test_a_harness_env_reaches_its_agent_on_a_worker(monkeypatch, directory, module, sources):
    monkeypatch.syspath_prepend(_EXAMPLES / directory)
    reads = sorted(_env_reads(_EXAMPLES / directory, sources) - _CREDENTIALS - _ROLLOUT_SIDE_ONLY)
    for name in reads:
        monkeypatch.setenv(name, f"set-{name}")
    args = SimpleNamespace(
        custom_agent_function_mode="worker",
        custom_agent_function_worker="http://unused:1",
        custom_agent_function_path=CustomFunctionConfig(path=f"{module}.run"),
        custom_agent_function_worker_env=[],
    )
    extra_env = agentic_tool_call._worker_options(args)["extra_env"]

    async def scenario():
        async with TestServer(create_app()) as server:
            return await call_agent_function(
                stubs.read_env, mode="worker", worker=str(server.make_url("")), extra_env=extra_env, names=reads
            )

    for name in reads:
        monkeypatch.delenv(name)
    assert asyncio.run(scenario()) == {name: f"set-{name}" for name in reads}
