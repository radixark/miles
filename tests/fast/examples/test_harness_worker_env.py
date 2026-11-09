import fnmatch
import importlib
import re
from pathlib import Path

import pytest

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, provider_env_names

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
_ENV_READ = re.compile(r'(?:getenv|environ\.get|_env_flag|_env_int|_allowed_hosts)\(\s*"(\w+)"|environ\["(\w+)"\]')
_CREDENTIALS = {name for spec in PROVIDER_CREDENTIALS.values() for name in spec["key_env_vars"]}
_ROLLOUT_SIDE_ONLY = {"HARBOR_ADMIN_SECRET"}
_OPENENV_LOOP = ("openenv_agent_function.py", "openenv_sandbox_common.py", "tb2_sandbox_recipe.py")


def _env_reads(directory: Path, sources: tuple[str, ...]) -> set[str]:
    return {a or b for source in sources for a, b in _ENV_READ.findall((directory / source).read_text())}


@pytest.mark.parametrize(
    ("directory", "module", "sources"),
    [
        ("experimental/harbor", "harbor_agent_function", ("harbor_agent_function.py",)),
        ("experimental/nemo-gym", "nemogym_agent_function", ("nemogym_agent_function.py",)),
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
    ],
)
def test_a_harness_declares_every_env_var_it_reads(monkeypatch, directory, module, sources):
    monkeypatch.syspath_prepend(_EXAMPLES / directory)
    patterns = (*provider_env_names(), *importlib.import_module(module).WORKER_ENV)

    reads = _env_reads(_EXAMPLES / directory, sources) - _CREDENTIALS - _ROLLOUT_SIDE_ONLY

    assert reads
    assert [name for name in reads if not any(fnmatch.fnmatchcase(name, p) for p in patterns)] == []
