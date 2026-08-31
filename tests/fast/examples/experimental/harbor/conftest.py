"""Put the example on sys.path and stand in for the ``harbor`` package.

The tests run where Harbor is not installed (CPU CI), so a minimal fake of the
four Harbor classes the agent function touches is registered in sys.modules
before the example is imported. Only their constructor signatures matter here:
the tests assert on what the agent function builds and how it maps results.
"""

import enum
import sys
import types
from types import SimpleNamespace

import pytest

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


class _EnvironmentType(str, enum.Enum):
    DOCKER = "docker"
    DAYTONA = "daytona"
    E2B = "e2b"
    MODAL = "modal"


def _record(name):
    def ctor(**kwargs):
        return SimpleNamespace(_kind=name, **kwargs)

    return ctor


class FakeTrial:
    """Records the config it was created with; ``run`` returns a scripted result."""

    created: list = []
    result = None
    run_delay_s = 0.0

    def __init__(self, config):
        self.config = config
        self.paths = SimpleNamespace(trial_dir=f"/tmp/harbor_trials/{config.task.path.name}")

    @classmethod
    async def create(cls, config):
        trial = cls(config)
        cls.created.append(trial)
        return trial

    async def run(self):
        import asyncio

        if self.run_delay_s:
            await asyncio.sleep(self.run_delay_s)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def fake_harbor(monkeypatch):
    harbor = types.ModuleType("harbor")
    models = types.ModuleType("harbor.models")
    env_type = types.ModuleType("harbor.models.environment_type")
    env_type.EnvironmentType = _EnvironmentType
    trial_models = types.ModuleType("harbor.models.trial")
    config = types.ModuleType("harbor.models.trial.config")
    for name in ("AgentConfig", "TaskConfig", "TrialConfig", "VerifierConfig", "EnvironmentConfig"):
        setattr(config, name, _record(name))
    trial_pkg = types.ModuleType("harbor.trial")
    trial_mod = types.ModuleType("harbor.trial.trial")
    trial_mod.Trial = FakeTrial
    for mod in (harbor, models, env_type, trial_models, config, trial_pkg, trial_mod):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    FakeTrial.created = []
    FakeTrial.result = None
    FakeTrial.run_delay_s = 0.0
    yield FakeTrial
