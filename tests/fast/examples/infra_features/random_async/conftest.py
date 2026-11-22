import importlib
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def random_rollout(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    directory = Path(__file__).resolve().parents[5] / "examples/infra_features/random_async"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("examples.infra_features.random_async.random_async_rollout")
