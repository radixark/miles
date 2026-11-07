import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _unexpected_backend_call(*args: object, **kwargs: object) -> None:
    raise AssertionError("Rejected requests must not initialize the TorchTitan backend")


@pytest.fixture
def titan_actor_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    config = ModuleType("miles.backends.torchtitan_utils.config")
    config.build_trainer_config = _unexpected_backend_call
    trainer = ModuleType("miles.backends.torchtitan_utils.trainer")
    trainer.TitanTrainer = _unexpected_backend_call
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setitem(sys.modules, trainer.__name__, trainer)

    path = Path(__file__).resolve().parents[4] / "miles/backends/torchtitan_utils/actor.py"
    spec = importlib.util.spec_from_file_location("_test_titan_actor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
