import importlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@dataclass(eq=False)
class _LocalResult:
    value: Any


class _LocalMethod:
    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function

    def remote(self, *args: Any, **kwargs: Any) -> _LocalResult:
        return _LocalResult(self.function(*args, **kwargs))


class _LocalConversionActor:
    def __init__(self, worker_type: type) -> None:
        self.worker_type = worker_type

    def options(self, **kwargs: Any) -> "_LocalConversionActor":
        return self

    def remote(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        worker = self.worker_type(*args, **kwargs)
        return SimpleNamespace(convert=_LocalMethod(worker.convert))


@pytest.fixture
def local_ray_converter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    node_id = "01" * 28
    monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)
    converter = importlib.import_module("tools.convert_torch_dist_to_hf_ray")
    monkeypatch.setattr(converter, "make_conversion_actor", lambda: _LocalConversionActor(converter.ConversionWorker))
    monkeypatch.setattr(converter, "initialize_ray", lambda: None)
    monkeypatch.setattr(converter.ray, "nodes", lambda: [{"NodeID": node_id, "Alive": True}])
    monkeypatch.setattr(converter.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: node_id))
    monkeypatch.setattr(converter.ray, "put", lambda value: value)
    monkeypatch.setattr(converter.ray, "get", lambda ref: ref.value)
    monkeypatch.setattr(converter.ray, "wait", lambda refs, **kwargs: ([refs[0]], refs[1:]))
    return converter
