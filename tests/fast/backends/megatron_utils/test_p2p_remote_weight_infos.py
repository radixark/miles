import importlib
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import pytest

_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils"


@dataclass
class _FakeServerArgs:
    model_path: str | None = None


_EXTERNAL_SDK_ATTRIBUTES = {
    "mooncake.engine": {"TransferEngine": object},
    "sglang.srt.server_args": {"ServerArgs": _FakeServerArgs},
}


class _FakeRolloutEngine:
    def __init__(self, engine_index: int):
        self._engine_index = engine_index
        self.calls: list[tuple[str, dict]] = []

    async def get_remote_instance_transfer_engine_info(self, rank: int):
        self.calls.append(("get_remote_instance_transfer_engine_info", {"rank": rank}))
        return f"session-{self._engine_index}-{rank}", {f"weight-{rank}": (0x1000 + rank, 4, 2)}

    async def get_parallelism_info(self, rank: int):
        self.calls.append(("get_parallelism_info", {"rank": rank}))
        return {"tp_rank": rank}

    async def get_server_info(self):
        self.calls.append(("get_server_info", {}))
        return {"model_path": f"/model/{self._engine_index}"}


@contextmanager
def _stubbed_missing_external_sdks():
    missing = object()
    saved_modules: dict[str, object] = {}
    saved_attributes: list[tuple[ModuleType, str, object]] = []
    for module_name, attributes in _EXTERNAL_SDK_ATTRIBUTES.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            saved_modules[name] = sys.modules.get(name, missing)
            if depth == len(parts) or name not in sys.modules:
                module = ModuleType(name)
                if depth < len(parts):
                    module.__path__ = []
                sys.modules[name] = module
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                attribute = parts[depth - 1]
                saved_attributes.append((parent, attribute, getattr(parent, attribute, missing)))
                setattr(parent, attribute, sys.modules[name])
        for attribute, value in attributes.items():
            setattr(sys.modules[module_name], attribute, value)

    try:
        yield
    finally:
        for parent, attribute, value in reversed(saved_attributes):
            if value is missing:
                delattr(parent, attribute)
            else:
                setattr(parent, attribute, value)
        for name, module in reversed(saved_modules.items()):
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def p2p_transfer_utils():
    package_name, attribute = _MODULE.rsplit(".", 1)
    package = importlib.import_module(package_name)
    missing = object()
    saved_module = sys.modules.get(_MODULE, missing)
    saved_attribute = getattr(package, attribute, missing)

    with _stubbed_missing_external_sdks():
        sys.modules.pop(_MODULE, None)
        if hasattr(package, attribute):
            delattr(package, attribute)
        try:
            yield importlib.import_module(_MODULE)
        finally:
            sys.modules.pop(_MODULE, None)
            if saved_module is not missing:
                sys.modules[_MODULE] = saved_module
            if saved_attribute is missing:
                if hasattr(package, attribute):
                    delattr(package, attribute)
            else:
                setattr(package, attribute, saved_attribute)


def _make_targets(module, pairs: list[tuple[int, int]]) -> list:
    return [
        module.TransferTaskP2PMeta(engine_ind=engine_ind, engine_rank=engine_rank, source_shard=source_shard)
        for source_shard, (engine_ind, engine_rank) in enumerate(pairs)
    ]


def _query(module, engines: list[_FakeRolloutEngine], pairs: list[tuple[int, int]]):
    return module.query_remote_weight_infos(engines, _make_targets(module, pairs))


class TestQueryRemoteWeightInfos:
    """Remote-info discovery over the rollout engines' HTTP API."""

    def test_repeated_targets_are_queried_once_each(self, p2p_transfer_utils):
        """The same engine rank appears once per source shard, and re-querying it wastes round trips."""
        engines = [_FakeRolloutEngine(0), _FakeRolloutEngine(1)]

        _query(p2p_transfer_utils, engines, [(0, 0), (0, 1), (0, 0), (1, 0)])

        assert Counter(name for name, _kwargs in engines[0].calls) == Counter(
            {
                "get_remote_instance_transfer_engine_info": 2,
                "get_parallelism_info": 2,
                "get_server_info": 2,
            }
        )
        assert sorted(kwargs["rank"] for name, kwargs in engines[0].calls if name == "get_parallelism_info") == [0, 1]
        assert [name for name, _kwargs in engines[1].calls] == [
            "get_remote_instance_transfer_engine_info",
            "get_parallelism_info",
            "get_server_info",
        ]

    def test_the_returned_maps_agree_on_every_session_id(self, p2p_transfer_utils):
        """Every weight, parallelism, and converted server-args entry must match its session ID."""
        engines = [_FakeRolloutEngine(0), _FakeRolloutEngine(1)]

        weight_infos, targets_to_session_id, session_id_to_server_args = _query(
            p2p_transfer_utils, engines, [(0, 0), (0, 1), (1, 0)]
        )

        assert targets_to_session_id == {
            (0, 0): "session-0-0",
            (0, 1): "session-0-1",
            (1, 0): "session-1-0",
        }
        assert weight_infos == {
            "session-0-0": ({"weight-0": (0x1000, 4, 2)}, {"tp_rank": 0}),
            "session-0-1": ({"weight-1": (0x1001, 4, 2)}, {"tp_rank": 1}),
            "session-1-0": ({"weight-0": (0x1000, 4, 2)}, {"tp_rank": 0}),
        }
        assert all(
            isinstance(server_args, p2p_transfer_utils.ServerArgs)
            for server_args in session_id_to_server_args.values()
        )
        assert {
            session_id: server_args.model_path for session_id, server_args in session_id_to_server_args.items()
        } == {
            "session-0-0": "/model/0",
            "session-0-1": "/model/0",
            "session-1-0": "/model/1",
        }
