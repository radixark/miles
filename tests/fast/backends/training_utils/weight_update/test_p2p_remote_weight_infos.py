import importlib
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import pytest

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils"


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


class _JsonRolloutEngine(_FakeRolloutEngine):
    async def get_remote_instance_transfer_engine_info(self, rank: int):
        session_id, weights_info = await super().get_remote_instance_transfer_engine_info(rank)
        return session_id, {name: list(location) for name, location in weights_info.items()}


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

        query = _query(p2p_transfer_utils, engines, [(0, 0), (0, 1), (1, 0)])
        weight_infos = query.remote_weight_infos_by_session_id
        targets_to_session_id = query.targets_to_session_id
        session_id_to_server_args = query.session_id_to_server_args

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

    def test_weight_locations_are_decoded_from_the_wire_into_named_fields(self, p2p_transfer_utils):
        """The engines answer over HTTP, so JSON lists must become RemoteWeightLocation before any caller indexes them."""
        engines = [_JsonRolloutEngine(0)]

        query = _query(p2p_transfer_utils, engines, [(0, 0)])

        location = query.remote_weight_infos_by_session_id["session-0-0"][0]["weight-0"]
        assert isinstance(location, p2p_transfer_utils.RemoteWeightLocation)
        assert (location.address, location.numel, location.element_size) == (0x1000, 4, 2)


class _DeadRolloutEngine(_FakeRolloutEngine):
    async def get_remote_instance_transfer_engine_info(self, rank: int):
        self.calls.append(("get_remote_instance_transfer_engine_info", {"rank": rank}))
        raise ConnectionError(f"engine {self._engine_index} is unreachable")


class _NamelessRolloutEngine(_FakeRolloutEngine):
    async def get_remote_instance_transfer_engine_info(self, rank: int):
        self.calls.append(("get_remote_instance_transfer_engine_info", {"rank": rank}))
        return None, {}


class TestQueryFailureAttribution:
    """A target that cannot be queried belongs to one inference cell, not to the whole update."""

    def test_a_dead_engine_is_reported_instead_of_raising(self, p2p_transfer_utils):
        """Raising here would kill the trainer for a fault that only one inference cell has."""
        engines = [_DeadRolloutEngine(0), _FakeRolloutEngine(1)]

        query = _query(p2p_transfer_utils, engines, [(0, 0), (1, 0)])

        assert sorted(query.failures_by_engine_ind) == [0]
        assert isinstance(query.failures_by_engine_ind[0], ConnectionError)
        assert query.targets_to_session_id == {(1, 0): "session-1-0"}
        assert sorted(query.remote_weight_infos_by_session_id) == ["session-1-0"]

    def test_a_healthy_engine_is_queried_even_when_another_one_is_dead(self, p2p_transfer_utils):
        """The healthy targets must have their requests issued, not be skipped behind a broken one."""
        engines = [_DeadRolloutEngine(0), _FakeRolloutEngine(1)]

        _query(p2p_transfer_utils, engines, [(0, 0), (1, 0), (1, 1)])

        assert sorted(kwargs["rank"] for name, kwargs in engines[1].calls if name == "get_parallelism_info") == [0, 1]

    def test_an_engine_without_a_session_id_is_reported_as_that_engine_failing(self, p2p_transfer_utils):
        """A target that answers without a session cannot be written to, and naming it is the whole point."""
        engines = [_NamelessRolloutEngine(0)]

        query = _query(p2p_transfer_utils, engines, [(0, 0)])

        assert isinstance(query.failures_by_engine_ind[0], AssertionError)
        assert query.targets_to_session_id == {}

    def test_a_server_configuration_this_trainer_cannot_read_stays_a_hard_failure(
        self, p2p_transfer_utils, monkeypatch
    ):
        """An incompatible SGLang build is a version mismatch of the whole run, not one dead target."""
        engines = [_FakeRolloutEngine(0)]

        def rejecting_server_args(_data_dict):
            raise TypeError("unexpected keyword argument")

        monkeypatch.setattr(p2p_transfer_utils, "create_server_args_from_dict", rejecting_server_args)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _query(p2p_transfer_utils, engines, [(0, 0)])
