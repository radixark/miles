import asyncio
import dataclasses
import importlib
import sys
from argparse import Namespace
from collections import Counter
from contextlib import contextmanager
from types import ModuleType

import msgspec
import pytest

from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils"


class _FakeServerArgs(msgspec.Struct):
    """A Struct, like the real ServerArgs since sglang v0.5.20."""

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
        module.TransferTaskP2PMeta(
            rollout_engine_ind=rollout_engine_ind,
            rollout_engine_rank=rollout_engine_rank,
            source_shard=source_shard,
        )
        for source_shard, (rollout_engine_ind, rollout_engine_rank) in enumerate(pairs)
    ]


def _query(module, engines: list[_FakeRolloutEngine], pairs: list[tuple[int, int]]):
    cell_ids = [f"cell-{index}" for index in range(len(engines))]
    cell_updaters = {
        cell_id: _RolloutCellUpdater(
            args=Namespace(update_weight_engine_request_timeout=10.0), cell_id=cell_id, api_client=engine
        )
        for cell_id, engine in zip(cell_ids, engines, strict=True)
    }
    return module.query_remote_weight_infos(
        cell_updaters_of_cell_id=cell_updaters,
        engine_cell_ids=cell_ids,
        targets=_make_targets(module, pairs),
    )


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_from_remote_info_filters_unknown_fields(p2p_transfer_utils, monkeypatch, record_factory):
    server_args_type = record_factory("ServerArgs", [("model_path", str)])
    monkeypatch.setattr(p2p_transfer_utils, "ServerArgs", server_args_type)

    result = p2p_transfer_utils.create_server_args_from_dict({"model_path": "/model", "unknown_field": True})

    assert isinstance(result, server_args_type)
    assert result.model_path == "/model"


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

    def test_weight_locations_are_decoded_from_the_wire_into_named_fields(self, p2p_transfer_utils):
        """The engines answer over HTTP, so JSON lists must become RemoteWeightLocation before any caller indexes them."""
        engines = [_JsonRolloutEngine(0)]

        weight_infos, _targets_to_session_id, _session_id_to_server_args = _query(
            p2p_transfer_utils, engines, [(0, 0)]
        )

        location = weight_infos["session-0-0"][0]["weight-0"]
        assert isinstance(location, p2p_transfer_utils.RemoteWeightLocation)
        assert (location.address, location.numel, location.element_size) == (0x1000, 4, 2)


def _make_cell_updaters(module: ModuleType, engines: list[_FakeRolloutEngine]) -> tuple[dict, list[str]]:
    engine_cell_ids = [f"cell-{index}" for index in range(len(engines))]
    args = Namespace(update_weight_engine_request_timeout=10.0)
    cell_updaters = {
        cell_id: _RolloutCellUpdater(args=args, cell_id=cell_id, api_client=engine)
        for cell_id, engine in zip(engine_cell_ids, engines, strict=True)
    }
    return cell_updaters, engine_cell_ids


class _UnreachableRolloutEngine(_FakeRolloutEngine):
    async def get_remote_instance_transfer_engine_info(self, rank: int) -> tuple[str, dict]:
        self.calls.append(("get_remote_instance_transfer_engine_info", {"rank": rank}))
        raise RuntimeError("the engine is gone")


class _HangingRolloutEngine(_FakeRolloutEngine):
    async def get_parallelism_info(self, rank: int) -> dict:
        self.calls.append(("get_parallelism_info", {"rank": rank}))
        await asyncio.sleep(3600)


class TestQueryRemoteWeightInfosGivesUpOnABrokenCell:
    def test_an_unreachable_engine_is_left_out_of_the_transfer_plan(self, p2p_transfer_utils: ModuleType) -> None:
        """Planning a transfer to an engine that never answered would write into an address nobody confirmed."""
        engines = [_UnreachableRolloutEngine(0), _FakeRolloutEngine(1)]
        cell_updaters, engine_cell_ids = _make_cell_updaters(p2p_transfer_utils, engines)

        weight_infos, targets_to_session_id, _server_args = p2p_transfer_utils.query_remote_weight_infos(
            cell_updaters, engine_cell_ids, _make_targets(p2p_transfer_utils, [(0, 0), (1, 0)])
        )

        assert cell_updaters["cell-0"].is_errored
        assert not cell_updaters["cell-1"].is_errored
        assert targets_to_session_id == {(1, 0): "session-1-0"}
        assert list(weight_infos) == ["session-1-0"]

    def test_a_metadata_query_that_never_returns_gives_the_cell_up(self, p2p_transfer_utils: ModuleType) -> None:
        """Connecting must not hang forever on one engine that stopped answering."""
        engines = [_HangingRolloutEngine(0)]
        cell_updaters, engine_cell_ids = _make_cell_updaters(p2p_transfer_utils, engines)
        cell_updaters["cell-0"]._args.update_weight_engine_request_timeout = 0.05

        _weight_infos, targets_to_session_id, _server_args = p2p_transfer_utils.query_remote_weight_infos(
            cell_updaters, engine_cell_ids, _make_targets(p2p_transfer_utils, [(0, 0)])
        )

        assert cell_updaters["cell-0"].is_errored
        assert targets_to_session_id == {}

    def test_a_cell_given_up_earlier_contributes_no_metadata(self, p2p_transfer_utils: ModuleType) -> None:
        """Re-querying a cell that already lost the update would only waste the trainer's deadline."""
        engines = [_FakeRolloutEngine(0)]
        cell_updaters, engine_cell_ids = _make_cell_updaters(p2p_transfer_utils, engines)
        cell_updaters["cell-0"].mark_errored(RuntimeError("lost"))

        _weight_infos, targets_to_session_id, _server_args = p2p_transfer_utils.query_remote_weight_infos(
            cell_updaters, engine_cell_ids, _make_targets(p2p_transfer_utils, [(0, 0)])
        )

        assert engines[0].calls == []
        assert targets_to_session_id == {}
