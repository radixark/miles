"""The engines that are up reach the run's observers once per change, GPUs and UUIDs aligned worker by worker."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from miles.ray.rollout.engine_topology import EngineTopologyReporter, engine_infos
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.utils import observers as observers_module
from miles.utils.misc import SingletonMeta
from miles.utils.observers import Observer, Observers
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class _TopologyRecorder(Observer):
    def __init__(self):
        self.topologies: list[list[dict]] = []

    def engine_topology(self, engines):
        self.topologies.append(engines)


@pytest.fixture
def recorder(monkeypatch) -> _TopologyRecorder:
    observer = _TopologyRecorder()
    monkeypatch.setattr(
        observers_module,
        "entry_points",
        lambda group: [SimpleNamespace(name="recorder", load=lambda: lambda: observer)],
    )
    SingletonMeta._instances.pop(Observers, None)
    return observer


class FakeWorkerHandle(BaseWorkerHandle):
    async def _get_gpu_uuids(self, *, gpu_ids):
        return [None] * len(gpu_ids)

    async def wait_ready(self, *, timeout):
        return None

    async def probe_is_dead(self):
        return True


class UuidWorkerHandle(BaseWorkerHandle):
    def __init__(self, uuid_by_gpu_id: dict[int, str]):
        self._uuid_by_gpu_id = uuid_by_gpu_id

    async def probe_is_dead(self) -> bool:
        raise AssertionError("reporting engines must not probe a worker for death")

    async def _get_gpu_uuids(self, *, gpu_ids):
        return [self._uuid_by_gpu_id[gpu_id] for gpu_id in gpu_ids]

    async def wait_ready(self, *, timeout):
        return None

    async def wait_dead(self, *, timeout):
        return None


class GatedWorkerHandle(BaseWorkerHandle):
    def __init__(self, uuid, *, signal=None, wait_for=None):
        self._uuid = uuid
        self._signal = signal
        self._wait_for = wait_for

    async def probe_is_dead(self) -> bool:
        raise AssertionError("reporting engines must not probe a worker for death")

    async def _get_gpu_uuids(self, *, gpu_ids):
        if self._signal is not None:
            self._signal.set()
        if self._wait_for is not None:
            await self._wait_for.wait()
        return [self._uuid for _ in gpu_ids]

    async def wait_ready(self, *, timeout):
        return None

    async def wait_dead(self, *, timeout):
        return None


class FlakyWorkerHandle(BaseWorkerHandle):
    def __init__(self):
        self.fail = True

    async def probe_is_dead(self) -> bool:
        raise AssertionError("reporting engines must not probe a worker for death")

    async def _get_gpu_uuids(self, *, gpu_ids):
        if self.fail:
            raise RuntimeError("worker unreachable")
        return [None] * len(gpu_ids)

    async def wait_ready(self, *, timeout):
        return None

    async def wait_dead(self, *, timeout):
        return None


class FakeWorkerProvider:
    def __init__(self, infos_by_cell, handles_by_name=None):
        self._infos_by_cell = infos_by_cell
        self._handles_by_name = handles_by_name or {}
        self.queried_cell_ids: list[list[str]] = []

    def get_worker_infos(self, *, cell_ids):
        self.queried_cell_ids.append(cell_ids)
        return [self._infos_by_cell[cell_id] for cell_id in cell_ids]

    def get_handles_of_worker_infos(self, infos):
        return {info.name: self._handles_by_name.get(info.name, FakeWorkerHandle()) for info in infos}


class FakeCell:
    """Duck-typed ServerCell: the topology reads only the driver-side routing facts."""

    def __init__(self, url, cell_index=0, alive=True, worker_type="regular"):
        self.meta = ServerCellMetadata(
            model_id="default",
            worker_type=worker_type,
            cell_id=f"inference-engine-0-0-{cell_index}",
            num_gpus_per_engine=1,
            gpu_offset=cell_index,
            sglang_api_key=None,
            worker_name=f"inference-engine-0-0-{cell_index}-0",
            needs_offload=False,
            update_weights=False,
            workers_hash=f"hash-{cell_index}",
        )
        self.server_url = url
        self.is_pending_weights_or_serving = alive


def _worker_info(name, node, gpus, generation=1):
    return WorkerInfo(
        name=name,
        generation=generation,
        self_addrs={"primary": HostAndPort(host=node, port=30001)},
        gpu_ids=gpus,
    )


def _servers(cells):
    server = type("FakeServer", (), {"all_server_cells": {f"cell-{i}": cell for i, cell in enumerate(cells)}})()
    return {"default": server}


async def test_report_groups_multinode_engines_and_dedups(recorder):
    """Worker-manager infos become one engine per cell; it is reported again only when a worker changed."""
    infos_by_cell = {
        "inference-engine-0-0-0": [
            _worker_info("inference-engine-0-0-0", "node-a", [0, 1]),
            _worker_info("inference-engine-0-0-1", "node-b", [0, 1]),
        ],
        "inference-engine-0-0-1": [_worker_info("inference-engine-0-1-0", "node-a", [2, 3])],
    }
    provider = FakeWorkerProvider(infos_by_cell)
    servers = _servers([FakeCell("http://a:1", cell_index=0), FakeCell("http://b:1", cell_index=1)])
    reporter = EngineTopologyReporter()

    await reporter.report_if_changed(servers, provider=provider)
    [engines] = recorder.topologies
    assert [e["addr"] for e in engines] == ["http://a:1", "http://b:1"]
    multinode = engines[0]
    assert multinode["gpus"] == [["node-a", 0], ["node-a", 1], ["node-b", 0], ["node-b", 1]]
    assert len(multinode["gpu_uuids"]) == 4

    await reporter.report_if_changed(servers, provider=provider)  # steady state: fingerprint unchanged
    assert len(recorder.topologies) == 1

    infos_by_cell["inference-engine-0-0-1"] = [_worker_info("inference-engine-0-1-0", "node-a", [2, 3], generation=2)]
    await reporter.report_if_changed(servers, provider=provider)  # recovery: same worker, new generation
    assert len(recorder.topologies) == 2

    # Counting the reports says one fired, not what it carried: the fingerprint watches the
    # worker while the addr comes from the cell, so a report can still carry stale engines.
    reported = recorder.topologies[1]
    assert [e["addr"] for e in reported] == ["http://a:1", "http://b:1"]
    assert reported[1]["gpus"] == [["node-a", 2], ["node-a", 3]]


async def test_report_skips_dead_cells(recorder):
    """Cells that are not alive are left out of the topology and never queried."""
    infos_by_cell = {"inference-engine-0-0-0": [_worker_info("inference-engine-0-0-0", "n", [0])]}
    provider = FakeWorkerProvider(infos_by_cell)

    await EngineTopologyReporter().report_if_changed(
        _servers([FakeCell("http://a:1", cell_index=0), FakeCell("http://b:1", cell_index=1, alive=False)]),
        provider=provider,
    )

    [engines] = recorder.topologies
    assert [e["addr"] for e in engines] == ["http://a:1"]
    assert provider.queried_cell_ids == [["inference-engine-0-0-0"]]


async def test_report_survives_a_missing_worker_manager(recorder, caplog):
    """Engines not yet owned by the worker manager degrade to a warning, not a crash."""

    class _UnreachableProvider:
        def get_worker_infos(self, *, cell_ids):
            raise ValueError("worker manager actor not found")

        def get_handles_of_worker_infos(self, infos):
            raise AssertionError("nothing to build a handle for")

    with caplog.at_level(logging.WARNING):
        await EngineTopologyReporter().report_if_changed(
            _servers([FakeCell("http://a:1")]), provider=_UnreachableProvider()
        )

    assert recorder.topologies == []
    assert any("failed in engine_topology" in r.message for r in caplog.records)


async def test_report_without_observers_does_not_ask_the_worker_manager():
    """With nothing listening the reporter must cost nothing, not even a worker-manager round trip."""
    provider = FakeWorkerProvider({})
    reporter = EngineTopologyReporter()

    await reporter.report_if_changed(_servers([FakeCell("http://a:1")]), provider=provider)

    assert provider.queried_cell_ids == []
    assert reporter._reported_fingerprint is None


async def test_report_publishes_the_topology_from_a_running_event_loop(recorder, caplog):
    """Driven from inside a running asyncio loop (as prepare_rollout does), the topology is reported without warning."""
    infos_by_cell = {"inference-engine-0-0-0": [_worker_info("inference-engine-0-0-0", "node-a", [0, 1])]}
    reporter = EngineTopologyReporter()

    assert asyncio.get_running_loop().is_running()
    with caplog.at_level(logging.WARNING):
        await reporter.report_if_changed(
            _servers([FakeCell("http://a:1")]), provider=FakeWorkerProvider(infos_by_cell)
        )

    assert [e["addr"] for e in recorder.topologies[0]] == ["http://a:1"]
    assert reporter._reported_fingerprint is not None
    assert not [r for r in caplog.records if "failed in engine_topology" in r.message]


async def test_engine_infos_projects_cells_and_workers_exactly():
    """Every cell becomes one engine whose gpu pairs and probed uuids stay aligned worker by worker."""
    cells = [
        FakeCell("http://a:1", cell_index=0, worker_type="decode"),
        FakeCell("http://b:1", cell_index=1, worker_type="prefill"),
    ]
    worker_infos_per_cell = [
        [
            _worker_info("engine-0-0", "[2001:db8::7]", [4, 5]),
            _worker_info("engine-0-1", "node-b", [0]),
        ],
        [_worker_info("engine-1-0", "node-c", [3])],
    ]
    provider = FakeWorkerProvider(
        {},
        {
            "engine-0-0": UuidWorkerHandle({4: "GPU-a", 5: "GPU-b"}),
            "engine-0-1": UuidWorkerHandle({0: "GPU-c"}),
            "engine-1-0": UuidWorkerHandle({3: "GPU-d"}),
        },
    )

    engines = await engine_infos(cells, worker_infos_per_cell, provider=provider)

    assert engines == [
        dict(
            addr="http://a:1",
            worker_type="decode",
            engine_rank=0,
            gpus=[["2001:db8::7", 4], ["2001:db8::7", 5], ["node-b", 0]],
            gpu_uuids=["GPU-a", "GPU-b", "GPU-c"],
        ),
        dict(
            addr="http://b:1",
            worker_type="prefill",
            engine_rank=1,
            gpus=[["node-c", 3]],
            gpu_uuids=["GPU-d"],
        ),
    ]


async def test_engine_infos_uses_null_gpu_uuids_when_provider_has_no_handle():
    """A worker without a callable handle retains one null UUID slot for each GPU."""

    class _MissingHandleWorkerProvider(BaseWorkerProvider):
        async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
            raise AssertionError(f"the topology must not request addresses for {worker_name}")

        def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
            raise AssertionError(f"the topology already received worker infos for {cell_ids}")

        def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
            return {infos[0].name: UuidWorkerHandle({0: "GPU-a"})}

    worker_infos = [
        _worker_info("engine-0-0", "node-a", [0]),
        _worker_info("engine-0-1", "node-b", [2, 3]),
    ]

    engines = await engine_infos([FakeCell("http://a:1")], [worker_infos], provider=_MissingHandleWorkerProvider())

    assert engines[0]["gpus"] == [["node-a", 0], ["node-b", 2], ["node-b", 3]]
    assert engines[0]["gpu_uuids"] == ["GPU-a", None, None]


async def test_engine_infos_probes_workers_concurrently():
    """The first worker's probe only finishes once the second one started, so serial probing would hang."""
    second_started = asyncio.Event()
    worker_infos = [_worker_info("engine-0-0", "node-a", [0]), _worker_info("engine-0-1", "node-b", [1])]
    provider = FakeWorkerProvider(
        {},
        {
            "engine-0-0": GatedWorkerHandle("GPU-a", wait_for=second_started),
            "engine-0-1": GatedWorkerHandle("GPU-b", signal=second_started),
        },
    )

    engines = await asyncio.wait_for(
        engine_infos([FakeCell("http://a:1")], [worker_infos], provider=provider), timeout=5
    )

    assert engines[0]["gpu_uuids"] == ["GPU-a", "GPU-b"]


async def test_report_retries_after_a_gpu_uuid_probe_failure(recorder, caplog):
    """A failed uuid probe reports nothing and leaves the fingerprint unset, so the next call reports again."""
    probe = FlakyWorkerHandle()
    infos_by_cell = {"inference-engine-0-0-0": [_worker_info("inference-engine-0-0-0", "node-a", [0])]}
    provider = FakeWorkerProvider(infos_by_cell, {"inference-engine-0-0-0": probe})
    servers = _servers([FakeCell("http://a:1")])
    reporter = EngineTopologyReporter()

    with caplog.at_level(logging.WARNING):
        await reporter.report_if_changed(servers, provider=provider)

    assert recorder.topologies == []
    assert reporter._reported_fingerprint is None
    assert any("failed in engine_topology" in r.message for r in caplog.records)

    probe.fail = False
    await reporter.report_if_changed(servers, provider=provider)

    [engines] = recorder.topologies
    assert [e["addr"] for e in engines] == ["http://a:1"]


async def test_report_follows_cell_liveness_transitions(recorder):
    """A cell that drops out and comes back must be removed from and then restored to the reported topology."""
    infos_by_cell = {
        "inference-engine-0-0-0": [_worker_info("inference-engine-0-0-0", "node-a", [0])],
        "inference-engine-0-0-1": [_worker_info("inference-engine-0-1-0", "node-b", [1])],
    }
    provider = FakeWorkerProvider(infos_by_cell)
    first, second = FakeCell("http://a:1", cell_index=0), FakeCell("http://b:1", cell_index=1)
    servers = _servers([first, second])
    reporter = EngineTopologyReporter()

    await reporter.report_if_changed(servers, provider=provider)
    second.is_pending_weights_or_serving = False
    await reporter.report_if_changed(servers, provider=provider)
    second.is_pending_weights_or_serving = True
    await reporter.report_if_changed(servers, provider=provider)

    assert [[e["addr"] for e in engines] for engines in recorder.topologies] == [
        ["http://a:1", "http://b:1"],
        ["http://a:1"],
        ["http://a:1", "http://b:1"],
    ]
