from __future__ import annotations

import asyncio
from unittest.mock import patch

from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, compute_nodes_per_engine
from miles.ray.rollout.server_engine import AddrInfo, ServerEngine

_CELL_MODULE = "miles.ray.rollout.server_cell"


class _RecordingRouterApiClient:
    def __init__(self, events: list[tuple[str, dict]], remove_worker_effect=None):
        self._events = events
        self._remove_worker_effect = remove_worker_effect

    async def add_worker(self, **kwargs):
        self._events.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs):
        self._events.append(("remove_worker", kwargs))
        if self._remove_worker_effect is not None:
            await self._remove_worker_effect()


def _build_server(
    *,
    events: list[tuple[str, dict]],
    num_engines: int = 1,
    num_gpus_per_engine: int = 1,
    worker_type: str = "regular",
    router_ip: str = "10.0.0.9",
    router_port: int = 9000,
    bootstrap_port: int | None = None,
    use_miles_router: bool = False,
    remove_worker_effect=None,
) -> RolloutServer:
    args = make_args(num_gpus_per_node=8, use_miles_router=use_miles_router)
    engines = []
    for index in range(num_engines):
        engine = ServerEngine()
        engine.mark_allocated_uninitialized(fake_actor_handle())
        engine.set_addressing(
            AddrInfo(server_url=f"http://10.0.0.{index + 1}:3000{index}", bootstrap_port=bootstrap_port)
        )
        engine.mark_alive()
        engines.append(engine)

    nodes_per_engine = compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    cells = [
        ServerCell(
            engines=engines[i : i + nodes_per_engine],
            args=args,
            num_gpus_per_engine=num_gpus_per_engine,
            worker_type=worker_type,
            rank_offset=i,
        )
        for i in range(0, len(engines), nodes_per_engine)
    ]

    srv = RolloutServer(
        server_cells=cells,
        args=args,
        router_ip=router_ip,
        router_port=router_port,
    )
    srv._recording_router_client = _RecordingRouterApiClient(events, remove_worker_effect=remove_worker_effect)
    return srv


def _with_recording_client(srv: RolloutServer):
    return patch.object(RolloutServer, "_router_api_client", property(lambda self: self._recording_router_client))


async def test_registration_publishes_the_url_the_engine_actually_serves():
    """The router must be told the url the rollout process derived from the allocator."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events)

    with _with_recording_client(srv):
        await srv.server_cells[0].register(srv._router_api_client)

    assert events == [
        (
            "add_worker",
            {
                "worker_url": "http://10.0.0.1:30000",
                "worker_type": "regular",
                "use_legacy_api": False,
                "bootstrap_port": None,
            },
        )
    ]


async def test_registration_passes_the_bootstrap_port_of_a_prefill_worker():
    """PD disaggregation needs the decode side to dial this port."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, worker_type="prefill", bootstrap_port=8998)

    with _with_recording_client(srv):
        await srv.server_cells[0].register(srv._router_api_client)

    assert events[0][1]["worker_type"] == "prefill"
    assert events[0][1]["bootstrap_port"] == 8998


async def test_stopping_a_cell_that_never_started_publishes_nothing():
    """An unallocated cell has no url, so teardown must not try to unregister it."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, num_engines=2)
    for engine in srv.server_cells[0].engines:
        engine.mark_stopped()

    with (
        _with_recording_client(srv),
        patch(f"{_CELL_MODULE}.ray"),
    ):
        await srv.stop_cells([0])

    assert events == []


async def test_registration_addresses_only_the_primary_engine_of_a_multi_node_cell():
    """Only the primary (node-0) engine serves the router-visible endpoint."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, num_engines=2, num_gpus_per_engine=16)

    with _with_recording_client(srv):
        await srv.server_cells[0].register(srv._router_api_client)

    assert [kwargs["worker_url"] for _name, kwargs in events] == ["http://10.0.0.1:30000"]


async def test_stop_cells_unregisters_before_killing_the_actor():
    """Killing first would leave the router routing to a dead worker."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events)

    with (
        _with_recording_client(srv),
        patch(f"{_CELL_MODULE}.ray") as ray_mock,
    ):
        ray_mock.get.side_effect = lambda *args, **kwargs: events.append(("shutdown", {}))
        ray_mock.kill.side_effect = lambda handle: events.append(("kill", {}))
        await srv.stop_cells([0])

    assert [name for name, _kwargs in events] == ["remove_worker", "shutdown", "kill"]
    assert events[0][1] == {"worker_url": "http://10.0.0.1:30000", "use_legacy_api": False}


async def test_a_router_that_rejects_the_unregister_still_kills_the_actor():
    """Teardown is how a wedged engine is reclaimed, so a router error must not abort it."""

    async def _reject():
        raise RuntimeError("router rejected the removal")

    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, remove_worker_effect=_reject)

    with (
        _with_recording_client(srv),
        patch(f"{_CELL_MODULE}.ray") as ray_mock,
    ):
        ray_mock.get.side_effect = lambda *args, **kwargs: events.append(("shutdown", {}))
        ray_mock.kill.side_effect = lambda handle: events.append(("kill", {}))
        await srv.stop_cells([0])

    assert [name for name, _kwargs in events] == ["remove_worker", "shutdown", "kill"]
    assert not srv.server_cells[0].primary_engine.is_allocated


async def test_a_router_that_never_answers_the_unregister_does_not_block_teardown():
    """The shared http client has no read timeout, so an unanswered removal would wedge teardown forever."""

    async def _hang():
        await asyncio.sleep(3600)

    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, remove_worker_effect=_hang)

    with (
        _with_recording_client(srv),
        patch(f"{_CELL_MODULE}.SHUTDOWN_TIMEOUT", 0.1),
        patch(f"{_CELL_MODULE}.ray") as ray_mock,
    ):
        ray_mock.kill.side_effect = lambda handle: events.append(("kill", {}))
        await srv.stop_cells([0])

    assert [name for name, _kwargs in events] == ["remove_worker", "kill"]
    assert not srv.server_cells[0].primary_engine.is_allocated


async def test_use_miles_router_reaches_both_router_calls():
    """--use-miles-router pins the legacy query-string API on register and unregister alike."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, use_miles_router=True)

    with (
        _with_recording_client(srv),
        patch(f"{_CELL_MODULE}.ray"),
    ):
        await srv.server_cells[0].register(srv._router_api_client)
        await srv.stop_cells([0])

    assert [kwargs["use_legacy_api"] for _name, kwargs in events] == [True, True]
