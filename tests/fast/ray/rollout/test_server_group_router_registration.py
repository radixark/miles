from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import chunk_engines_into_cells, fake_actor_handle, make_args

from miles.ray.rollout.server_engine import AddrInfo, ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from miles.utils import async_utils

_MODULE = "miles.ray.rollout.server_group"


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


def _build_group(
    *,
    events: list[tuple[str, dict]],
    num_engines: int = 1,
    num_gpus_per_engine: int = 1,
    worker_type: str = "regular",
    router_ip: str | None = "10.0.0.9",
    router_port: int | None = 9000,
    bootstrap_port: int | None = None,
    use_miles_router: bool = False,
    rollout_external: bool = False,
    remove_worker_effect=None,
) -> ServerGroup:
    args = make_args(num_gpus_per_node=8, use_miles_router=use_miles_router, rollout_external=rollout_external)
    engines = []
    for index in range(num_engines):
        engine = ServerEngine()
        engine.mark_allocated_uninitialized(fake_actor_handle())
        engine.set_addressing(
            AddrInfo(server_url=f"http://10.0.0.{index + 1}:3000{index}", bootstrap_port=bootstrap_port)
        )
        engine.mark_alive()
        engines.append(engine)

    group = ServerGroup(
        args=args,
        pg=(None, [], []),
        cells=chunk_engines_into_cells(engines, num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8),
        num_gpus_per_engine=num_gpus_per_engine,
        has_new_engines=False,
        worker_type=worker_type,
        router_ip=router_ip,
        router_port=router_port,
    )
    group._recording_router_client = _RecordingRouterApiClient(events, remove_worker_effect=remove_worker_effect)
    return group


def _with_recording_client(group: ServerGroup):
    return patch.object(ServerGroup, "_router_api_client", property(lambda self: self._recording_router_client))


async def test_registration_publishes_the_url_the_engine_actually_serves():
    """The router must be told the url the rollout process derived from the allocator."""
    events: list[tuple[str, dict]] = []
    group = _build_group(events=events)

    with _with_recording_client(group):
        await group.register_workers([0])

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
    group = _build_group(events=events, worker_type="prefill", bootstrap_port=8998)

    with _with_recording_client(group):
        await group.register_workers([0])

    assert events[0][1]["worker_type"] == "prefill"
    assert events[0][1]["bootstrap_port"] == 8998


async def test_registration_addresses_only_node0_of_a_multi_node_engine():
    """Only node 0 serves the router-visible endpoint."""
    events: list[tuple[str, dict]] = []
    group = _build_group(events=events, num_engines=2, num_gpus_per_engine=16)

    with _with_recording_client(group):
        await group.register_workers([0, 1])

    assert [kwargs["worker_url"] for _name, kwargs in events] == ["http://10.0.0.1:30000"]


@pytest.mark.parametrize("missing", [dict(router_ip=None), dict(router_port=None)])
async def test_registration_is_skipped_without_a_router(missing):
    events: list[tuple[str, dict]] = []
    group = _build_group(events=events, **missing)

    with _with_recording_client(group):
        await group.register_workers([0])
        await group.unregister_workers([0])

    assert events == []


async def test_an_external_engine_is_never_registered_or_unregistered():
    """External engines are published by whoever runs them."""
    events: list[tuple[str, dict]] = []
    group = _build_group(events=events, rollout_external=True)

    with _with_recording_client(group):
        await group.register_workers([0])
        await group.unregister_workers([0])

    assert events == []


def test_stop_engines_unregisters_before_killing_the_actor():
    """Killing first would leave the router routing to a dead worker."""
    events: list[tuple[str, dict]] = []
    group = _build_group(events=events)

    with (
        _with_recording_client(group),
        patch(f"{_MODULE}.ray") as ray_mock,
    ):
        ray_mock.get.side_effect = lambda *args, **kwargs: events.append(("shutdown", {}))
        ray_mock.kill.side_effect = lambda handle: events.append(("kill", {}))
        group.stop_engines(engine_indices=[0])

    assert [name for name, _kwargs in events] == ["remove_worker", "shutdown", "kill"]
    assert events[0][1] == {"worker_url": "http://10.0.0.1:30000", "use_legacy_api": False}


def test_a_router_that_rejects_the_unregister_still_kills_the_actor():
    """Teardown is how a wedged engine is reclaimed, so a router error must not abort it."""

    async def _reject():
        raise RuntimeError("router rejected the removal")

    events: list[tuple[str, dict]] = []
    group = _build_group(events=events, remove_worker_effect=_reject)

    with (
        _with_recording_client(group),
        patch(f"{_MODULE}.ray") as ray_mock,
    ):
        ray_mock.get.side_effect = lambda *args, **kwargs: events.append(("shutdown", {}))
        ray_mock.kill.side_effect = lambda handle: events.append(("kill", {}))
        group.stop_engines(engine_indices=[0])

    assert [name for name, _kwargs in events] == ["remove_worker", "shutdown", "kill"]
    assert not group.all_engines[0].is_allocated


def test_a_router_that_never_answers_the_unregister_does_not_block_teardown():
    """The shared http client has no read timeout, so an unanswered removal would wedge teardown forever."""

    async def _hang():
        await asyncio.sleep(3600)

    events: list[tuple[str, dict]] = []
    group = _build_group(events=events, remove_worker_effect=_hang)

    with (
        _with_recording_client(group),
        patch(f"{_MODULE}._SHUTDOWN_TIMEOUT", 0.1),
        patch(f"{_MODULE}.ray") as ray_mock,
    ):
        ray_mock.kill.side_effect = lambda handle: events.append(("kill", {}))
        group.stop_engines(engine_indices=[0])

    assert [name for name, _kwargs in events] == ["remove_worker", "kill"]
    assert not group.all_engines[0].is_allocated


def test_use_miles_router_reaches_both_router_calls():
    """--use-miles-router pins the legacy query-string API on register and unregister alike."""
    events: list[tuple[str, dict]] = []
    group = _build_group(events=events, use_miles_router=True)

    with (
        _with_recording_client(group),
        patch(f"{_MODULE}.ray"),
    ):
        async_utils.run(group.register_workers([0]))
        group.stop_engines(engine_indices=[0])

    assert [kwargs["use_legacy_api"] for _name, kwargs in events] == [True, True]
