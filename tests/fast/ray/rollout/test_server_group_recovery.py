from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.router_worker_client import (
    RouterWorkerRegistrationFailed,
    RouterWorkerSubmissionUnknown,
)
from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.server_group import ServerGroup

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


class _RemoteRegistration:
    def __init__(
        self,
        failures: list[BaseException] | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.calls = 0
        self._failures = iter(failures or [])
        self._on_call = on_call

    async def remote(self) -> None:
        self.calls += 1
        if self._on_call is not None:
            self._on_call()
        failure = next(self._failures, None)
        if failure is not None:
            raise failure


def _actor(registration: _RemoteRegistration) -> SimpleNamespace:
    return SimpleNamespace(
        register_with_router=registration,
    )


def _group(*, all_engines: list[object], num_gpus_per_engine: int = 1) -> ServerGroup:
    return ServerGroup(
        args=make_args(num_gpus_per_node=8),
        pg=None,
        all_engines=all_engines,
        num_gpus_per_engine=num_gpus_per_engine,
        has_new_engines=False,
    )


@pytest.mark.asyncio
async def test_recover_replaces_complete_multinode_engine_when_one_rank_stops() -> None:
    engines = [
        SimpleNamespace(is_allocated=True, mark_alive=MagicMock()),
        SimpleNamespace(is_allocated=False, mark_alive=MagicMock()),
        SimpleNamespace(is_allocated=True, mark_alive=MagicMock()),
        SimpleNamespace(is_allocated=True, mark_alive=MagicMock()),
    ]
    group = _group(all_engines=engines, num_gpus_per_engine=16)

    def stop_engines(*, engine_indices: list[int]) -> None:
        assert engine_indices == [0]
        engines[0].is_allocated = False

    group.stop_engines = MagicMock(side_effect=stop_engines)

    def start_engines(
        port_cursors: PortCursors,
        *,
        register_with_router: bool,
        start_indices: list[int] | None,
    ) -> tuple[list[object], list[int]]:
        assert port_cursors == PortCursors.empty()
        assert register_with_router is False
        assert start_indices == [0, 1]
        return [], [0, 1]

    group.start_engines = MagicMock(side_effect=start_engines)

    await group.recover(
        port_cursors=PortCursors.empty(),
        register_with_router=False,
        filter_indices=[1],
    )

    assert group.stop_engines.call_count == 1
    assert group.start_engines.call_count == 1
    assert engines[0].mark_alive.call_count == 1
    assert engines[1].mark_alive.call_count == 1
    assert engines[2].mark_alive.call_count == 0
    assert engines[3].mark_alive.call_count == 0


@pytest.mark.asyncio
async def test_recover_stops_new_engines_when_initialization_fails() -> None:
    engines = [
        SimpleNamespace(is_allocated=False, mark_alive=MagicMock()),
        SimpleNamespace(is_allocated=False, mark_alive=MagicMock()),
    ]
    group = _group(all_engines=engines, num_gpus_per_engine=16)

    async def failed_initialization() -> None:
        raise RouterWorkerRegistrationFailed("registration failed")

    group.start_engines = MagicMock(return_value=([failed_initialization()], [0, 1]))
    group.stop_engines = MagicMock()

    with pytest.raises(RouterWorkerRegistrationFailed, match="registration failed"):
        await group.recover(
            port_cursors=PortCursors.empty(),
            register_with_router=True,
            filter_indices=[0],
        )

    group.stop_engines.assert_called_once_with(engine_indices=[0, 1])
    assert engines[0].mark_alive.call_count == 0
    assert engines[1].mark_alive.call_count == 0


def test_pending_registration_tracks_only_multinode_engine_roots() -> None:
    engines = [SimpleNamespace() for _ in range(4)]
    group = _group(all_engines=engines, num_gpus_per_engine=16)

    assert group._logical_engine_root_indices([0, 1, 2, 3]) == [0, 2]


@pytest.mark.asyncio
async def test_pending_registration_retries_only_failed_engines() -> None:
    first_registration = _RemoteRegistration()
    second_registration = _RemoteRegistration([RuntimeError("registration failed")])
    engines = [
        SimpleNamespace(
            is_alive=True,
            generation_id="first",
            actor_handle=_actor(first_registration),
        ),
        SimpleNamespace(
            is_alive=True,
            generation_id="second",
            actor_handle=_actor(second_registration),
        ),
    ]
    group = _group(all_engines=engines)
    group.pending_router_registration = {0, 1}

    with pytest.raises(RuntimeError, match="registration failed"):
        await group.register_pending_engines_with_router(
            rollout_engine_generation_ids=["first", "second"],
        )

    assert group.pending_router_registration == {1}

    await group.register_pending_engines_with_router(
        rollout_engine_generation_ids=["second"],
    )

    assert group.pending_router_registration == set()
    assert first_registration.calls == 1
    assert second_registration.calls == 2


@pytest.mark.asyncio
async def test_pending_registration_ignores_replacement_after_weight_snapshot() -> None:
    updated_registration = _RemoteRegistration()
    later_registration = _RemoteRegistration()
    engines = [
        SimpleNamespace(is_alive=True, generation_id="updated", actor_handle=_actor(updated_registration)),
        SimpleNamespace(is_alive=True, generation_id="later", actor_handle=_actor(later_registration)),
    ]
    group = _group(all_engines=engines)
    group.pending_router_registration = {0, 1}

    await group.register_pending_engines_with_router(
        rollout_engine_generation_ids=["updated"],
    )

    assert group.pending_router_registration == {1}
    assert updated_registration.calls == 1
    assert later_registration.calls == 0


@pytest.mark.asyncio
async def test_completed_registration_does_not_clear_newer_replacement() -> None:
    engines: list[SimpleNamespace] = []
    replacement_registration = _RemoteRegistration()
    replacement_actor = _actor(replacement_registration)

    def replace_actor() -> None:
        engines[0].actor_handle = replacement_actor
        engines[0].generation_id = "replacement"

    original_registration = _RemoteRegistration(on_call=replace_actor)
    engines.append(
        SimpleNamespace(
            is_alive=True,
            generation_id="original",
            actor_handle=_actor(original_registration),
        )
    )
    group = _group(all_engines=engines)
    group.pending_router_registration = {0}

    await group.register_pending_engines_with_router(
        rollout_engine_generation_ids=["original"],
    )

    assert group.pending_router_registration == {0}
    assert original_registration.calls == 1
    assert replacement_registration.calls == 0


@pytest.mark.asyncio
async def test_unknown_submission_quarantines_engine_generation() -> None:
    registration = _RemoteRegistration([RouterWorkerSubmissionUnknown("submission outcome unknown")])
    engines = [
        SimpleNamespace(
            is_alive=True,
            generation_id="unknown",
            actor_handle=_actor(registration),
        )
    ]
    group = _group(all_engines=engines)
    group.pending_router_registration = {0}
    group.stop_engines = MagicMock()

    with pytest.raises(RouterWorkerSubmissionUnknown, match="outcome unknown"):
        await group.register_pending_engines_with_router(
            rollout_engine_generation_ids=["unknown"],
        )

    group.stop_engines.assert_called_once_with(engine_indices=[0])


@pytest.mark.asyncio
async def test_failed_registration_quarantines_engine_generation() -> None:
    registration = _RemoteRegistration([RouterWorkerRegistrationFailed("registration failed")])
    engines = [
        SimpleNamespace(
            is_alive=True,
            generation_id="failed",
            actor_handle=_actor(registration),
        ),
        SimpleNamespace(
            is_alive=True,
            generation_id="failed-peer",
            actor_handle=_actor(_RemoteRegistration()),
        ),
    ]
    group = _group(all_engines=engines, num_gpus_per_engine=16)
    group.pending_router_registration = {0}
    group.stop_engines = MagicMock()

    with pytest.raises(RouterWorkerRegistrationFailed, match="registration failed"):
        await group.register_pending_engines_with_router(
            rollout_engine_generation_ids=["failed"],
        )

    group.stop_engines.assert_called_once_with(engine_indices=[0, 1])
