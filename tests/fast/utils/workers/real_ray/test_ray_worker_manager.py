from __future__ import annotations

import asyncio
import os
import time

import pytest
import ray
from tests.fast.utils.workers.real_ray.conftest import (
    is_process_running,
    kill_quietly,
    make_command_spec,
    wait_until_named_manager_is_gone,
)

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.http_utils import wait_tcp_ready_async
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, PortInfo


class TestLaunchOnRealRay:
    def test_every_worker_of_every_cell_ends_up_running_its_own_command(self, manager_factory, worker_probe_factory):
        """The manager starts one live subprocess per worker, each with its own launch context."""
        probe = worker_probe_factory()
        handle = manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=probe.launch_command)]
        )

        records = probe.wait_for_records(4)

        assert sorted(records) == ["0-0", "0-1", "1-0", "1-1"]
        assert all(is_process_running(record["pid"]) for record in records.values())
        assert len({record["pid"] for record in records.values()}) == 4
        for name, record in records.items():
            cell_index, worker_in_cell_index = (int(part) for part in name.split("-"))
            assert record["context"]["cell_index"] == cell_index
            assert record["context"]["worker_in_cell_index"] == worker_in_cell_index
            advertised = ray.get(handle.get_worker_addrs.remote(f"engine-{name}"))["primary"]
            assert record["context"]["self_addrs"]["primary"] == {
                "host": advertised.host,
                "port": advertised.port,
            }

    async def test_the_advertised_address_is_one_the_worker_can_serve_on(self, manager_factory, worker_probe_factory):
        """A worker can bind the port allocated for it, and that endpoint is what the manager advertises."""
        probe = worker_probe_factory(bind_primary=True)
        handle = manager_factory(
            [make_command_spec("engine", num_workers_per_cell=3, launch_command=probe.launch_command)]
        )

        probe.wait_for_records(3)
        addrs = [ray.get(handle.get_worker_addrs.remote(f"engine-0-{index}"))["primary"] for index in range(3)]

        assert len({(addr.host, addr.port) for addr in addrs}) == 3
        for addr in addrs:
            await wait_tcp_ready_async(addr.host, addr.port, timeout=30)

    def test_the_worker_process_gets_the_env_vars_declared_by_its_spec(self, manager_factory, worker_probe_factory):
        """Env vars from the spec are visible inside the launched process."""
        probe = worker_probe_factory(env_names=("MILES_REAL_RAY_PROBE_VAR",))
        manager_factory(
            [
                make_command_spec(
                    "router",
                    launch_command=probe.launch_command,
                    env_var={"MILES_REAL_RAY_PROBE_VAR": "from-spec"},
                )
            ]
        )

        records = probe.wait_for_records(1)

        assert records["0-0"]["env"] == {"MILES_REAL_RAY_PROBE_VAR": "from-spec"}

    def test_a_static_port_reaches_the_worker_unchanged(self, manager_factory, worker_probe_factory):
        """A spec that pins its port keeps it instead of being handed an allocated one."""
        probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec(
                    "router",
                    launch_command=probe.launch_command,
                    port_infos=[PortInfo(name="primary", static_port=21987, allow_dynamic=False)],
                )
            ]
        )

        records = probe.wait_for_records(1)

        assert records["0-0"]["context"]["self_addrs"]["primary"]["port"] == 21987
        assert ray.get(handle.get_worker_addrs.remote("router-0-0"))["primary"].port == 21987

    def test_a_spec_without_cells_launches_no_worker(self, manager_factory, worker_probe_factory):
        """A disabled spec is accepted and simply contributes no workers."""
        disabled_probe = worker_probe_factory()
        enabled_probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec("session-server", num_cells=0, launch_command=disabled_probe.launch_command),
                make_command_spec("router", launch_command=enabled_probe.launch_command),
            ]
        )

        enabled_probe.wait_for_records(1)

        assert disabled_probe.read_records() == {}
        assert ray.get(handle.get_worker_addrs.remote("router-0-0"))["primary"].port > 0


class TestNamedManagerActor:
    async def test_a_driver_knowing_only_the_worker_name_reaches_its_live_endpoint(
        self, manager_factory, worker_probe_factory
    ):
        """The provider resolves a worker's real endpoint through the named manager actor."""
        probe = worker_probe_factory(bind_primary=True)
        manager_factory([make_command_spec("router", launch_command=probe.launch_command)])
        records = probe.wait_for_records(1)

        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        addr = (await provider.get_addrs(worker_name="router-0-0"))["primary"]

        assert isinstance(addr, HostAndPort)
        assert records["0-0"]["context"]["self_addrs"]["primary"] == {"host": addr.host, "port": addr.port}
        await wait_tcp_ready_async(addr.host, addr.port, timeout=30)

    def test_an_unknown_worker_name_is_not_answered_with_another_workers_address(
        self, manager_factory, worker_probe_factory
    ):
        """The lookup forwards the requested name and fails when nothing matches it."""
        probe = worker_probe_factory()
        manager_factory([make_command_spec("router", launch_command=probe.launch_command)])
        probe.wait_for_records(1)

        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(RayWorkerManager.get_handle().get_worker_addrs.remote("router-9-9"))


class TestScaleOnRealRay:
    def test_a_larger_pool_still_gets_disjoint_port_blocks(self, manager_factory, worker_probe_factory):
        """Six workers with multi-port specs must not overlap, including inside reserved blocks."""
        probe = worker_probe_factory()
        manager_factory(
            [
                make_command_spec(
                    "engine",
                    num_cells=3,
                    num_workers_per_cell=2,
                    launch_command=probe.launch_command,
                    port_infos=[
                        PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                        PortInfo(name="nccl", static_port=10000, allow_dynamic=True, num_consecutive=3),
                    ],
                )
            ]
        )

        records = probe.wait_for_records(6)

        reserved: list[int] = []
        for record in records.values():
            addrs = record["context"]["self_addrs"]
            reserved.append(addrs["primary"]["port"])
            reserved.extend(range(addrs["nccl"]["port"], addrs["nccl"]["port"] + 3))
        assert len(reserved) == len(set(reserved))


class TestManagerRelaunchOnRealRay:
    def test_the_well_known_name_can_be_reused_after_the_manager_is_gone(self, manager_factory, worker_probe_factory):
        """A restarted driver must be able to claim the manager name again."""
        first_probe = worker_probe_factory()
        first_handle = manager_factory([make_command_spec("router", launch_command=first_probe.launch_command)])
        first_probe.wait_for_records(1)

        kill_quietly(first_handle)
        wait_until_named_manager_is_gone()

        second_probe = worker_probe_factory()
        manager_factory([make_command_spec("router", launch_command=second_probe.launch_command)])

        assert second_probe.wait_for_records(1)["0-0"]["pid"] != first_probe.read_records()["0-0"]["pid"]


class TestCrossSpecWiringOnRealRay:
    def test_a_dependent_workers_command_sees_the_addresses_of_the_spec_it_depends_on(
        self, manager_factory, worker_probe_factory
    ):
        """A session-server-shaped worker can only find its router because the manager renders after all allocation."""
        router_probe = worker_probe_factory()
        session_probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec("inference-router-0", launch_command=router_probe.launch_command),
                make_command_spec("session-server", num_cells=2, launch_command=session_probe.launch_command),
            ]
        )

        router_probe.wait_for_records(1)
        session_records = session_probe.wait_for_records(2)
        addrs = ray.get(handle.get_addrs.remote())

        router_addr = addrs["inference-router-0"][0]["primary"]
        for record in session_records.values():
            pool_addrs = record["context"]["pool_addrs"]
            assert sorted(pool_addrs) == ["inference-router-0", "session-server"]
            assert pool_addrs["inference-router-0"][0]["primary"] == {
                "host": router_addr.host,
                "port": router_addr.port,
            }
            assert len(pool_addrs["session-server"]) == 2

    def test_each_spec_only_gets_its_own_env(self, manager_factory, worker_probe_factory):
        """Env vars of one spec must not reach another spec's workers."""
        router_probe = worker_probe_factory(env_names=("ROUTER_ONLY", "ENGINE_ONLY"))
        engine_probe = worker_probe_factory(env_names=("ROUTER_ONLY", "ENGINE_ONLY"))
        manager_factory(
            [
                make_command_spec("router", launch_command=router_probe.launch_command, env_var={"ROUTER_ONLY": "r"}),
                make_command_spec("engine", launch_command=engine_probe.launch_command, env_var={"ENGINE_ONLY": "e"}),
            ]
        )

        router_record = router_probe.wait_for_records(1)["0-0"]
        engine_record = engine_probe.wait_for_records(1)["0-0"]

        assert router_record["env"] == {"ROUTER_ONLY": "r", "ENGINE_ONLY": None}
        assert engine_record["env"] == {"ROUTER_ONLY": None, "ENGINE_ONLY": "e"}


class TestMasterPortsOnRealRay:
    def test_all_ranks_of_a_cell_are_launched_with_their_cells_master_endpoint(
        self, manager_factory, worker_probe_factory
    ):
        """Every worker of a cell receives the same master endpoint, allocated once by rank 0."""
        probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec(
                    "engine",
                    num_cells=2,
                    num_workers_per_cell=2,
                    launch_command=probe.launch_command,
                    port_infos=[
                        PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                        PortInfo(
                            name="dist_init", static_port=9000, mode="master", allow_dynamic=True, num_consecutive=4
                        ),
                    ],
                )
            ]
        )

        records = probe.wait_for_records(4)
        masters = {name: record["context"]["self_addrs"]["dist_init"] for name, record in records.items()}

        assert masters["0-0"] == masters["0-1"]
        assert masters["1-0"] == masters["1-1"]
        assert masters["0-0"] != masters["1-0"]
        addrs = ray.get(handle.get_addrs.remote())["engine"]
        assert [sorted(addr) for addr in addrs] == [
            ["dist_init", "primary"],
            ["primary"],
            ["dist_init", "primary"],
            ["primary"],
        ]
        primary_ports = [addr["primary"].port for addr in addrs]
        assert len(set(primary_ports)) == 4
        for master in [masters["0-0"], masters["1-0"]]:
            reserved = range(master["port"], master["port"] + 4)
            assert not set(reserved) & set(primary_ports)


class TestPlacementOnRealRay:
    def test_workers_run_inside_their_placement_group_bundles_with_their_gpu_slice(
        self, manager_factory, worker_probe_factory, placement_group_factory
    ):
        """A pg-bound spec starts every worker on its own bundle and hands it the gpu ids of its slots."""
        probe = worker_probe_factory(env_names=("CUDA_VISIBLE_DEVICES",))
        pgs = {"rollout": placement_group_factory(num_bundles=4, first_gpu_id=10)}
        handle = manager_factory(
            [
                make_command_spec(
                    "engine",
                    num_cells=2,
                    launch_command=probe.launch_command,
                    num_gpus_per_worker=0.5,
                    num_gpu_slots_per_worker=2,
                    pg_name="rollout",
                    env_var={name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
                )
            ],
            pgs,
        )

        records = probe.wait_for_records(2)

        assert records["0-0"]["context"]["gpu_ids"] == [10, 11]
        assert records["1-0"]["context"]["gpu_ids"] == [12, 13]
        assert {record["env"]["CUDA_VISIBLE_DEVICES"] for record in records.values()} == {
            os.environ.get("CUDA_VISIBLE_DEVICES")
        }
        assert len(ray.get(handle.get_addrs.remote())["engine"]) == 2


class TestStopCellOnRealRay:
    def test_stopping_a_cell_ends_its_worker_processes_and_leaves_the_others_alone(
        self, cell_stoppable_manager_factory, worker_probe_factory
    ):
        """Tearing down one cell kills exactly that cell's worker processes."""
        probe = worker_probe_factory()
        manager_handle = cell_stoppable_manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=probe.launch_command)]
        )
        records = probe.wait_for_records(4)
        stopped_pids = [records["1-0"]["pid"], records["1-1"]["pid"]]
        surviving_pids = [records["0-0"]["pid"], records["0-1"]["pid"]]

        ray.get(manager_handle.stop_cell.remote("engine", 1))

        probe.wait_until_gone(stopped_pids)
        assert all(is_process_running(pid) for pid in surviving_pids)


class TestWorkerInfosOnRealRay:
    async def test_a_driver_can_describe_and_reach_the_workers_of_one_cell(
        self, manager_factory, worker_probe_factory
    ):
        """Worker infos survive the trip to the driver, including usable actor handles."""
        probe = worker_probe_factory()
        manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=probe.launch_command)]
        )
        records = probe.wait_for_records(4)

        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        (infos,) = provider.get_worker_infos(cell_ids=["engine-1"])
        handles = provider.get_handles_of_worker_infos(infos)

        assert [info.name for info in infos] == ["engine-1-0", "engine-1-1"]
        assert [info.generation for info in infos] == [1, 1]
        assert [info.gpu_ids for info in infos] == [[], []]
        for worker_in_cell_index, info in enumerate(infos):
            recorded = records[f"1-{worker_in_cell_index}"]["context"]["self_addrs"]["primary"]
            assert {"host": info.self_addrs["primary"].host, "port": info.self_addrs["primary"].port} == recorded
            node_ip = await handles[info.name]._get_node_ip()
            assert info.self_addrs["primary"].host.strip("[]") == node_ip


class TestWorkerDeathOnRealRay:
    async def test_an_actor_dies_with_the_command_it_babysits(self, manager_factory, worker_probe_factory):
        """The actor's whole reason to exist is its subprocess, so its death must be visible to the driver."""
        probe = worker_probe_factory()
        manager_factory([make_command_spec("engine", num_workers_per_cell=2, launch_command=probe.launch_command)])
        probe.wait_for_records(2)
        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        (infos,) = provider.get_worker_infos(cell_ids=["engine-0"])
        handles = [provider.get_handle(info.name) for info in infos]

        # The babysit thread may reach os._exit before this reply is sent, so losing the reply is
        # the death under test arriving early rather than a failure; the loop below still has to
        # observe it, so nothing is taken on faith here.
        try:
            await handles[0].kill_subprocess()
        except WorkerUnreachableError:
            pass

        deadline = time.monotonic() + 60
        while True:
            try:
                await asyncio.wait_for(handles[0]._get_node_ip(), timeout=5)
            except (WorkerUnreachableError, asyncio.TimeoutError):
                break
            assert time.monotonic() < deadline, "the actor of a dead command is still alive"
            await asyncio.sleep(0.5)
        assert await asyncio.wait_for(handles[1]._get_node_ip(), timeout=30)
