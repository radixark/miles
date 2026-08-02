from __future__ import annotations


import pytest
import ray
from tests.fast.utils.workers.real_ray.conftest import (
    is_process_running,
    kill_quietly,
    make_command_spec,
    wait_until_named_manager_is_gone,
)

from miles.utils.http_utils import wait_tcp_ready
from miles.utils.workers.ray_worker_manager import RayWorkerManager
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
            advertised = ray.get(handle.get_worker_addr.remote(f"engine-{name}"))
            assert record["context"]["self_addrs"]["primary"] == {
                "host": advertised.host,
                "port": advertised.port,
            }

    def test_the_advertised_address_is_one_the_worker_can_serve_on(self, manager_factory, worker_probe_factory):
        """A worker can bind the port allocated for it, and that endpoint is what the manager advertises."""
        probe = worker_probe_factory(bind_primary=True)
        handle = manager_factory(
            [make_command_spec("engine", num_workers_per_cell=3, launch_command=probe.launch_command)]
        )

        probe.wait_for_records(3)
        addrs = [ray.get(handle.get_worker_addr.remote(f"engine-0-{index}")) for index in range(3)]

        assert len({(addr.host, addr.port) for addr in addrs}) == 3
        for addr in addrs:
            wait_tcp_ready(addr.host, addr.port, timeout=30)

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
        assert ray.get(handle.get_worker_addr.remote("router-0-0")).port == 21987

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
        assert ray.get(handle.get_worker_addr.remote("router-0-0")).port > 0


class TestNamedManagerActor:
    async def test_a_driver_knowing_only_the_worker_name_reaches_its_live_endpoint(
        self, manager_factory, worker_probe_factory
    ):
        """The provider resolves a worker's real endpoint through the named manager actor."""
        probe = worker_probe_factory(bind_primary=True)
        manager_factory([make_command_spec("router", launch_command=probe.launch_command)])
        records = probe.wait_for_records(1)

        addr = await RayWorkerProvider.create().get_addr(worker_name="router-0-0")

        assert isinstance(addr, HostAndPort)
        assert records["0-0"]["context"]["self_addrs"]["primary"] == {"host": addr.host, "port": addr.port}
        wait_tcp_ready(addr.host, addr.port, timeout=30)

    def test_an_unknown_worker_name_is_not_answered_with_another_workers_address(
        self, manager_factory, worker_probe_factory
    ):
        """The lookup forwards the requested name and fails when nothing matches it."""
        probe = worker_probe_factory()
        manager_factory([make_command_spec("router", launch_command=probe.launch_command)])
        probe.wait_for_records(1)

        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(RayWorkerManager.get_handle().get_worker_addr.remote("router-9-9"))


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
