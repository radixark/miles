from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml
from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule

from miles.ray.placement_group import PlacementGroupInfo
from miles.ray.specs.inference import specs_inference_engine, specs_router
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_spec import CommandWorkerSpec, LaunchCommandContext, NamedHostAndPorts


@pytest.fixture
def fake_ray_cluster(monkeypatch: pytest.MonkeyPatch) -> FakeRayCluster:
    """In-process stand-in for Ray, letting the manager allocate real addresses without a cluster."""
    import miles.utils.workers.addr_allocator as addr_allocator_mod
    import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(ray_worker_manager_mod, "ray", fake_ray)
    monkeypatch.setattr(addr_allocator_mod, "ray", fake_ray)
    return cluster


def _make_args(
    *, tmp_path: Path, worker_types: list[str], num_gpus: int, num_gpus_per_engine: int, **overrides
) -> Namespace:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(
            server_groups=[
                {"worker_type": worker_type, "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
                for worker_type in worker_types
            ]
        )
    )
    return make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=num_gpus * len(worker_types),
        use_session_server=False,
        **overrides,
    )


def _make_pgs(*, num_slots: int) -> dict[str, PlacementGroupInfo]:
    return {
        "rollout": PlacementGroupInfo(
            pg="fake-pg",
            pg_reordered_bundle_indices=list(range(num_slots)),
            pg_reordered_gpu_ids=list(range(num_slots)),
        )
    }


async def _launch_engines(
    args: Namespace, *, extra_specs: list[CommandWorkerSpec] | None = None
) -> dict[str, LaunchCommandContext]:
    """Run the real launch pipeline and return, per worker name, the context its launch command got."""
    contexts: dict[str, LaunchCommandContext] = {}

    def _recording_spec(spec: CommandWorkerSpec) -> CommandWorkerSpec:
        def _record(ctx: LaunchCommandContext) -> str:
            worker_name = compute_worker_name(
                pool_id=spec.name, cell_index=ctx.cell_index, worker_in_cell_index=ctx.worker_in_cell_index
            )
            contexts[worker_name] = ctx
            return f"launch {worker_name}"

        return spec.model_copy(update={"launch_command": _record})

    specs = [_recording_spec(spec) for spec in [*specs_inference_engine(args), *(extra_specs or [])]]
    num_slots = sum(
        spec.scheduling.num_cells * spec.scheduling.num_workers_per_cell * spec.scheduling.num_gpu_slots_per_worker
        for spec in specs
    )

    manager = RayWorkerManager()
    await manager.init(specs, _make_pgs(num_slots=max(num_slots, 1)))

    return contexts


_PORT_NAMES = ("primary", "dist_init", "nccl", "engine_info_bootstrap", "disaggregation_bootstrap")


def _all_ports(addrs: NamedHostAndPorts) -> list[int]:
    return [addrs[name].port for name in _PORT_NAMES if name in addrs]


class TestAddressingOfLaunchedEngines:
    async def test_single_node_8_cards_tp1(self, tmp_path: Path, fake_ray_cluster: FakeRayCluster):
        """Eight single-gpu engines on one node get complete, mutually distinct addressing."""
        args = _make_args(tmp_path=tmp_path, worker_types=["regular"], num_gpus=8, num_gpus_per_engine=1)

        contexts = await _launch_engines(args)

        assert sorted(contexts) == [f"inference-engine-0-0-{cell_index}-0" for cell_index in range(8)]
        issued: list[int] = []
        for ctx in contexts.values():
            addrs = ctx.self_addrs
            assert sorted(addrs) == ["dist_init", "engine_info_bootstrap", "nccl", "primary"]
            assert all(addr.host == "10.0.0.1" for addr in addrs.values())
            ports = _all_ports(addrs)
            assert len(set(ports)) == len(ports), f"an engine reused a port: {addrs}"
            issued.extend(ports)

        assert len(set(issued)) == len(issued), f"port collision across engines on the same node: {issued}"

    async def test_prefill_worker_gets_disagg_bootstrap_port(self, tmp_path: Path, fake_ray_cluster: FakeRayCluster):
        """A prefill engine's disaggregation bootstrap port is distinct from its other ports."""
        args = _make_args(tmp_path=tmp_path, worker_types=["prefill", "decode"], num_gpus=4, num_gpus_per_engine=1)

        contexts = await _launch_engines(args)

        prefill = contexts["inference-engine-0-0-0-0"].self_addrs
        assert "disaggregation_bootstrap" in prefill
        ports = _all_ports(prefill)
        assert len(set(ports)) == len(ports)

    async def test_regular_worker_does_not_get_disagg_bootstrap_port(
        self, tmp_path: Path, fake_ray_cluster: FakeRayCluster
    ):
        """Only prefill engines carry a disaggregation bootstrap port; others must not reserve one."""
        args = _make_args(tmp_path=tmp_path, worker_types=["regular"], num_gpus=2, num_gpus_per_engine=1)

        contexts = await _launch_engines(args)

        assert all("disaggregation_bootstrap" not in ctx.self_addrs for ctx in contexts.values())

    async def test_multinode_cell_shares_a_dist_init_addr_on_the_primary_node(
        self, tmp_path: Path, fake_ray_cluster: FakeRayCluster
    ):
        """A 2-node engine (16 gpus, 8 per node) gets one dist_init addr, taken from its first node-rank."""
        fake_ray_cluster.use_node_ips("10.0.0.42", "10.0.0.43")
        args = _make_args(tmp_path=tmp_path, worker_types=["regular"], num_gpus=16, num_gpus_per_engine=16)

        contexts = await _launch_engines(args)

        first = contexts["inference-engine-0-0-0-0"].self_addrs
        second = contexts["inference-engine-0-0-0-1"].self_addrs
        assert (first["primary"].host, second["primary"].host) == ("10.0.0.42", "10.0.0.43")
        assert first["dist_init"] == second["dist_init"]
        assert first["dist_init"].host == "10.0.0.42"

    async def test_the_dist_init_block_is_reserved_away_from_every_other_port(
        self, tmp_path: Path, fake_ray_cluster: FakeRayCluster
    ):
        """dist_init reserves a whole block, so no later engine may be handed a port inside it."""
        args = _make_args(tmp_path=tmp_path, worker_types=["regular"], num_gpus=8, num_gpus_per_engine=1)

        contexts = await _launch_engines(args)

        reserved: set[int] = set()
        for ctx in contexts.values():
            dist_init_port = ctx.self_addrs["dist_init"].port
            reserved.update(range(dist_init_port, dist_init_port + 31))
        other_ports = [
            addr.port for ctx in contexts.values() for name, addr in ctx.self_addrs.items() if name != "dist_init"
        ]
        assert all(port not in reserved for port in other_ports)


class TestAddressingAcrossSpecs:
    async def test_engines_never_share_a_port_with_the_router(self, tmp_path: Path, fake_ray_cluster: FakeRayCluster):
        """Every spec draws from the same node cursor, so a router and an engine can never collide."""
        args = _make_args(tmp_path=tmp_path, worker_types=["regular"], num_gpus=4, num_gpus_per_engine=1)

        contexts = await _launch_engines(args, extra_specs=specs_router(args))

        engine_ports = [
            port
            for name, ctx in contexts.items()
            if name.startswith("inference-engine")
            for port in _all_ports(ctx.self_addrs)
        ]
        router_ports = [
            port
            for name, ctx in contexts.items()
            if name.startswith("inference-router")
            for port in [ctx.self_addrs["primary"].port, ctx.self_addrs["prometheus"].port]
        ]
        assert set(engine_ports).isdisjoint(router_ports)
