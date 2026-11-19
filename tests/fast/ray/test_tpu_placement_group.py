from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule
from tests.fast.utils.workers.fake_specs import FakeCommandSpec, FakeServeSpec

from miles import platforms
from miles.ray import placement_group as pg_mod
from miles.ray.placement_group import PlacementGroupInfo, create_placement_groups
from miles.ray.specs.train import TrainerSpec, compute_trainer_env_vars
from miles.utils.workers import ray_worker_manager
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, WorkerLaunchContext


def _load_tpu_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch_tpu", ModuleType("torch_tpu"))
    spec = importlib.util.spec_from_file_location(
        "_test_tpu_pg_platform", Path(platforms.__file__).with_name("tpu.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tpu = module.TpuPlatform()
    monkeypatch.setattr(platforms, "current_platform", tpu)
    return tpu


class _FakePG:
    def __init__(self, pg_id: str, bundles: list[dict[str, float]], strategy: str, node_ids: list[str]):
        self.id = pg_id
        self.bundles = bundles
        self.strategy = strategy
        self.node_ids = node_ids

    def ready(self):
        return f"ready:{self.id}"


def _install_fake_ray_for_pg(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nodes: list[dict],
) -> list[_FakePG]:
    created_pgs: list[_FakePG] = []

    def _assign_nodes(bundles: list[dict[str, float]], strategy: str) -> list[str]:
        # Pick alive nodes matching any tpu-group-* label in bundle[0]
        group_keys = [k for k in bundles[0] if k.startswith("tpu-group-")]
        eligible = [
            n
            for n in nodes
            if n.get("Alive", True)
            and int(n.get("Resources", {}).get("TPU", 0)) > 0
            and all(n.get("Resources", {}).get(k, 0) > 0 for k in group_keys)
        ]
        assert eligible, f"No eligible node for bundles={bundles}"
        if strategy == "STRICT_PACK":
            return [eligible[0]["NodeID"]] * len(bundles)
        assigned: list[str] = []
        for node in eligible:
            cap = int(node["Resources"]["TPU"])
            assigned.extend([node["NodeID"]] * min(cap, len(bundles) - len(assigned)))
            if len(assigned) == len(bundles):
                break
        assert len(assigned) == len(bundles)
        return assigned

    def fake_placement_group(bundles: list[dict[str, float]], strategy: str = "PACK") -> _FakePG:
        node_ids = _assign_nodes(bundles, strategy)
        pg = _FakePG(f"pg-{len(created_pgs)}", bundles, strategy, node_ids)
        created_pgs.append(pg)
        return pg

    def fake_placement_group_table(pg: _FakePG) -> dict:
        return {
            "placement_group_id": pg.id,
            "state": "CREATED",
            "strategy": pg.strategy,
            "bundles_to_node_id": {i: nid for i, nid in enumerate(pg.node_ids)},
        }

    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        nodes=lambda: nodes,
        get=lambda ref: None,
        util=SimpleNamespace(placement_group_table=fake_placement_group_table),
        remote=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("TPU PG resolution must never spawn InfoActor")),
    )
    monkeypatch.setattr(pg_mod, "ray", fake_ray)
    monkeypatch.setattr(pg_mod, "placement_group", fake_placement_group)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return created_pgs


def _tpu_args(
    *,
    actor_num_nodes: int = 1,
    actor_num_gpus_per_node: int = 8,
    rollout_num_gpus: int = 8,
    num_gpus_per_node: int = 8,
    eval_num_gpus: int = 0,
    debug_train_only: bool = False,
    debug_rollout_only: bool = False,
    rollout_external: bool = False,
    colocate: bool = False,
    use_critic: bool = False,
    deploy_component: str = "all",
) -> Namespace:
    return Namespace(
        actor_num_nodes=actor_num_nodes,
        actor_num_gpus_per_node=actor_num_gpus_per_node,
        rollout_num_gpus=rollout_num_gpus,
        num_gpus_per_node=num_gpus_per_node,
        eval_num_gpus=eval_num_gpus,
        debug_train_only=debug_train_only,
        debug_rollout_only=debug_rollout_only,
        rollout_external=rollout_external,
        colocate=colocate,
        use_critic=use_critic,
        deploy_component=deploy_component,
        raw_megatron=SimpleNamespace(trainers=[SimpleNamespace(role="actor")]),
    )


def test_two_kuberay_slices_pin_actor_and_rollout_separately(monkeypatch):
    _load_tpu_platform(monkeypatch)
    nodes = [
        {"NodeID": "head", "NodeManagerAddress": "10.4.2.9", "Alive": True, "Resources": {"CPU": 8.0}},
        {
            "NodeID": "node-0",
            "NodeManagerAddress": "10.4.17.5",
            "Alive": True,
            "Resources": {"CPU": 60.0, "TPU": 8.0, "tpu-group-0": 1.0},
        },
        {
            "NodeID": "node-1",
            "NodeManagerAddress": "10.4.17.6",
            "Alive": True,
            "Resources": {"CPU": 60.0, "TPU": 8.0, "tpu-group-1": 1.0},
        },
    ]
    created_pgs = _install_fake_ray_for_pg(monkeypatch, nodes=nodes)

    pgs = create_placement_groups(_tpu_args(actor_num_gpus_per_node=8, rollout_num_gpus=8))

    assert len(created_pgs) == 2
    assert pgs["actor"].pg is not pgs["rollout"].pg

    assert pgs["actor"].slice_label == "tpu-group-0"
    assert pgs["actor"].pg_reordered_gpu_ids[0] == 0
    assert pgs["actor"].devices_per_node == 8
    assert pgs["actor"].pg_reordered_node_ips == ["10.4.17.5"] * 8
    assert pgs["actor"].pg_reordered_gpu_ids == list(range(8))
    assert created_pgs[0].strategy == "STRICT_PACK"
    assert created_pgs[0].bundles == [{"CPU": 1, "TPU": 1, "tpu-group-0": 1e-4}] * 8

    assert pgs["rollout"].slice_label == "tpu-group-1"
    assert pgs["rollout"].pg_reordered_gpu_ids[0] == 0
    assert pgs["rollout"].devices_per_node == 8
    assert pgs["rollout"].pg_reordered_node_ips == ["10.4.17.6"] * 8
    assert pgs["rollout"].pg_reordered_gpu_ids == list(range(8))
    assert created_pgs[1].strategy == "STRICT_PACK"
    assert created_pgs[1].bundles == [{"CPU": 1, "TPU": 1, "tpu-group-1": 1e-4}] * 8


def test_single_live_slice_falls_back_to_4_plus_4_subhost_slots(monkeypatch):
    _load_tpu_platform(monkeypatch)
    # Simulates ray-tpu-7x when tpu-group-0 is pending and only tpu-group-1 (2x2x1 = 8 TPUs) is live
    nodes = [
        {"NodeID": "head", "NodeManagerAddress": "10.4.2.9", "Alive": True, "Resources": {"CPU": 8.0}},
        {
            "NodeID": "node-1",
            "NodeManagerAddress": "10.4.17.6",
            "Alive": True,
            "Resources": {"CPU": 60.0, "TPU": 8.0, "tpu-group-1": 1.0},
        },
    ]
    created_pgs = _install_fake_ray_for_pg(monkeypatch, nodes=nodes)

    pgs = create_placement_groups(_tpu_args(actor_num_gpus_per_node=4, rollout_num_gpus=4, num_gpus_per_node=8))

    assert len(created_pgs) == 2
    assert created_pgs[0].strategy == "STRICT_PACK"
    assert created_pgs[1].strategy == "STRICT_PACK"

    # Slot 0: trainer ranks 0..3 on physical host of 8 TPUs
    assert pgs["actor"].slice_label == "tpu-group-1"
    assert pgs["actor"].pg_reordered_gpu_ids[0] == 0
    assert pgs["actor"].devices_per_node == 8
    assert pgs["actor"].pg_reordered_gpu_ids == [0, 1, 2, 3]
    assert pgs["actor"].pg_reordered_node_ips == ["10.4.17.6"] * 4

    # Slot 1: rollout TP=4 on devices 4..7 of the same 8-TPU host
    assert pgs["rollout"].slice_label == "tpu-group-1"
    assert pgs["rollout"].pg_reordered_gpu_ids[0] == 4
    assert pgs["rollout"].devices_per_node == 8
    assert pgs["rollout"].pg_reordered_gpu_ids == [4, 5, 6, 7]
    assert pgs["rollout"].pg_reordered_node_ips == ["10.4.17.6"] * 4


def test_insufficient_slice_capacity_raises_clear_error(monkeypatch):
    _load_tpu_platform(monkeypatch)
    nodes = [
        {
            "NodeID": "node-1",
            "NodeManagerAddress": "10.4.17.6",
            "Alive": True,
            "Resources": {"CPU": 60.0, "TPU": 8.0, "tpu-group-1": 1.0},
        },
    ]
    _install_fake_ray_for_pg(monkeypatch, nodes=nodes)

    with pytest.raises(RuntimeError, match="Insufficient TPU slice capacity"):
        create_placement_groups(_tpu_args(actor_num_gpus_per_node=8, rollout_num_gpus=8))


def test_standalone_tpu_cluster_without_kuberay_labels(monkeypatch):
    _load_tpu_platform(monkeypatch)
    # Single standalone 8-TPU VM (no tpu-group-* resources) running 4+4
    single_vm = [
        {"NodeID": "vm-0", "NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"CPU": 60.0, "TPU": 8.0}}
    ]
    created_pgs = _install_fake_ray_for_pg(monkeypatch, nodes=single_vm)

    pgs = create_placement_groups(_tpu_args(actor_num_gpus_per_node=4, rollout_num_gpus=4, num_gpus_per_node=8))
    assert pgs["actor"].slice_label is None
    assert pgs["actor"].pg_reordered_gpu_ids[0] == 0
    assert pgs["actor"].pg_reordered_gpu_ids == [0, 1, 2, 3]
    assert created_pgs[0].strategy == "STRICT_PACK"
    assert created_pgs[0].bundles == [{"CPU": 1, "TPU": 1}] * 4

    assert pgs["rollout"].slice_label is None
    assert pgs["rollout"].pg_reordered_gpu_ids[0] == 4
    assert pgs["rollout"].pg_reordered_gpu_ids == [4, 5, 6, 7]
    assert created_pgs[1].strategy == "STRICT_PACK"

    # Multi-host 16-TPU standalone uses PACK (not STRICT_PACK)
    two_vms = [
        {"NodeID": "vm-0", "NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"CPU": 60.0, "TPU": 8.0}},
        {"NodeID": "vm-1", "NodeManagerAddress": "10.0.0.2", "Alive": True, "Resources": {"CPU": 60.0, "TPU": 8.0}},
    ]
    created_multi = _install_fake_ray_for_pg(monkeypatch, nodes=two_vms)
    pgs_multi = create_placement_groups(
        _tpu_args(actor_num_nodes=2, actor_num_gpus_per_node=8, rollout_num_gpus=0, debug_train_only=True)
    )
    assert len(created_multi) == 1
    assert created_multi[0].strategy == "PACK"
    assert pgs_multi["actor"].pg_reordered_node_ips == ["10.0.0.1"] * 8 + ["10.0.0.2"] * 8
    assert pgs_multi["rollout"].pg is None


async def test_ray_worker_manager_propagates_tpu_context_and_whole_device_resources(monkeypatch):
    _load_tpu_platform(monkeypatch)
    monkeypatch.setenv("MILES_ACCELERATOR", "tpu")

    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(ray_worker_manager, "ray", fake_ray)

    captured_contexts: list[WorkerLaunchContext] = []

    def _record_env(ctx: WorkerLaunchContext) -> dict[str, str]:
        captured_contexts.append(ctx)
        return {"TEST_ROLE": "trainer" if ctx.devices_per_node == 8 else "other"}

    trainer_spec = FakeServeSpec(
        name="trainer-engine-actor",
        worker_class="tests.fast.utils.workers.fake_specs.FakeServeSpec",
        env_vars=_record_env,
        fixed_scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=4,
            num_gpus_per_worker=0.4,
            num_cpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=4,
            pg_name="actor",
            claim_whole_device=True,
        ),
    )
    rollout_spec = FakeCommandSpec(
        name="inference-engine-all-0-0",
        port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True)],
        env_vars=_record_env,
        fixed_scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0.2,
            num_cpus_per_worker=0.2,
            num_gpu_slots_per_worker=4,
            num_gpus_per_node=8,
            pg_name="rollout",
            claim_whole_device=False,
        ),
        command=lambda ctx: "echo launch",
    )

    pgs = {
        "actor": PlacementGroupInfo(
            pg="actor-pg",
            pg_reordered_bundle_indices=[0, 1, 2, 3],
            pg_reordered_gpu_ids=[0, 1, 2, 3],
            pg_reordered_node_ips=["10.4.17.6"] * 4,
            devices_per_node=8,
            slice_label="tpu-group-1",
        ),
        "rollout": PlacementGroupInfo(
            pg="rollout-pg",
            pg_reordered_bundle_indices=[0, 1, 2, 3],
            pg_reordered_gpu_ids=[4, 5, 6, 7],
            pg_reordered_node_ips=["10.4.17.6"] * 4,
            devices_per_node=8,
            slice_label="tpu-group-1",
        ),
    }

    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), [trainer_spec, rollout_spec], pgs, comm_backend=WorkerCommBackend.RAY)

    # First 4 handles are trainer workers (TPU: 1, no num_gpus); 5th handle is rollout launcher (TPU: 0, no num_gpus)
    assert len(cluster.handles) == 5
    for handle in cluster.handles[:4]:
        assert "num_gpus" not in handle.options
        assert handle.options["resources"] == {"TPU": 1}
        assert handle.options["runtime_env"]["env_vars"]["MILES_ACCELERATOR"] == "tpu"

    rollout_handle = cluster.handles[4]
    assert "num_gpus" not in rollout_handle.options
    assert "resources" not in rollout_handle.options
    assert rollout_handle.options["runtime_env"]["env_vars"]["MILES_ACCELERATOR"] == "tpu"

    # Verify WorkerLaunchContext carried PG placement metadata
    trainer_ctx = captured_contexts[0]
    assert trainer_ctx.bundle_node_ips == ["10.4.17.6"] * 4
    assert trainer_ctx.subhost_device_offset == 0
    assert trainer_ctx.devices_per_node == 8
    assert trainer_ctx.slice_label == "tpu-group-1"

    rollout_ctx = captured_contexts[4]
    assert rollout_ctx.bundle_node_ips == ["10.4.17.6"] * 4
    assert rollout_ctx.subhost_device_offset == 4
    assert rollout_ctx.devices_per_node == 8
    assert rollout_ctx.slice_label == "tpu-group-1"


def test_trainer_spec_env_vars_and_claim_whole_device_on_tpu(monkeypatch):
    _load_tpu_platform(monkeypatch)
    monkeypatch.delenv("MILES_TPU_SHARED_SLICE", raising=False)
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)

    fake_args = SimpleNamespace(
        train_env_vars={"CUSTOM_FLAG": "yes"},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        train_backend="fsdp",
        update_weight_transfer_mode="tpu-sync",
    )
    ctx = WorkerLaunchContext(
        args=fake_args,
        cell_index=0,
        worker_in_cell_index=2,
        num_workers_per_cell=4,
        gpu_ids=[2],
        bundle_node_ips=["10.4.17.6"] * 4,
        subhost_device_offset=0,
        devices_per_node=8,
        slice_label="tpu-group-1",
    )
    env = compute_trainer_env_vars(fake_args, ctx, fp8_scales="0")

    assert env["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] == "1"
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in env
    assert env["TORCH_TPU_TOPOLOGY"] == "1,2,1,2"
    assert env["ALLOW_MULTIPLE_LIBTPU_LOAD"] == "1"
    assert env["LOCAL_RANK"] == "2"
    assert env["LOCAL_WORLD_SIZE"] == "4"
    assert env["MILES_TPU_SLOT_DEVICE"] == "2"
    assert "TPU_VISIBLE_CHIPS" not in env and "TPU_VISIBLE_DEVICES" not in env
    assert env["MILES_TPU_POP_WEBHOOK_ENV"] == "1"
    assert env["CUSTOM_FLAG"] == "yes"

    # TrainerSpec declares claim_whole_device=True
    spec = TrainerSpec.model_construct(
        args=SimpleNamespace(
            trainer_role="actor",
            trainer_actor_index=None,
            train_backend="fsdp",
            indep_dp=False,
        ),
        name="trainer-engine-actor",
        port_infos=[],
        worker_class="miles.backends.fsdp_utils.actor.FSDPTrainRayActor",
        fp8_block_scaling_fp32_scales="0",
    )
    scheduling = spec.scheduling(
        SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=4, critic_num_gpus_per_node=None)
    )
    assert scheduling.claim_whole_device is True


def test_placement_groups_initialize_ray_before_reading_cluster_state(monkeypatch):
    """The driver never calls ray.init(); ray.nodes() is a state API that, unlike placement_group(), won't auto-init."""
    _load_tpu_platform(monkeypatch)
    nodes = [
        {
            "NodeID": "node-1",
            "NodeManagerAddress": "10.4.17.6",
            "Alive": True,
            "Resources": {"CPU": 60.0, "TPU": 8.0, "tpu-group-1": 1.0},
        },
    ]
    _install_fake_ray_for_pg(monkeypatch, nodes=nodes)
    initialized: list[bool] = []

    def nodes_after_init() -> list[dict]:
        assert initialized, "ray.nodes() was called before ray.init()"
        return nodes

    monkeypatch.setattr(pg_mod.ray, "is_initialized", lambda: bool(initialized))
    monkeypatch.setattr(pg_mod.ray, "init", lambda: initialized.append(True), raising=False)
    monkeypatch.setattr(pg_mod.ray, "nodes", nodes_after_init)

    pgs = create_placement_groups(_tpu_args(actor_num_gpus_per_node=4, rollout_num_gpus=4, num_gpus_per_node=8))

    assert initialized == [True]
    assert pgs["rollout"].pg_reordered_gpu_ids == [4, 5, 6, 7]
