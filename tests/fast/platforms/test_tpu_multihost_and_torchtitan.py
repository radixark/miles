import importlib.util
import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Shard

from miles import platforms
from miles.backends.torchtitan_utils import hf_weight_iterator
from miles.platforms import tpu_shared_slice
from miles.ray import placement_group


def _install_fake_torch_tpu(monkeypatch: pytest.MonkeyPatch):
    torch_tpu_mod = ModuleType("torch_tpu")
    internal_mod = ModuleType("torch_tpu._internal")
    dist_pkg = ModuleType("torch_tpu._internal.distributed")
    tpu_dist = ModuleType("torch_tpu._internal.distributed.tpu_distributed")
    captured_join_env: dict[str, str] = {}

    def global_device_id() -> int:
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            captured_join_env[key] = os.environ[key]
        return 100 + int(os.environ["RANK"])

    tpu_dist.global_device_id = global_device_id
    tpu_dist.create_process_group = lambda store, rank, size, timeout: SimpleNamespace(rank=rank, size=size)
    dist_pkg.tpu_distributed = tpu_dist
    monkeypatch.setitem(sys.modules, "torch_tpu", torch_tpu_mod)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", internal_mod)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.distributed", dist_pkg)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.distributed.tpu_distributed", tpu_dist)
    return captured_join_env


class _FakeTCPStore:
    def __init__(self, host, port, size, is_master, timeout):
        self._kv = {f"device:{r}": str(100 + r).encode("utf-8") for r in range(size)}

    def set(self, key: str, val: str) -> None:
        self._kv[key] = val.encode("utf-8")

    def get(self, key: str) -> bytes:
        return self._kv[key]


def test_tpu_shared_slice_join_wraps_local_rank_on_multihost_and_preserves_single_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_torch_tpu(monkeypatch)
    monkeypatch.setattr(tpu_shared_slice.dist, "TCPStore", _FakeTCPStore)
    monkeypatch.setattr(tpu_shared_slice, "PrefixStore", lambda prefix, store: store)
    monkeypatch.setattr(
        tpu_shared_slice,
        "receiver_contract",
        lambda: SimpleNamespace(BACKEND_NAME="tpu_shared_slice"),
    )

    # Multi-host: 64-TensorCore slice (8 hosts x 8 devices), 32-rank trainer job, rank 19 (host 2, local 3).
    shared_64 = tpu_shared_slice.SharedSlice(rank_offset=0, size=64, store_host="10.0.0.1", store_port=29600)
    tpu_shared_slice.join(shared_64, job_rank=19, job_size=32)
    assert captured == {"RANK": "19", "LOCAL_RANK": "3", "WORLD_SIZE": "64", "LOCAL_WORLD_SIZE": "8"}
    assert os.environ["RANK"] == "19"
    assert os.environ["LOCAL_RANK"] == "3"
    assert os.environ["WORLD_SIZE"] == "32"
    assert os.environ["LOCAL_WORLD_SIZE"] == "8"
    assert len(os.environ["TORCH_TPU_JOB_DEVICE_IDS"].split(",")) == 32

    # Single-host: 8-TensorCore slice, 4-rank trainer job, rank 3 (unchanged from single-host behavior).
    shared_8 = tpu_shared_slice.SharedSlice(rank_offset=0, size=8, store_host="127.0.0.1", store_port=29600)
    tpu_shared_slice.join(shared_8, job_rank=3, job_size=4)
    assert captured == {"RANK": "3", "LOCAL_RANK": "3", "WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8"}
    assert os.environ["RANK"] == "3"
    assert os.environ["LOCAL_RANK"] == "3"
    assert os.environ["WORLD_SIZE"] == "4"
    assert os.environ["LOCAL_WORLD_SIZE"] == "4"


def test_placement_group_sort_key_respects_topological_node_ips(monkeypatch: pytest.MonkeyPatch) -> None:
    bundles = [
        (0, "10.0.0.20", 1),
        (1, "10.0.0.5", 0),
        (2, "10.0.0.20", 0),
        (3, "10.0.0.12", 0),
    ]
    monkeypatch.delenv(placement_group.MILES_TPU_TRAINER_NODE_IPS_ENV, raising=False)
    default_order = [b[0] for b in sorted(bundles, key=placement_group.sort_key)]
    assert default_order == [1, 3, 2, 0]

    # TPU slice (z, y, x) host coordinate order differs from numerical IP order.
    monkeypatch.setenv(placement_group.MILES_TPU_TRAINER_NODE_IPS_ENV, "10.0.0.20, 10.0.0.5, 10.0.0.12")
    topo_order = [b[0] for b in sorted(bundles, key=placement_group.sort_key)]
    assert topo_order == [2, 0, 1, 3]


def test_grouped_experts_intra_host_subgroup_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """world=32, ep_size=4 -> span=8 ranks per subgroup; rank 9 (subgroup 1) gathers experts [8..15] of 32."""
    world, ep_size, num_experts = 32, 4, 32
    span = world // ep_size  # 8 ranks per host
    hf_weight_iterator._expert_gather_groups.cache_clear()
    created_groups: list[list[int]] = []

    def fake_new_group(ranks: list[int]):
        created_groups.append(list(ranks))
        return tuple(ranks)

    def fake_all_gather_into_tensor(out: torch.Tensor, local: torch.Tensor, group=None):
        assert isinstance(group, tuple) and len(group) == span
        for slot, r in enumerate(group):
            out[slot] = torch.full_like(local[0], float(r))

    monkeypatch.setattr(hf_weight_iterator.dist, "new_group", fake_new_group)
    monkeypatch.setattr(hf_weight_iterator.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)

    class _FakeGroupedDTensor:
        def __init__(self, local: torch.Tensor):
            self._local = local
            self.placements = (Shard(0),)
            self.device_mesh = SimpleNamespace(size=lambda: world)
            self.shape = torch.Size([num_experts, 4, 3])

        def to_local(self) -> torch.Tensor:
            return self._local

    # Monkeypatch isinstance(..., DTensor) target in module to accept our lightweight shard wrapper without PG init.
    monkeypatch.setattr(hf_weight_iterator, "DTensor", _FakeGroupedDTensor)
    local_shard = torch.full((1, 4, 3), 9.0)
    converted = {"model.language_model.layers.0.mlp.experts.gate_up_proj": _FakeGroupedDTensor(local_shard)}
    exported = dict(
        hf_weight_iterator._grouped_experts_by_id(
            converted,
            titan_shape=torch.Size([num_experts, 2, 2, 3]),
            rank=9,
            world=world,
            ep_size=ep_size,
        )
    )
    assert created_groups == [list(range(g * 8, (g + 1) * 8)) for g in range(4)]
    expected_ids = list(range(8, 16))
    assert [int(k.split(".experts.")[1].split(".")[0]) for k in exported][::2] == expected_ids
    for offset, exp_id in enumerate(expected_ids):
        gate = exported[f"model.language_model.layers.0.mlp.experts.{exp_id}.gate_proj.weight"]
        assert torch.all(gate == float(8 + offset))

    # Verify _resolve_sglang_ep_size handles both Namespace(sglang_ep_size=4) and TrainerConfig-style args.sglang.
    assert hf_weight_iterator._resolve_sglang_ep_size(Namespace(sglang_ep_size=4)) == 4
    sglang_cfg = SimpleNamespace(base_args={"ep_size": 4}, actor_value=lambda k: 4 if k == "ep_size" else None)
    assert hf_weight_iterator._resolve_sglang_ep_size(SimpleNamespace(sglang=sglang_cfg)) == 4


def test_tpu_platform_disables_fused_adamw_and_selects_foreach_on_hbm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load TpuPlatform with stub torch_tpu and verify capability attributes.
    monkeypatch.setitem(sys.modules, "torch_tpu", ModuleType("torch_tpu"))
    spec = importlib.util.spec_from_file_location("_test_tpu_plat", Path(platforms.__file__).with_name("tpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert platforms.Platform.fused_adamw is True
    assert platforms.Platform.max_local_devices is None
    assert mod.TpuPlatform.fused_adamw is False
    assert mod.TpuPlatform.max_local_devices == 8

    # Verify config._optim_config against both upstream pytorch/torchtitan (foreach/fused) and
    # google-pytorch/torchtitan (implementation="foreach") signatures without requiring real torchtitan install.
    @dataclass
    class _UpstreamAdamWConfig:
        pattern: str = r".*"
        lr: float = 1e-6
        betas: tuple[float, float] = (0.9, 0.98)
        eps: float = 1e-8
        weight_decay: float = 0.1
        foreach: bool | None = None
        fused: bool | None = True

    @dataclass
    class _ForkAdamWConfig:
        pattern: str = r".*"
        lr: float = 1e-6
        betas: tuple[float, float] = (0.9, 0.98)
        eps: float = 1e-8
        weight_decay: float = 0.1
        implementation: str = "fused"

    @dataclass
    class _LRConfig:
        warmup_steps: int = 0
        decay_type: str = "linear"
        min_lr_factor: float = 1.0

    @dataclass
    class _OptimContainerConfig:
        optimizers: list
        implementation: str = "fused"

    @dataclass
    class _OptimConfig:
        optimizer: object
        lr_scheduler: object
        max_norm: float

    # Stub minimal torchtitan modules if torchtitan is not installed in the CPU unit-test environment.
    stub_names = [
        "torchtitan",
        "torchtitan.components",
        "torchtitan.components.optim",
        "torchtitan.config",
        "torchtitan.config.configs",
        "torchtitan.config.parallelism",
        "torchtitan.config.transform",
        "torchtitan.config.transform.apply",
        "torchtitan.distributed",
        "torchtitan.distributed.activation_remat",
        "torchtitan.distributed.parallelism_context",
        "torchtitan.distributed.spmd_types",
        "torchtitan.models",
        "torchtitan.models.base",
        "torchtitan.models.common",
        "torchtitan.models.common.attention",
        "torchtitan.models.common.attention.cp_attention",
        "torchtitan.models.common.linear",
        "torchtitan.models.common.moe",
        "miles.backends.torchtitan_utils.components",
        "miles.backends.torchtitan_utils.lora",
        "miles.backends.torchtitan_utils.loss",
        "miles.backends.torchtitan_utils.trainer",
    ]
    for name in stub_names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    sys.modules["torchtitan.components.optim"].AdamW = SimpleNamespace(Config=_UpstreamAdamWConfig)
    sys.modules["torchtitan.components.optim"].LRSchedulersContainer = SimpleNamespace(Config=_LRConfig)
    sys.modules["torchtitan.components.optim"].Optim = SimpleNamespace(Config=_OptimConfig)
    sys.modules["torchtitan.components.optim"].OptimizersContainer = SimpleNamespace(Config=_OptimContainerConfig)
    sys.modules["torchtitan.config"].TrainingConfig = object
    sys.modules["torchtitan.config.configs"].DebugConfig = object
    sys.modules["torchtitan.config.parallelism"].ParallelismConfig = object
    sys.modules["torchtitan.config.transform"].ContextParallelTransform = object
    sys.modules["torchtitan.config.transform.apply"].transform_model_config_ = lambda m, _t: m
    sys.modules["torchtitan.distributed.activation_remat"].FullActivationRematConfig = object
    sys.modules["torchtitan.distributed.parallelism_context"].MeshAxisName = object
    sys.modules["torchtitan.distributed.spmd_types"].spmd_mesh_group = lambda _a: None
    sys.modules["torchtitan.models.base"].BaseModel = SimpleNamespace(Config=object)

    class _InnerAttn:
        class Config:
            pass

    class _Ulysses:
        class Config:
            pass

    sys.modules["torchtitan.models.common.attention"].FlexInnerAttention = _InnerAttn
    sys.modules["torchtitan.models.common.attention"].GQAttention = _InnerAttn
    sys.modules["torchtitan.models.common.attention"].InnerAttention = _InnerAttn
    sys.modules["torchtitan.models.common.attention"].MLAAttention = _InnerAttn
    sys.modules["torchtitan.models.common.attention.cp_attention"].AllGatherKVFlexInnerAttention = _InnerAttn
    sys.modules["torchtitan.models.common.attention.cp_attention"].Ulysses = _Ulysses
    sys.modules["torchtitan.models.common.linear"].Linear = _InnerAttn
    sys.modules["torchtitan.models.common.moe"].MoE = _InnerAttn
    sys.modules["miles.backends.torchtitan_utils.components"].TiedCheckpointManager = SimpleNamespace(Config=object)
    sys.modules["miles.backends.torchtitan_utils.loss"].RLLossAdapter = SimpleNamespace(Config=object)
    sys.modules["miles.backends.torchtitan_utils.trainer"].TitanTrainer = SimpleNamespace(Config=object)

    cfg_path = Path(hf_weight_iterator.__file__).with_name("config.py")
    cfg_spec = importlib.util.spec_from_file_location("_test_titan_cfg", cfg_path)
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)

    def _make_args(*, offload: bool):
        return SimpleNamespace(
            backend=SimpleNamespace(
                lr_decay_style="constant",
                lr_warmup_iters=0,
                lr=1e-6,
                min_lr=1e-6,
                adam_beta1=0.9,
                adam_beta2=0.98,
                adam_eps=1e-8,
                weight_decay=0.1,
                clip_grad=1.0,
                optimizer_cpu_offload=offload,
            )
        )

    # On TPU HBM (fused_adamw=False, offload=False): upstream AdamW.Config gets foreach=True, fused=False.
    monkeypatch.setattr(platforms.current_platform, "fused_adamw", False)
    opt_hbm = cfg_mod._optim_config(_make_args(offload=False))
    assert opt_hbm.optimizer.optimizers[0].foreach is True
    assert opt_hbm.optimizer.optimizers[0].fused is False
    assert opt_hbm.optimizer.implementation == "foreach"

    # Under CPU offload (offload=True): leave default AdamW settings untouched.
    opt_cpu = cfg_mod._optim_config(_make_args(offload=True))
    assert opt_cpu.optimizer.optimizers[0].foreach is None
    assert opt_cpu.optimizer.optimizers[0].fused is True

    # Fork signature (implementation="foreach"):
    monkeypatch.setattr(cfg_mod, "AdamW", SimpleNamespace(Config=_ForkAdamWConfig))
    opt_fork = cfg_mod._optim_config(_make_args(offload=False))
    assert opt_fork.optimizer.optimizers[0].implementation == "foreach"
