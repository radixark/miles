import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from miles import platforms
from miles.backends.sglang_utils import arguments as sglang_arguments
from miles.backends.sglang_utils import constants as sglang_constants
from miles.backends.sglang_utils import router_args_utils, server_args_utils, sglang_engine, sglang_router_api_client
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.router.backend import RouterKind
from miles.utils import arguments as miles_arguments
from miles.utils import dumper_utils
from miles.utils.args.configs import backend_fields, router
from miles.utils.chat_template_utils import deepseek as chat_deepseek
from miles.utils.chat_template_utils import template as chat_template
from miles.utils.output_store import config as output_store_config


def _load_tpu_module(monkeypatch: pytest.MonkeyPatch, *, torch_tpu_stub: ModuleType | None = None) -> ModuleType:
    monkeypatch.setitem(sys.modules, "torch_tpu", torch_tpu_stub or ModuleType("torch_tpu"))
    spec = importlib.util.spec_from_file_location("_test_tpu_platform", Path(platforms.__file__).with_name("tpu.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda or tpu would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda or tpu"):
        platforms._select_platform()


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper


def test_cuda_platform_defaults_are_preserved(monkeypatch):
    cuda = platforms.Platform()

    assert cuda.device_type == "cuda"
    assert cuda.ray_accelerator_resource == "GPU"
    assert cuda.default_distributed_backend == "nccl"
    assert cuda.default_attn_implementation == "flash_attention_2"
    assert cuda.supports_pin_memory is True
    assert cuda.supports_colocate is True
    assert cuda.separate_role_placement_groups is False
    assert cuda.torch_memory_history is True
    assert cuda.weight_transfer_modes is None
    assert cuda.noset_visible_devices_env_vars == {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

    assert cuda.ray_actor_accelerator_options(0.4, claim_whole_device=True) == {"num_gpus": 0.4}
    assert cuda.ray_actor_accelerator_options(0.2, claim_whole_device=False) == {"num_gpus": 0.2}
    assert cuda.ray_actor_accelerator_options(0) == {"num_gpus": 0}

    bundles, strategy, slice_label, offset = cuda.create_placement_group_bundles(
        role="actor", num_devices=4, devices_per_node=8, allocated_slices={}, ray_nodes=[]
    )
    assert bundles == [{"GPU": 1, "CPU": 1}] * 4
    assert strategy == "PACK"
    assert slice_label is None
    assert offset == 0

    assert (
        cuda.worker_distributed_env_vars(
            bundle_node_ips=["10.0.0.1"] * 4,
            rank=0,
            local_rank=0,
            world_size=4,
            local_world_size=4,
            devices_per_node=8,
        )
        == {}
    )
    assert cuda.rollout_worker_env_vars(bundle_node_ips=["10.0.0.1"] * 4, tp_size=4, devices_per_node=8) == {}

    called: list[bool] = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: called.append(True))
    cuda.empty_cache()
    assert called == [True]


def test_tpu_platform_capabilities_and_methods(monkeypatch):
    tpu_mod = _load_tpu_module(monkeypatch)
    tpu = tpu_mod.TpuPlatform()

    assert tpu.device_type == "tpu"
    assert tpu.ray_accelerator_resource == "TPU"
    assert tpu.default_distributed_backend == "tpu_dist"
    assert tpu.default_attn_implementation == "sdpa"
    assert tpu.supports_pin_memory is False
    assert tpu.supports_colocate is False
    assert tpu.separate_role_placement_groups is True
    assert tpu.torch_memory_history is False
    assert tpu.weight_transfer_modes == frozenset({"ici", "tpu-sync"})
    assert tpu.noset_visible_devices_env_vars == {"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1"}

    # empty_cache is an explicit no-op on TPU so compiled XLA caches are never evicted
    called: list[bool] = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: called.append(True))
    tpu.empty_cache()
    assert called == []

    # CPU-only KubeRay head pod safety for current_device()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    assert tpu.current_device() == torch.device("cpu")

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch, "device", lambda dev_type, index=None: (dev_type, index))
    assert tpu.current_device() == ("tpu", 0)

    # Ray actor options: trainer claims whole TPU, server launcher claims 0 TPUs
    assert tpu.ray_actor_accelerator_options(0.4, claim_whole_device=True) == {"resources": {"TPU": 1}}
    assert tpu.ray_actor_accelerator_options(1, claim_whole_device=True) == {"resources": {"TPU": 1}}
    assert tpu.ray_actor_accelerator_options(0.2, claim_whole_device=False) == {}
    assert tpu.ray_actor_accelerator_options(0, claim_whole_device=True) == {}


def test_tpu_local_device_id_resolution(monkeypatch):
    tpu = _load_tpu_module(monkeypatch).TpuPlatform()

    def unused() -> list[str]:
        raise AssertionError("ray is not asked when the index is already known")

    assert tpu.local_device_id(worker_in_pod_index=3, ray_device_ids=unused) == 3

    monkeypatch.setenv("LOCAL_RANK", "6")
    assert tpu.local_device_id(worker_in_pod_index=None, ray_device_ids=unused) == 6

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert tpu.local_device_id(worker_in_pod_index=None, ray_device_ids=lambda: ["5"]) == 5
    with pytest.raises(RuntimeError, match="Ray assigned this worker no TPU"):
        tpu.local_device_id(worker_in_pod_index=None, ray_device_ids=list)


def test_tpu_module_pops_gke_webhook_env_before_importing_torch_tpu(monkeypatch, tmp_path):
    webhook_vars = (
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_WORKER_ID",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "TPU_RUNTIME_METRICS_PORTS",
        "TORCH_TPU_DEVICES_PER_HOST",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
    )
    for var in webhook_vars:
        monkeypatch.setenv(var, "stale-webhook-val")

    log_dir = tmp_path / "tpu_logs" / "rank_4"
    monkeypatch.setenv("MILES_TPU_POP_WEBHOOK_ENV", "1")
    monkeypatch.setenv("TPU_LOG_DIR", str(log_dir))
    monkeypatch.setenv("MILES_TPU_SLOT_DEVICE", "5")
    visibility_vars = ("TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES")
    for var in visibility_vars:
        monkeypatch.setenv(var, "0,1,2,3,4,5,6,7")  # restored after the test; the module overwrites it

    # Capture env inside a loader that runs when `import torch_tpu` executes
    seen_during_torch_tpu_import: dict[str, str | None] = {}

    class _ProbeLoader:
        def create_module(self, spec):
            return ModuleType(spec.name)

        def exec_module(self, module):
            import os

            for var in (*webhook_vars, *visibility_vars):
                seen_during_torch_tpu_import[var] = os.environ.get(var)

    fake_spec = importlib.machinery.ModuleSpec("torch_tpu", _ProbeLoader())

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch_tpu":
                return fake_spec
            return None

    monkeypatch.delitem(sys.modules, "torch_tpu", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])

    spec = importlib.util.spec_from_file_location("_test_tpu_webhook", Path(platforms.__file__).with_name("tpu.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert all(seen_during_torch_tpu_import[var] is None for var in webhook_vars)
    assert all(seen_during_torch_tpu_import[var] == "5" for var in visibility_vars)
    assert log_dir.is_dir()


def test_tpu_worker_distributed_env_vars_full_multi_subhost_and_shared_slice(monkeypatch):
    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.delenv("MILES_TPU_SHARED_SLICE", raising=False)
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)

    # Full-host 8-rank TPU
    full = tpu.worker_distributed_env_vars(
        bundle_node_ips=["10.4.17.6"] * 8,
        rank=3,
        local_rank=3,
        world_size=8,
        local_world_size=8,
        devices_per_node=8,
    )
    assert full["LOCAL_RANK"] == "3"
    assert full["LOCAL_WORLD_SIZE"] == "8"
    assert full["TPU_WORKER_HOSTNAMES"] == "10.4.17.6"
    assert full["TORCH_TPU_TOPOLOGY"] == "2,2,1,2"
    assert full["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == ",".join(f"10.4.17.6:{8471 + r}" for r in range(8))
    assert "ALLOW_MULTIPLE_LIBTPU_LOAD" not in full
    assert "MILES_TPU_POP_WEBHOOK_ENV" not in full

    # Multi-host 16-rank TPU
    multi = tpu.worker_distributed_env_vars(
        bundle_node_ips=["10.4.17.6"] * 8 + ["10.4.17.7"] * 8,
        rank=10,
        local_rank=2,
        world_size=16,
        local_world_size=8,
        devices_per_node=8,
    )
    assert multi["LOCAL_RANK"] == "2"
    assert multi["LOCAL_WORLD_SIZE"] == "8"
    assert multi["TPU_WORKER_HOSTNAMES"] == "10.4.17.6,10.4.17.7"
    assert multi["TORCH_TPU_TOPOLOGY"] == "2,2,2,2"
    assert len(multi["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == 16
    assert multi["TORCH_TPU_SLICEBUILDER_ADDRESSES"].startswith("10.4.17.6:8471,")
    assert multi["TORCH_TPU_SLICEBUILDER_ADDRESSES"].endswith(",10.4.17.7:8478")

    # Sub-host 4-rank slot 0 (offset=0)
    sub0 = tpu.worker_distributed_env_vars(
        bundle_node_ips=["10.4.17.6"] * 4,
        rank=2,
        local_rank=2,
        world_size=4,
        local_world_size=4,
        devices_per_node=8,
        subhost_offset=0,
    )
    assert sub0["TORCH_TPU_TOPOLOGY"] == "1,2,1,2"
    assert sub0["ALLOW_MULTIPLE_LIBTPU_LOAD"] == "1"
    assert sub0["LOCAL_RANK"] == "2"
    assert sub0["LOCAL_WORLD_SIZE"] == "4"
    assert sub0["MILES_TPU_SLOT_DEVICE"] == "2"
    # Ray indexes the startup TPU_VISIBLE_CHIPS with the TPU id it assigned, so it must never ride in runtime_env
    assert "TPU_VISIBLE_CHIPS" not in sub0 and "TPU_VISIBLE_DEVICES" not in sub0
    assert sub0["MEGASCALE_COORDINATOR_ADDRESS"] == "127.0.0.1:21000"
    assert sub0["MEGASCALE_PORT"] == "21004"
    assert sub0["MEGASCALE_DEBUG_PORT"] == "21005"
    assert sub0["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == ",".join(f"localhost:{21300 + r}" for r in range(4))
    assert sub0["MILES_TPU_POP_WEBHOOK_ENV"] == "1"

    # Sub-host 4-rank slot 1 (offset=4)
    sub1 = tpu.worker_distributed_env_vars(
        bundle_node_ips=["10.4.17.6"] * 4,
        rank=1,
        local_rank=1,
        world_size=4,
        local_world_size=4,
        devices_per_node=8,
        subhost_offset=4,
    )
    assert sub1["TORCH_TPU_TOPOLOGY"] == "1,2,1,2"
    assert sub1["LOCAL_RANK"] == "5"
    assert sub1["LOCAL_WORLD_SIZE"] == "8"
    assert sub1["MILES_TPU_SLOT_DEVICE"] == "5"
    # Ray indexes the startup TPU_VISIBLE_CHIPS with the TPU id it assigned, so it must never ride in runtime_env
    assert "TPU_VISIBLE_CHIPS" not in sub1 and "TPU_VISIBLE_DEVICES" not in sub1
    assert sub1["MEGASCALE_COORDINATOR_ADDRESS"] == "127.0.0.1:21080"
    assert sub1["MEGASCALE_PORT"] == "21082"
    assert sub1["MEGASCALE_DEBUG_PORT"] == "21083"
    assert sub1["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == ",".join(f"localhost:{21380 + r}" for r in range(4))
    assert sub1["TORCHINDUCTOR_CACHE_DIR"] != sub0["TORCHINDUCTOR_CACHE_DIR"]
    assert sub1["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] != sub0["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"]

    # Sub-host 2-rank topology
    sub2 = tpu.worker_distributed_env_vars(
        bundle_node_ips=["10.4.17.6"] * 2,
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        devices_per_node=8,
        subhost_offset=0,
    )
    assert sub2["TORCH_TPU_TOPOLOGY"] == "1,1,1,2"

    # Shared-slice ICI path preserved untouched
    monkeypatch.setenv("MILES_TPU_SHARED_SLICE", "0,8,localhost:30108")
    assert (
        tpu.worker_distributed_env_vars(
            bundle_node_ips=["10.4.17.6"] * 4,
            rank=0,
            local_rank=0,
            world_size=4,
            local_world_size=4,
            devices_per_node=8,
        )
        == {}
    )


def test_tpu_rollout_worker_env_vars_full_and_subhost(monkeypatch):
    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)

    full = tpu.rollout_worker_env_vars(
        bundle_node_ips=["10.4.17.6"] * 8,
        tp_size=8,
        devices_per_node=8,
        subhost_offset=0,
    )
    assert full["TPU_MULTIHOST_BACKEND"] == "mp"
    assert full["VLLM_SERVER_DEV_MODE"] == "1"
    assert full["TORCH_TPU_TOPOLOGY"] == "2,2,1,2"
    assert full["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == ",".join(f"10.4.17.6:{8471 + r}" for r in range(8))
    assert "MILES_TPU_POP_WEBHOOK_ENV" not in full

    sub = tpu.rollout_worker_env_vars(
        bundle_node_ips=["10.4.17.6"] * 4,
        tp_size=4,
        devices_per_node=8,
        subhost_offset=4,
    )
    assert sub["TPU_MULTIHOST_BACKEND"] == "mp"
    assert sub["VLLM_SERVER_DEV_MODE"] == "1"
    assert sub["TORCH_TPU_TOPOLOGY"] == "1,2,1,2"
    assert sub["ALLOW_MULTIPLE_LIBTPU_LOAD"] == "1"
    assert sub["TPU_LOCAL_RANK_OFFSET"] == "4"
    assert sub["DEBUG_TPU_LOCAL_RANK_OFFSET"] == "4"
    assert sub["MEGASCALE_COORDINATOR_ADDRESS"] == "127.0.0.1:21080"
    assert sub["MILES_TPU_SUBHOST_PORT_BASE"] == "21080"
    assert sub["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == ",".join(f"localhost:{21380 + r}" for r in range(4))
    assert sub["MILES_TPU_POP_WEBHOOK_ENV"] == "1"
    assert "tp4_off4" in sub["VLLM_CACHE_ROOT"]
    assert "tp4_off4" in sub["TORCHINDUCTOR_CACHE_DIR"]
    assert "tp4_off4" in sub["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"]


def test_tpu_compile_caches_persist_under_the_run_cache_root_keyed_by_slot(monkeypatch):
    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)
    monkeypatch.delenv("MILES_TPU_SHARED_SLICE", raising=False)
    cache_keys = ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")

    def rollout(tp_size: int, offset: int) -> dict[str, str]:
        return tpu.rollout_worker_env_vars(
            bundle_node_ips=["10.4.17.6"] * tp_size, tp_size=tp_size, devices_per_node=8, subhost_offset=offset
        )

    def trainer(world_size: int, offset: int) -> dict[str, str]:
        return tpu.worker_distributed_env_vars(
            bundle_node_ips=["10.4.17.6"] * world_size,
            rank=0,
            local_rank=0,
            world_size=world_size,
            local_world_size=world_size,
            devices_per_node=8,
            subhost_offset=offset,
        )

    # without a run cache root: sub-host slots keep today's node-local dirs, a full host sets none
    monkeypatch.delenv("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", raising=False)
    assert rollout(4, 4)["VLLM_CACHE_ROOT"] == "/tmp/vllm_cache_tp4_off4"
    assert trainer(4, 0)["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] == "/tmp/tpu_tier3_w4_off0"
    assert not set(cache_keys) & set(rollout(8, 0)) and not set(cache_keys) & set(trainer(8, 0))

    # with --compile-cache-dir: every slot gets its own dirs under the persistent root
    monkeypatch.setenv("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", "/data/cache")
    assert {k: rollout(4, 4)[k] for k in cache_keys} == {
        "VLLM_CACHE_ROOT": "/data/cache/rollout_tp4_off4/vllm_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/data/cache/rollout_tp4_off4/inductor_cache",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/data/cache/rollout_tp4_off4/tpu_tier3",
    }
    assert rollout(8, 0)["VLLM_CACHE_ROOT"] == "/data/cache/rollout_tp8_off0/vllm_cache"
    assert trainer(4, 0)["TORCHINDUCTOR_CACHE_DIR"] == "/data/cache/trainer_w4_off0/inductor_cache"
    assert trainer(8, 0)["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] == "/data/cache/trainer_w8_off0/tpu_tier3"
    assert "VLLM_CACHE_ROOT" not in trainer(4, 0)


def test_tpu_topology_follows_the_slot_not_the_pod_wide_env(monkeypatch):
    """A manager placed on a KubeRay TPU worker inherits the host's 8-core TORCH_TPU_TOPOLOGY; a 4-core slot ignores it."""
    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.delenv("MILES_TPU_SHARED_SLICE", raising=False)
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2,2,1,2")
    rollout = tpu.rollout_worker_env_vars(
        bundle_node_ips=["10.4.3.6"] * 4, tp_size=4, devices_per_node=8, subhost_offset=4
    )
    trainer = tpu.worker_distributed_env_vars(
        bundle_node_ips=["10.4.3.6"] * 4,
        rank=0,
        local_rank=0,
        world_size=4,
        local_world_size=4,
        devices_per_node=8,
        subhost_offset=0,
    )
    assert rollout["TORCH_TPU_TOPOLOGY"] == trainer["TORCH_TPU_TOPOLOGY"] == "1,2,1,2"
    with pytest.raises(ValueError, match="no TPU v7x topology for 3 cores"):
        tpu.rollout_worker_env_vars(bundle_node_ips=["10.4.3.6"] * 3, tp_size=3, devices_per_node=8)


def test_weight_transfer_is_checked_against_the_platform(monkeypatch):
    cuda = platforms.Platform()
    cuda.check_weight_transfer(mode="broadcast", colocate=True)
    with pytest.raises(ValueError, match="tpu-sync syncs to TPU engines; not on cuda"):
        cuda.check_weight_transfer(mode="tpu-sync", colocate=False)

    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    tpu.check_weight_transfer(mode="tpu-sync", colocate=False)
    tpu.check_weight_transfer(mode="ici", colocate=False)
    for mode, colocate in (("broadcast", False), ("tpu-sync", True)):
        with pytest.raises(ValueError, match="update-weight-transfer-mode ici or tpu-sync"):
            tpu.check_weight_transfer(mode=mode, colocate=colocate)


def test_the_torch_memory_recorder_is_refused_without_cuda_allocator_history(monkeypatch):
    from miles.utils import profile_utils

    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.setattr(platforms, "current_platform", tpu)
    with pytest.raises(ValueError, match="--memory-recorder memray"):
        profile_utils._BaseMemoryProfiler.create(SimpleNamespace(memory_recorder="torch"))


def test_tpu_env_layout_is_importable_without_torch_tpu(monkeypatch):
    from miles.utils import tpu_env

    monkeypatch.setenv("TPU_WORKER_ID", "3")
    env = {"MEGASCALE_SLICE_ID": "1", "KEEP": "x"}
    tpu_env.pop_gke_webhook_env_vars(env)
    assert env == {"KEEP": "x"}
    tpu_env.pop_gke_webhook_env_vars()
    assert "TPU_WORKER_ID" not in os.environ
    assert tpu_env.subhost_port_base(0) == 21000
    assert tpu_env.subhost_port_base(4) == 21080


def test_sglang_constants_and_import_guards_work_without_sglang(monkeypatch):
    assert sglang_constants.GPU_MEMORY_TYPE_WEIGHTS == "weights"
    assert sglang_constants.GPU_MEMORY_TYPE_KV_CACHE == "kv_cache"
    assert sglang_constants.GPU_MEMORY_TYPE_CUDA_GRAPH == "cuda_graph"

    # Simulate missing sglang and sglang_router modules
    monkeypatch.setattr(sglang_arguments, "ServerArgs", None)
    monkeypatch.setattr(router, "RouterArgs", None)
    monkeypatch.setattr(output_store_config, "ServerArgs", None)
    monkeypatch.setattr(sglang_engine, "ServerArgs", None)
    monkeypatch.setattr(server_args_utils, "ServerArgs", None)
    monkeypatch.setattr(router_args_utils, "RouterArgs", None)
    monkeypatch.setattr(sglang_router_api_client, "sglang_router", None)
    monkeypatch.setattr(dumper_utils, "DumperConfig", None)
    monkeypatch.setattr(dumper_utils, "dumper", None)

    assert output_store_config._sglang_has_output_store() is False
    assert dumper_utils.get_sglang_env(SimpleNamespace(dumper_enable=False, dumper_inference=None)) == {}
    assert chat_template.extract_tool_dicts([{"name": "add", "description": "sum"}]) == [
        {
            "type": "function",
            "function": {"name": "add", "description": "sum", "parameters": None, "strict": None},
        }
    ]
    assert chat_deepseek._tool_payload({"type": "function", "function": {"name": "add"}}) == {
        "type": "function",
        "function": {"name": "add"},
    }

    with pytest.raises(ImportError, match="sglang is required"):
        sglang_engine._assert_launch_gate_served.__wrapped__()
    with pytest.raises(ImportError, match="sglang_router is required"):
        router_args_utils.router_args_to_argv({})


def test_parse_args_without_sglang_for_cuda_and_tpu_vllm(monkeypatch):
    monkeypatch.setattr(sglang_arguments, "ServerArgs", None)
    monkeypatch.setattr(router, "RouterArgs", None)

    # Register a stub vllm_single_turn.generate if the sibling author's file isn't present yet
    stub_mod = ModuleType("miles.rollout.generate_hub.vllm_single_turn")

    async def _stub_generate(args, sample, sampling_params, evaluation=False):
        return sample

    stub_mod.generate = _stub_generate
    if "miles.rollout.generate_hub.vllm_single_turn" not in sys.modules:
        try:
            importlib.import_module("miles.rollout.generate_hub.vllm_single_turn")
        except ImportError:
            monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.vllm_single_turn", stub_mod)

    # 1. CUDA defaults preserved without sglang installed
    cuda_argv = [
        "train.py",
        "--train-backend",
        "fsdp",
        "--rollout-batch-size",
        "2",
        "--num-rollout",
        "1",
        "--actor-num-gpus-per-node",
        "1",
        "--rollout-num-gpus",
        "1",
        "--micro-batch-size",
        "1",
    ]
    with patch.object(sys, "argv", cuda_argv):
        cuda_args = miles_arguments.parse_args()
    assert cuda_args.distributed_backend == "nccl"
    assert cuda_args.rollout_engine == "sglang"
    assert cuda_args.router is RouterKind.SGLANG
    assert cuda_args.sglang.common_value("dp_size") == 1
    assert cuda_args.sglang.common_value("api_key") is None
    assert cuda_args.sglang.common_value("enable_deterministic_inference") is False

    # 2. TPU + FSDP + vLLM + tpu-sync defaults and validations
    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.setattr(platforms, "current_platform", tpu)
    monkeypatch.setattr(sglang_arguments, "current_platform", tpu)

    tpu_argv = [
        "train.py",
        "--train-backend",
        "fsdp",
        "--rollout-engine",
        "vllm",
        "--update-weight-transfer-mode",
        "tpu-sync",
        "--vllm-max-model-len",
        "4096",
        "--vllm-gpu-memory-utilization",
        "0.8",
        "--vllm-max-num-seqs",
        "64",
        "--vllm-dtype",
        "bfloat16",
        "--vllm-enforce-eager",
        "--vllm-extra-args=--disable-log-requests",
        "--rollout-batch-size",
        "4",
        "--num-rollout",
        "1",
        "--actor-num-gpus-per-node",
        "4",
        "--rollout-num-gpus",
        "4",
        "--rollout-num-gpus-per-engine",
        "4",
        "--micro-batch-size",
        "1",
    ]
    with patch.object(sys, "argv", tpu_argv):
        tpu_args = miles_arguments.parse_args()
    assert tpu_args.distributed_backend == "tpu_dist"
    assert tpu_args.rollout_engine == "vllm"
    assert tpu_args.router is RouterKind.MILES
    assert tpu_args.update_weight_transfer_mode == "tpu-sync"
    assert tpu_args.custom_generate_function_path.path == "miles.rollout.generate_hub.vllm_single_turn.generate"
    assert tpu_args.vllm_max_model_len == 4096
    assert tpu_args.vllm_gpu_memory_utilization == 0.8
    assert tpu_args.vllm_max_num_seqs == 64
    assert tpu_args.vllm_dtype == "bfloat16"
    assert tpu_args.vllm_enforce_eager is True
    assert tpu_args.vllm_extra_args == "--disable-log-requests"
    assert tpu_args.sglang.common_value("kv_cache_dtype") == "bfloat16"

    # 3. TPU refuses --colocate
    with patch.object(sys, "argv", [*tpu_argv, "--colocate"]), pytest.raises(ValueError, match="--colocate"):
        miles_arguments.parse_args()

    # 4. TPU refuses unsupported transfer mode (broadcast)
    bad_mode_argv = [arg if arg != "tpu-sync" else "broadcast" for arg in tpu_argv]
    with patch.object(sys, "argv", bad_mode_argv), pytest.raises(ValueError, match="update-weight-transfer-mode"):
        miles_arguments.parse_args()

    # 5. vLLM refuses what its rollout path would silently ignore, and tpu-sync what it cannot sync
    for extra, match in (
        (["--rollout-top-logprobs-num", "5"], "--rollout-top-logprobs-num"),
        (["--check-weight-update-equal"], "--check-weight-update-equal"),
        (["--use-rollout-routing-replay"], "--use-rollout-routing-replay"),
        (
            ["--custom-generate-function-path", "miles.rollout.generate_hub.multi_turn.generate"],
            "SGLang generate function",
        ),
        (["--sglang-pp-size", "2"], "--sglang-pp-size 1"),
        (["--sglang-speculative-algorithm", "EAGLE"], "speculative draft weights"),
    ):
        with patch.object(sys, "argv", [*tpu_argv, *extra]), pytest.raises(ValueError, match=match):
            miles_arguments.parse_args()

    # 6. tpu-sync is refused on CUDA, where no engine can receive it
    monkeypatch.setattr(platforms, "current_platform", platforms.Platform())
    monkeypatch.setattr(sglang_arguments, "current_platform", platforms.Platform())
    with patch.object(sys, "argv", tpu_argv), pytest.raises(ValueError, match="tpu-sync syncs to TPU engines"):
        miles_arguments.parse_args()


def test_backend_fields_distributed_backend_follows_platform(monkeypatch):
    parser_cuda = argparse.ArgumentParser()
    backend_fields.TrainerBackendTraitConfig.add_arguments(parser_cuda)
    assert parser_cuda.parse_args([]).distributed_backend == "nccl"

    tpu = _load_tpu_module(monkeypatch).TpuPlatform()
    monkeypatch.setattr(platforms, "current_platform", tpu)
    parser_tpu = argparse.ArgumentParser()
    backend_fields.TrainerBackendTraitConfig.add_arguments(parser_tpu)
    assert parser_tpu.parse_args([]).distributed_backend == "tpu_dist"
