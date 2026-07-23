import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.ray.train import actor_factory


def test_megatron_offload_uses_torch_memory_saver_preload_resolver(monkeypatch):
    expected_path = Path("/opt/torch_memory_saver_hook_mode_preload_cu13.abi3.so")
    remote_options = {}
    remote_call_count = 0
    get_binary_path = MagicMock(return_value=expected_path)

    class FakeMegatronTrainRayActor:
        pass

    torch_memory_saver_module = ModuleType("torch_memory_saver")
    torch_memory_saver_module.__path__ = []
    torch_memory_saver_utils_module = ModuleType("torch_memory_saver.utils")
    torch_memory_saver_utils_module.get_binary_path_from_package = get_binary_path
    megatron_actor_module = ModuleType("miles.backends.megatron_utils.actor")
    megatron_actor_module.MegatronTrainRayActor = FakeMegatronTrainRayActor
    monkeypatch.setitem(sys.modules, "torch_memory_saver", torch_memory_saver_module)
    monkeypatch.setitem(sys.modules, "torch_memory_saver.utils", torch_memory_saver_utils_module)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.actor", megatron_actor_module)
    monkeypatch.setattr(actor_factory, "default_fp8_block_scaling_fp32_scales", lambda: "1")

    def fake_remote(**kwargs):
        nonlocal remote_call_count
        remote_call_count += 1
        remote_options.update(kwargs)
        return lambda actor_impl: actor_impl

    monkeypatch.setattr(actor_factory.ray, "remote", fake_remote)
    args = SimpleNamespace(
        dumper_source_patcher_config_train=None,
        offload_train=True,
        train_backend="megatron",
        train_env_vars={},
        use_fault_tolerance=False,
    )

    actors = actor_factory.allocate_gpus_for_actor(
        args=args,
        gpus_per_cell=0,
        pg=(object(), [], []),
        num_gpus_per_actor=1,
        indep_dp_store_addr="",
        role="actor",
        cell_index=0,
    )

    assert actors == []
    get_binary_path.assert_called_once_with("torch_memory_saver_hook_mode_preload")
    env_vars = remote_options["runtime_env"]["env_vars"]
    assert env_vars["LD_PRELOAD"] == str(expected_path)
    assert env_vars["TMS_INIT_ENABLE"] == "1"
    assert env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"

    get_binary_path.side_effect = RuntimeError("missing preload library")
    with pytest.raises(RuntimeError, match="missing preload library"):
        actor_factory.allocate_gpus_for_actor(
            args=args,
            gpus_per_cell=0,
            pg=(object(), [], []),
            num_gpus_per_actor=1,
            indep_dp_store_addr="",
            role="actor",
            cell_index=0,
        )
    assert remote_call_count == 1
