import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from miles.backends.training_utils import parallel as shared_parallel


@dataclass
class _InitializationSdk:
    module: Any
    tokenizer: Mock
    microbatches: Mock
    installed_args: Mock
    experimental: Mock
    communicators: Mock
    deterministic: Mock


@pytest.fixture
def initialization_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[_InitializationSdk]:
    from megatron.training import initialize as upstream_initialize

    from miles.backends.megatron_utils import initialize, parallel

    sdk = _InitializationSdk(
        module=initialize,
        tokenizer=Mock(),
        microbatches=Mock(),
        installed_args=Mock(),
        experimental=Mock(),
        communicators=Mock(),
        deterministic=Mock(),
    )
    monkeypatch.setattr(shared_parallel, "_parallel_state", None)
    monkeypatch.setattr(initialize, "set_args", sdk.installed_args)
    monkeypatch.setattr(initialize, "_build_tokenizer", sdk.tokenizer)
    monkeypatch.setattr(initialize, "init_num_microbatches_calculator", sdk.microbatches)
    monkeypatch.setattr(initialize, "set_experimental_flag", sdk.experimental)
    monkeypatch.setattr(upstream_initialize, "_initialize_tp_communicators", sdk.communicators)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", sdk.deterministic)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", True)
    monkeypatch.setenv("MILES_FILL_UNINIT", "0")
    monkeypatch.setattr(initialize.mpu, "initialize_model_parallel", Mock())
    monkeypatch.setattr(initialize.tensor_parallel, "model_parallel_cuda_manual_seed", Mock())
    for name in (
        "data_parallel",
        "context_parallel",
        "tensor_model_parallel",
        "pipeline_model_parallel",
        "expert_model_parallel",
        "expert_tensor_parallel",
        "expert_data_parallel",
    ):
        monkeypatch.setattr(initialize.mpu, f"get_{name}_rank", lambda **_kwargs: 0)
        monkeypatch.setattr(initialize.mpu, f"get_{name}_world_size", lambda **_kwargs: 1)
        monkeypatch.setattr(initialize.mpu, f"get_{name}_group", lambda **_kwargs: None)
    monkeypatch.setattr(initialize.mpu, "get_data_parallel_group_gloo", lambda **_kwargs: None)
    monkeypatch.setattr(initialize.mpu, "get_tensor_and_data_parallel_group", lambda **_kwargs: None)
    monkeypatch.setattr(initialize.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(initialize.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(initialize.dist, "get_rank", lambda _group=None: 0)
    monkeypatch.setattr(initialize.dist, "get_world_size", lambda _group=None: 1)
    monkeypatch.setattr(parallel, "get_args", lambda: sdk.installed_args.call_args.args[0])
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield sdk
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _initialization_args(**overrides: Any) -> Any:
    from tests.fast.fixtures.args_fixtures import make_trainer_config

    defaults = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "pipeline_model_parallel_comm_backend": None,
        "context_parallel_size": 1,
        "hierarchical_context_parallel_sizes": None,
        "expert_model_parallel_size": 1,
        "num_distributed_optimizer_instances": 1,
        "expert_tensor_parallel_size": 1,
        "distributed_timeout_minutes": 17,
        "nccl_communicator_config_path": None,
        "use_tp_pp_dp_mapping": False,
        "use_gloo_process_groups": False,
        "enable_experimental": False,
        "tp_comm_overlap": False,
        "deterministic_mode": False,
        "rank": 0,
        "seed": 71,
        "data_parallel_random_init": False,
        "te_rng_tracker": False,
        "inference_rng_tracker": False,
        "cp_comm_type": "p2p",
        "indep_dp": False,
        "rampup_batch_size": [4, 8, 16],
        "global_batch_size": 12,
        "micro_batch_size": 3,
        "data_parallel_size": 2,
        "decrease_batch_size_if_needed": True,
        "debug_deterministic_collective": False,
        "custom_megatron_init_path": None,
    }
    return make_trainer_config(**(defaults | overrides))
