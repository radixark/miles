from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


@pytest.fixture
def nvme_setup_chain(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    original_empty = torch.empty

    def unpinned_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", unpinned_empty)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.config = SimpleNamespace(
        defer_main_param_initialization=True,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        log_num_zeros_in_grad=False,
    )
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.model_float16_groups = []
    optimizer.shard_fp32_from_float16_groups = []
    optimizer.model_fp32_groups = []
    optimizer.shard_fp32_groups = []
    optimizer.grad_scaler = None
    optimizer.distributed_optimizer_instance_id = 0
    return SimpleNamespace(chained_optimizers=[optimizer], get_grad_norm=lambda: 0.0)


def make_nvme_setup_args(*, directory: Path, no_load_optim: bool) -> Namespace:
    return Namespace(
        offload_train_disk_dir=str(directory),
        offload_train_disk_chunk_mb=1,
        stream_optimizer_state_moment_dtype="fp32",
        no_load_optim=not no_load_optim,
        backend=Namespace(no_load_optim=no_load_optim),
    )
