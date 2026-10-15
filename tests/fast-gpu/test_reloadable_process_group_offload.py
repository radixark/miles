import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cuda_ci

from miles.utils.reloadable_process_group import (
    ReloadableProcessGroup,
    destroy_process_groups,
    monkey_patch_torch_dist,
    reload_process_groups,
)

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])


def _run_singleton_offload(_rank: int, init_method: str) -> None:
    # A child process keeps the monkey patch and process-group registry isolated
    # from the rest of the test suite.
    os.environ["NCCL_NET_PLUGIN"] = "none"
    os.environ["NCCL_NET"] = "Socket"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=init_method, rank=0, world_size=1, timeout=timedelta(seconds=30))
    try:
        monkey_patch_torch_dist()
        gloo_group = dist.new_group([0], backend="gloo")
        assert not isinstance(gloo_group, ReloadableProcessGroup)

        group = dist.new_group([0], backend="nccl")
        assert isinstance(group, ReloadableProcessGroup)
        tensor = torch.tensor([3.0], device="cuda")
        for cycle in range(2):
            # Exercise the inner NCCL communicator; this test targets its
            # lifecycle, independent of PyTorch's wrapper dispatch behavior.
            dist.all_reduce(tensor, group=group.group)
            assert tensor.item() == 3.0
            destroy_process_groups()
            assert group.group is None
            if cycle == 0:
                reload_process_groups()
                assert group.group is not None
    finally:
        destroy_process_groups()
        dist.destroy_process_group()


def test_singleton_nccl_group_is_destroyed_and_reloaded(tmp_path: Path) -> None:
    assert torch.cuda.is_available()
    assert dist.is_nccl_available()
    mp.spawn(_run_singleton_offload, args=((tmp_path / "store").as_uri(),), nprocs=1, join=True)
