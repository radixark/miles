import pytest
import torch.distributed as dist


@pytest.fixture
def nccl_world(tmp_path):
    """A 1-rank NCCL group: the torch backend's fused CE reaches a collective."""
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
