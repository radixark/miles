"""tpu_shared_slice.join() across hosts: a trainer rank keeps the device Ray gave it, and a rank the placement put on
the wrong device fails at the join instead of hanging in the slice builder."""

import os
import sys
import types
from unittest.mock import patch

import pytest
import torch.distributed as dist

from miles.platforms import tpu_shared_slice


class _FakeTCPStore:
    """A real c10d store (PrefixStore wraps it) in place of the slice's TCPStore, every other rank's device published."""

    def __new__(cls, host, port, world, is_master, timeout):
        store = dist.HashStore()
        for rank in range(world):
            store.set(f"device:{rank}", str(500 + rank))
        return store


def _fake_torch_tpu():
    tpu_distributed = types.SimpleNamespace(
        global_device_id=lambda: 42,
        create_process_group=lambda store, rank, size, timeout: types.SimpleNamespace(rank=rank, size=size),
    )
    distributed = types.ModuleType("torch_tpu._internal.distributed")
    distributed.tpu_distributed = tpu_distributed
    return {
        "torch_tpu": types.ModuleType("torch_tpu"),
        "torch_tpu._internal": types.ModuleType("torch_tpu._internal"),
        "torch_tpu._internal.distributed": distributed,
    }


def _join(*, job_rank, job_size, local_rank, size, devices_per_host=None):
    shared = tpu_shared_slice.SharedSlice(0, size, "10.0.0.7", 30108)
    with (
        patch.dict(os.environ, {"LOCAL_RANK": local_rank}),
        patch.dict(sys.modules, _fake_torch_tpu()),
        patch.object(dist, "TCPStore", _FakeTCPStore),
        patch.object(tpu_shared_slice, "_membership", None),
        patch.object(
            tpu_shared_slice, "receiver_contract", return_value=types.SimpleNamespace(BACKEND_NAME="tpu_shared_slice")
        ),
    ):
        for name in ("MASTER_ADDR", "MASTER_PORT", tpu_shared_slice.DEVICES_PER_HOST_ENV_VAR):
            os.environ.pop(name, None)
        if devices_per_host is not None:
            os.environ[tpu_shared_slice.DEVICES_PER_HOST_ENV_VAR] = devices_per_host
        tpu_shared_slice.join(shared, job_rank=job_rank, job_size=job_size)
        return dict(os.environ), tpu_shared_slice._membership


def test_a_single_host_rank_keeps_its_device_and_the_job_env():
    env, membership = _join(job_rank=3, job_size=4, local_rank="3", size=8)
    assert (env["RANK"], env["WORLD_SIZE"], env["LOCAL_RANK"], env["LOCAL_WORLD_SIZE"]) == ("3", "4", "3", "4")
    # this rank published its own device (42) under its slice rank before reading the job's four
    assert membership.slice_rank == 3 and env["TORCH_TPU_JOB_DEVICE_IDS"] == "500,501,502,42"
    assert (env["MASTER_ADDR"], env["MASTER_PORT"]) == ("10.0.0.7", "30108")


@pytest.mark.parametrize(
    ("job_rank", "size", "local_rank"), [(11, 32, "3"), (8, 32, "0"), (63, 128, "7"), (0, 128, "0")]
)
def test_a_rank_on_a_further_host_keeps_its_host_device_index(job_rank, size, local_rank):
    """Trainer rank 11 of 16 is slice rank 11: device 3 of host 1, which is the device Ray placed the actor on."""
    env, membership = _join(job_rank=job_rank, job_size=size // 2, local_rank=local_rank, size=size)
    assert (env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"]) == (str(job_rank), local_rank, str(size // 2))
    assert membership.slice_rank == job_rank


def test_the_devices_per_host_of_the_slice_builders_is_read():
    env, _ = _join(job_rank=5, job_size=8, local_rank="1", size=16, devices_per_host="4")
    assert env["LOCAL_RANK"] == "1"
    with pytest.raises(AssertionError, match="device 1 of host 1"):
        _join(job_rank=5, job_size=8, local_rank="5", size=16, devices_per_host="4")


def test_a_rank_placed_on_the_wrong_device_is_refused():
    """Slice rank 11 must bind host 1's builder for device 3; from device 5 the builder would wait forever."""
    with pytest.raises(AssertionError, match="MILES_RAY_NODE_RANK"):
        _join(job_rank=11, job_size=16, local_rank="5", size=32)
