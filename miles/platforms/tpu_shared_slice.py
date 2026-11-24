"""Share one TPU slice with a rollout engine so weights move to it over ICI.

Enabled by MILES_TPU_SHARED_SLICE="<slice rank offset>,<slice size>,<store host>:<store port>", with the engine
given the matching SGLANG_TPU_SHARED_SLICE. libtpu builds the slice from the slice-wide RANK / WORLD_SIZE, while
the trainer's own world keeps its job ranks. Needs a torch_tpu build with job-local worlds
(TORCH_TPU_JOB_DEVICE_IDS), so neither job waits on the other's programs.

The slice may span hosts: slice ranks are host-contiguous blocks of TORCH_TPU_DEVICES_PER_HOST devices in
TORCH_TPU_SLICEBUILDER_ADDRESSES order, so the trainer's ranks must follow that host order (see
MILES_RAY_NODE_RANK in miles.ray.placement_group) and each rank drives device ``slice rank % devices per host``.
"""

import json
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore

ENV_VAR = "MILES_TPU_SHARED_SLICE"
# torch_tpu's slice-builder layout: one host's devices are consecutive slice ranks (TORCH_TPU_SLICEBUILDER_ADDRESSES
# lists host 0's devices, then host 1's, ...), so a slice rank's device index on its host is the rank modulo this
DEVICES_PER_HOST_ENV_VAR = "TORCH_TPU_DEVICES_PER_HOST"
DEFAULT_DEVICES_PER_HOST = 8
# joining waits for the other job to start
_JOIN_TIMEOUT = timedelta(minutes=30)
# a send waits until the engine posts its receive, which never comes once the engine has refused the bucket
_SEND_TIMEOUT = timedelta(minutes=10)


@dataclass(frozen=True)
class SharedSlice:
    rank_offset: int
    size: int
    store_host: str
    store_port: int


@dataclass
class _Membership:
    backend: dist.ProcessGroup
    store: dist.Store
    slice_rank: int
    next_tag: int = 0


# a process joins the slice once, before its first TPU op, and sends through it for its whole life
_membership: _Membership | None = None


def spec() -> SharedSlice | None:
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    offset, size, address = value.split(",")
    host, port = address.rsplit(":", 1)
    return SharedSlice(int(offset), int(size), host, int(port))


def receiver_contract():
    """sglang_tpu's shared-slice module: its packed layout, names key and backend name are the wire contract."""
    from sglang_tpu import shared_slice

    return shared_slice


def join(shared: SharedSlice, *, job_rank: int, job_size: int) -> None:
    """Join the shared slice as one job and open the slice-wide send/recv backend.

    Runs before the first TPU op: torch_tpu fixes the job's devices (TORCH_TPU_JOB_DEVICE_IDS) at its first
    compile, and forms the trainer's world over them only.
    """
    global _membership
    from torch_tpu._internal.distributed import tpu_distributed

    slice_rank = shared.rank_offset + job_rank
    # LOCAL_RANK is this process's device on its host (the actor's Ray device id); torch_tpu pins the device to it.
    # On one host it equals the slice rank; on a further host of the slice the rank carries that host's offset.
    devices_per_host = int(os.environ.get(DEVICES_PER_HOST_ENV_VAR) or DEFAULT_DEVICES_PER_HOST)
    local_rank = os.environ.get("LOCAL_RANK") or str(slice_rank % devices_per_host)
    assert int(local_rank) == slice_rank % devices_per_host, (
        f"trainer rank {job_rank} (slice rank {slice_rank}) sits on device {local_rank} of its host, but the slice "
        f"builders put slice rank {slice_rank} on device {slice_rank % devices_per_host} of host "
        f"{slice_rank // devices_per_host}: the trainer's ranks must follow the slice's host order (blocks of "
        f"{devices_per_host} devices per host in TORCH_TPU_SLICEBUILDER_ADDRESSES order); set MILES_RAY_NODE_RANK on "
        "each Ray node to its host's position in that order"
    )
    os.environ["RANK"] = str(slice_rank)
    os.environ["LOCAL_RANK"] = local_rank
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(shared.size)
    # libtpu only checks these exist; the trainer sets its world's real ones before init_process_group
    os.environ.setdefault("MASTER_ADDR", shared.store_host)
    os.environ.setdefault("MASTER_PORT", str(shared.store_port))
    # starts the runtime, which joins the slice
    device_id = tpu_distributed.global_device_id()
    store = dist.TCPStore(shared.store_host, shared.store_port, shared.size, slice_rank == 0, _JOIN_TIMEOUT)
    store.set(f"device:{slice_rank}", str(device_id))
    os.environ["TORCH_TPU_JOB_DEVICE_IDS"] = ",".join(
        store.get(f"device:{shared.rank_offset + rank}").decode() for rank in range(job_size)
    )
    backend = tpu_distributed.create_process_group(
        PrefixStore(receiver_contract().BACKEND_NAME, store), slice_rank, shared.size, _SEND_TIMEOUT
    )
    _membership = _Membership(backend=backend, store=store, slice_rank=slice_rank)
    os.environ["RANK"] = str(job_rank)
    os.environ["LOCAL_RANK"] = local_rank
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(job_size)


def reset_tags() -> None:
    """The engine numbers its receives from 1 again for every weight-update group it opens."""
    assert _membership is not None, f"{ENV_VAR} is unset or the slice was not joined"
    _membership.next_tag = 0


def send(packed: torch.Tensor, dst_slice_rank: int, *, names: list[str]) -> None:
    """Send one packed bucket to one engine rank, with the HF names of what this rank sends."""
    assert _membership is not None, f"{ENV_VAR} is unset or the slice was not joined"
    # the tag also keys this send's names in the store, which the engine deletes once read
    _membership.next_tag += 1
    key = receiver_contract().bucket_names_key(_membership.slice_rank, dst_slice_rank, _membership.next_tag)
    _membership.store.set(key, json.dumps(names))
    _membership.backend.send([packed], dst_slice_rank, _membership.next_tag).wait()


def pack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """One int16 buffer holding every tensor's bits in its 4 KiB-aligned region; the engine views them back.

    One buffer per bucket: a send of many small buffers crashes libtpu's CopyToRemoteDevice. int16, not bytes:
    viewing a bf16 tensor as a 1-byte type pads its new trailing axis of 2 to a full TPU tile, ~64x the tensor.
    """
    regions = receiver_contract().packed_regions([t.numel() for t in tensors], [t.dtype for t in tensors])
    pieces = []
    for tensor, (_, size) in zip(tensors, regions, strict=True):
        assert tensor.dtype.itemsize % 2 == 0, f"cannot pack {tensor.dtype} into 16-bit words"
        words = tensor.detach().reshape(-1).view(torch.int16)
        pieces += [words, words.new_zeros(size // 2 - words.numel())]
    # one concatenation: writing each tensor into a slice of a preallocated buffer rewrites the whole buffer per
    # tensor on TPU
    return torch.cat(pieces)
