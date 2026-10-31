"""Share one TPU slice with a rollout engine so weights move to it over ICI.

Enabled by MILES_TPU_SHARED_SLICE="<slice rank offset>,<slice size>,<store host>:<store port>", with the engine
given the matching SGLANG_TPU_SHARED_SLICE. libtpu builds the slice from the slice-wide RANK / WORLD_SIZE, while
the trainer's own world keeps its job ranks. Needs a torch_tpu build with job-local worlds
(TORCH_TPU_JOB_DEVICE_IDS), so neither job waits on the other's programs.
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore

ENV_VAR = "MILES_TPU_SHARED_SLICE"
# the weight-update backend name the engine binds its receiver to (sglang_tpu.shared_slice.BACKEND_NAME)
ENGINE_BACKEND_NAME = "tpu_shared_slice"
TRANSFER_ALIGN_BYTES = 4096
_TIMEOUT = timedelta(minutes=30)


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


def join(shared: SharedSlice, *, job_rank: int, job_size: int) -> None:
    """Join the shared slice as one job and open the slice-wide send/recv backend.

    Runs before the first TPU op: torch_tpu fixes the job's devices (TORCH_TPU_JOB_DEVICE_IDS) at its first
    compile, and forms the trainer's world over them only.
    """
    global _membership
    from torch_tpu._internal.distributed import tpu_distributed

    slice_rank = shared.rank_offset + job_rank
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(slice_rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(shared.size)
    # libtpu only checks these exist; the trainer sets its world's real ones before init_process_group
    os.environ.setdefault("MASTER_ADDR", shared.store_host)
    os.environ.setdefault("MASTER_PORT", str(shared.store_port))
    # starts the runtime, which joins the slice
    device_id = tpu_distributed.global_device_id()
    store = dist.TCPStore(shared.store_host, shared.store_port, shared.size, slice_rank == 0, _TIMEOUT)
    store.set(f"device:{slice_rank}", str(device_id))
    os.environ["TORCH_TPU_JOB_DEVICE_IDS"] = ",".join(
        store.get(f"device:{shared.rank_offset + rank}").decode() for rank in range(job_size)
    )
    backend = tpu_distributed.create_process_group(
        PrefixStore(ENGINE_BACKEND_NAME, store), slice_rank, shared.size, _TIMEOUT
    )
    _membership = _Membership(backend=backend, store=store, slice_rank=slice_rank)
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(job_rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(job_size)


def send(packed: torch.Tensor, dst_slice_rank: int, *, names: list[str]) -> None:
    """Send one packed bucket to one engine rank, with the HF names of what this rank sends."""
    assert _membership is not None, f"{ENV_VAR} is unset or the slice was not joined"
    # the tag also keys this send's names in the store, which the engine deletes once read
    _membership.next_tag += 1
    key = bucket_names_key(_membership.slice_rank, dst_slice_rank, _membership.next_tag)
    _membership.store.set(key, json.dumps(names))
    _membership.backend.send([packed], dst_slice_rank, _membership.next_tag).wait()


def bucket_names_key(src_slice_rank: int, dst_slice_rank: int, tag: int) -> str:
    # the engine reads and deletes it (sglang_tpu.shared_slice.bucket_names_key)
    return f"bucket_names:{src_slice_rank}:{dst_slice_rank}:{tag}"


def packed_regions(numels: list[int], dtypes: list[torch.dtype]) -> list[tuple[int, int]]:
    """(byte offset, padded byte size) of each tensor in a packed bucket: whole 4 KiB regions."""
    regions, offset = [], 0
    for numel, dtype in zip(numels, dtypes, strict=True):
        size = max(1, math.ceil(numel * dtype.itemsize / TRANSFER_ALIGN_BYTES)) * TRANSFER_ALIGN_BYTES
        regions.append((offset, size))
        offset += size
    return regions


def pack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """One int16 buffer holding every tensor's bits in its 4 KiB-aligned region; the engine views them back.

    One buffer per bucket: a send of many small buffers crashes libtpu's CopyToRemoteDevice. int16, not bytes:
    viewing a bf16 tensor as a 1-byte type pads its new trailing axis of 2 to a full TPU tile, ~64x the tensor.
    """
    regions = packed_regions([t.numel() for t in tensors], [t.dtype for t in tensors])
    pieces = []
    for tensor, (_, size) in zip(tensors, regions, strict=True):
        assert tensor.dtype.itemsize % 2 == 0, f"cannot pack {tensor.dtype} into 16-bit words"
        words = tensor.detach().reshape(-1).view(torch.int16)
        pieces += [words, words.new_zeros(size // 2 - words.numel())]
    # one concatenation: writing each tensor into a slice of a preallocated buffer rewrites the whole buffer per
    # tensor on TPU
    return torch.cat(pieces)
