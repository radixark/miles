"""SAIL (Zhenwu-M890 / Zhenwu-810E) through its CUDA-compatible runtime.

SAIL exposes a CUDA-compatible API through its torch backend: it runs on
torch.cuda with PCCL registered as "nccl", devices come from CUDA_VISIBLE_DEVICES
and Ray reports GPU. Only the identity moves, so every CUDA capability stays on:
the NVML NUMA affinity bind, the expandable-segments allocator setting
(PYTORCH_CUDA_ALLOC_CONF), and the per-bucket barrier the colocated IPC weight
push already takes on each chunk.
"""

from miles.platforms import Platform


class SailPlatform(Platform):
    # torch.cuda is the runtime; "sail" is not a torch namespace, so the class itself carries the SAIL identity
    device_type = "cuda"
