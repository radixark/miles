"""Host lists to device tensors without a stream synchronization.

``torch.tensor(values, device="cuda")`` copies from pageable memory, which synchronizes the stream: the host stops
queueing kernels until the device drains. TOP builds many small index tensors per microbatch and layer (replay
rounds, fork keys, decode split calls), so it copies them from pinned memory, non-blocking, instead."""

import torch


def device_tensor(values, *, device, dtype=torch.int64):
    host = torch.tensor(values, dtype=dtype)
    return host.pin_memory().to(device, non_blocking=True) if torch.device(device).type == "cuda" else host
