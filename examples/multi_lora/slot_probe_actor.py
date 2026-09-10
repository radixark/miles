"""Measurement-only worker selected by the standalone slot_capacity.py probe.

The existing worker-class specification is the extension point. All model,
forward/backward and optimizer implementations are inherited unchanged.
"""

import torch

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters


class SlotProbeActor(MegatronTrainRayActor):
    def pressure_memory_snapshot(self, phase: str) -> dict:
        self._heartbeat.bump()
        return _memory_snapshot(self.args, self.model, phase, optimizer=self.optimizer)


def _memory_snapshot(args, model, phase: str, *, optimizer) -> dict:
    """Synchronize every rank, count owned storage, and capture allocator peaks."""
    assert phase in ("before", "measure", "warmup", "after_fb", "after"), phase
    torch.cuda.synchronize()
    if phase in ("before", "measure"):
        torch.cuda.reset_peak_memory_stats()
    act_peak = max(0, torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated())
    torch.cuda.empty_cache()  # inactive allocator cache is reusable, not slot residency
    free, total = torch.cuda.mem_get_info()
    local, full = _adapter_param_counts(args, model)
    return {
        "rank": args.rank,
        "data_parallel_size": args.data_parallel_size,
        "optimizer_local_params": sum(
            p.numel() for child in optimizer.chained_optimizers for p in child.get_parameters()
        ),
        "free": free,
        "total": total,
        "act_peak": act_peak,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
        "resident_slot_bytes": _resident_slot_bytes(model, optimizer),
        "adapter_local_params": local,
        "adapter_full_params": full,
    }


def _resident_slot_bytes(model, optimizer) -> int:
    storages = {}

    def record(tensor):
        if tensor is not None and tensor.is_cuda:
            storage = tensor.untyped_storage()
            storages[(tensor.device.index, storage.data_ptr())] = storage.nbytes()

    for param in adapter_slot_parameters(model, 0):
        for tensor in (param, param.grad, getattr(param, "main_grad", None), getattr(param, "main_param", None)):
            record(tensor)
    for child in optimizer.chained_optimizers:
        for param in child.get_parameters():
            record(param)
        inner = getattr(child, "optimizer", None)
        if inner is not None:
            for state in inner.state.values():
                for value in state.values():
                    if hasattr(value, "is_cuda"):
                        record(value)
    return sum(storages.values())


def _adapter_param_counts(args, model) -> tuple[int, int]:
    """(rank-local, unsharded) params of the resident adapter. Full counts come
    from megatron's own sharding attributes, so there is no shape table to drift."""
    tp = args.tensor_model_parallel_size
    ep = getattr(args, "expert_model_parallel_size", 1) or 1
    local = full = 0
    for chunk in model:
        for name, param in chunk.named_parameters():
            # These are Bridge training names, not exported HF .lora_A/.lora_B
            # names. Count just the probe slot, including grouped experts.
            if ".adapters.0." not in name:
                continue
            numel = param.numel()
            local += numel
            expert = ".experts." in name
            shard_size = (getattr(args, "expert_tensor_parallel_size", 1) or 1) if expert else tp
            multiplier = shard_size if getattr(param, "tensor_model_parallel", False) else 1
            if expert:
                multiplier *= ep  # expert adapters shard by EP; etp == 1 is validated at launch
            full += numel * multiplier
    return local, full
