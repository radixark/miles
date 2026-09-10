"""Slot capacity: how many resident LoRA tenants fit.

Measurement is the authority, sglang-style: after the base model loads, one
probe slot runs a max-size forward/backward and an optimizer step through the
real executor path, and the bytes that slot owns, the activation peak, and the
memory still free give every rank's head-room. The closed-form prediction only
cross-checks the measurement and explains the log line.

The trainer sizes its slot pool at construction (the Bridge adapter modules and
the per-slot LayerWise optimizers), so ``auto`` probes a one-slot trainer and
rebuilds it at the resolved count; serve_tinker owns that sequence.
"""

import logging
from argparse import Namespace
from dataclasses import dataclass

import torch

from miles.backends.megatron_utils.lora.optimizer import _slot_children, adapter_slot_parameters

logger = logging.getLogger(__name__)

AUTO_SLOT_CAPACITY = -1  # --multi-lora-n-adapters auto
PROBE_SLOTS = 1  # pool size of the probe trainer
PROBE_SLOT = 0
_PROBE_BATCH_ID = -1

_PROBE_ADAM_PARAMS = {
    # lr 0: the step only materializes the Adam moments, the weights stay put
    "learning_rate": 0.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


@dataclass(frozen=True)
class RankProbe:
    free: int  # bytes free with the probe slot resident, after a steady-state step
    slot_bytes: int  # CUDA bytes the resident slot owns: weights, grad buffers, fp32 masters, Adam moments
    act_peak: int  # transient peak of one max-size forward/backward; shared across slots (single issue)
    adapter_local_params: int  # this rank's shard of one max-rank adapter

    def capacity(self, margin_bytes: int) -> int:
        if self.slot_bytes <= 0:
            raise ValueError(f"probe measured a non-positive slot residency: {self.slot_bytes}")
        # the pool is rebuilt from scratch, so the probe slot's own bytes are head-room too
        return max((self.free + self.slot_bytes - self.act_peak - margin_bytes) // self.slot_bytes, 0)


def bytes_per_train_param(args: Namespace) -> int:
    """Per-slot resident bytes per LoRA param, from the precision flags."""
    weight = 2 if (args.bf16 or args.fp16) else 4
    grad = 4 if args.accumulate_allreduce_grads_in_fp32 else weight
    master = 4 if weight < 4 else 0  # mixed precision keeps an fp32 master; pure fp32 does not
    moments = 8  # per-slot torch Adam: fp32 exp_avg + exp_avg_sq (not precision-aware)
    return weight + grad + master + moments


def memory_snapshot(model, optimizer, phase: str) -> dict:
    """Actor-side half of the probe; the orchestration lives in probe_slot_capacity.
    ``reset`` arms the peak tracker before the measured step, ``measure`` reads it after."""
    torch.cuda.synchronize()
    if phase == "reset":
        torch.cuda.reset_peak_memory_stats()
        return {}
    assert phase == "measure", f"unknown memory_snapshot phase {phase!r}"
    act_peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    torch.cuda.empty_cache()  # cached-but-unused blocks are head-room, not residency
    free, _ = torch.cuda.mem_get_info()
    return {
        "free": free,
        "slot_bytes": resident_slot_bytes(model, optimizer, PROBE_SLOT),
        "act_peak": max(act_peak, 0),
        "adapter_local_params": sum(param.numel() for param in adapter_slot_parameters(model, PROBE_SLOT)),
    }


def resident_slot_bytes(model, optimizer, slot: int) -> int:
    """CUDA bytes one resident slot owns: the adapter weights, their grad buffers, the
    fp32 masters and the Adam moments. Views into one allocation count it once."""
    storages: dict[tuple[int, int], int] = {}

    def record(tensor) -> None:
        if tensor is not None and getattr(tensor, "is_cuda", False):
            storage = tensor.untyped_storage()
            storages[(tensor.device.index, storage.data_ptr())] = storage.nbytes()

    for param in adapter_slot_parameters(model, slot):
        for tensor in (param, param.grad, getattr(param, "main_grad", None), getattr(param, "main_param", None)):
            record(tensor)
    for child in _slot_children(optimizer, slot):
        for param in child.get_parameters():  # the fp32 masters the mixed-precision wrapper steps
            record(param)
        for state in child.optimizer.state.values():
            for value in state.values():
                record(value)
    return sum(storages.values())


async def probe_slot_capacity(args: Namespace, backend, trainer) -> list[RankProbe]:
    """Load one max-rank probe slot and run two max-size steps through the real
    executor path: the first materializes the Adam moments and the process-wide
    workspaces every later step shares, the second is measured. Every rank reports
    its own head-room; the trainer side is self-contained and never touches an engine."""
    await backend.load_slot(PROBE_SLOT, args.lora_rank, float(args.lora_alpha or 2 * args.lora_rank))
    row = _probe_row(args.max_tokens_per_gpu)
    await _probe_step(backend, row)
    await trainer.multi_lora_memory_probe("reset")
    await _probe_step(backend, row)
    snapshots = await trainer.multi_lora_memory_probe("measure")
    await backend.unload_slot(PROBE_SLOT)

    probes = [RankProbe(**snapshot) for snapshot in snapshots]
    predicted = probes[0].adapter_local_params * bytes_per_train_param(args)
    if abs(probes[0].slot_bytes - predicted) > 0.2 * max(predicted, 1):
        logger.warning(
            f"measured slot bytes {probes[0].slot_bytes} diverge from predicted {predicted}: "
            "unaccounted per-slot memory; trust the measurement"
        )
    return probes


async def _probe_step(backend, row: dict) -> None:
    await backend.forward_backward(_PROBE_BATCH_ID, [(PROBE_SLOT, row)], "cross_entropy", {})
    outcomes = await backend.optim_step({PROBE_SLOT: _PROBE_ADAM_PARAMS})
    assert "grad_norm" in outcomes.get(PROBE_SLOT, {}), f"probe optimizer step did not settle: {outcomes}"


def _probe_row(tokens: int) -> dict:
    """One datum that fills --max-tokens-per-gpu: a one-token prompt and a max-length target."""
    assert tokens >= 2, f"--max-tokens-per-gpu {tokens} leaves no room for a prompt and a target"
    return {"tokens": [1] * tokens, "target_len": tokens - 1, "weights": [1.0] * (tokens - 1)}


def resolve_slot_capacity(args: Namespace, probes: list[RankProbe]) -> int:
    """The worst rank rules; the log carries the numbers behind the count."""
    margin = getattr(args, "train_memory_margin_bytes", 0) or 0
    worst = min(probes, key=lambda probe: probe.capacity(margin))
    n = worst.capacity(margin)
    assert n >= 1, (
        f"no room for one rank-{args.lora_rank} adapter slot: a slot needs {worst.slot_bytes >> 20} MiB, "
        "free after the model, a max-size batch and the margin is "
        f"{(worst.free + worst.slot_bytes - worst.act_peak - margin) >> 20} MiB. "
        "Lower --lora-rank or --max-tokens-per-gpu."
    )
    logger.info(
        f"multi-LoRA capacity: {n} slots, bound by the worst trainer rank "
        f"(slot={worst.slot_bytes >> 20}MiB, act_peak={worst.act_peak >> 20}MiB, "
        f"free={worst.free >> 20}MiB, margin={margin >> 20}MiB)"
    )
    return n
