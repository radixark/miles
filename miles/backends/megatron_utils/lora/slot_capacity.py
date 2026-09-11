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
# torch._grouped_mm's CUDA kernel refuses more than 1024 groups ("Can't process more than 1024
# groups"); the expert adapters run one group per (slot, local expert), so the pool cannot hold
# more than 1024 / local_experts slots however much memory is free. The probe's single slot
# never trips it, so the bound is applied by arithmetic.
GROUPED_MM_MAX_GROUPS = 1024

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
    adapter_expert_params: int = 0  # the part of that shard living on MoE experts (sharded EP-wise, not DP-wise)
    expert_groups_per_slot: int = (
        0  # grouped-GEMM groups one slot adds: this rank's local experts (0: no expert adapters)
    )

    def capacity(self, margin_bytes: int) -> int:
        if self.slot_bytes <= 0:
            raise ValueError(f"probe measured a non-positive slot residency: {self.slot_bytes}")
        # the pool is rebuilt from scratch, so the probe slot's own bytes are head-room too
        return max((self.free + self.slot_bytes - self.act_peak - margin_bytes) // self.slot_bytes, 0)


def _weight_and_grad_bytes(args: Namespace) -> int:
    """Per LoRA param on every rank: the replicated weight and the all-reduced gradient."""
    weight = 2 if (args.bf16 or args.fp16) else 4
    grad = 4 if args.accumulate_allreduce_grads_in_fp32 else weight
    return weight + grad


def _optimizer_state_bytes(args: Namespace) -> int:
    """Per LoRA param the owning rank keeps: fp32 master under mixed precision, fp32 Adam moments."""
    weight = 2 if (args.bf16 or args.fp16) else 4
    master = 4 if weight < 4 else 0  # mixed precision keeps an fp32 master; pure fp32 does not
    return master + 8


def bytes_per_train_param(args: Namespace, dp_size: int = 1) -> float:
    """Per-slot resident bytes per dense LoRA param on one rank, from the precision flags.

    The weights are replicated and DDP all-reduces full gradients, but the LayerWise
    optimizer scatters whole params across data-parallel ranks, so each rank keeps the
    fp32 master and the Adam moments for only its share of the slot."""
    return _weight_and_grad_bytes(args) + _optimizer_state_bytes(args) / dp_size


def expert_data_parallel_size(args: Namespace, dp_size: int) -> int:
    """Ranks that replicate one expert's params: experts are already split EP (and ETP) ways,
    so their optimizer state is scattered over only world / (EP * ETP * PP) ranks."""
    tp = getattr(args, "tensor_model_parallel_size", 1) or 1
    cp = getattr(args, "context_parallel_size", 1) or 1
    ep = getattr(args, "expert_model_parallel_size", 1) or 1
    etp = getattr(args, "expert_tensor_parallel_size", None) or tp
    return max(1, dp_size * tp * cp // (ep * etp))


def predicted_slot_bytes(args: Namespace, probe: RankProbe, dp_size: int) -> float:
    """The closed-form counterpart of the measurement: dense adapters share optimizer state
    across the data-parallel ranks, expert adapters across the expert-data-parallel ranks."""
    shared, state = _weight_and_grad_bytes(args), _optimizer_state_bytes(args)
    dense = probe.adapter_local_params - probe.adapter_expert_params
    return dense * (shared + state / dp_size) + probe.adapter_expert_params * (
        shared + state / expert_data_parallel_size(args, dp_size)
    )


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
    local_params, expert_params = adapter_param_counts(model, PROBE_SLOT)
    return {
        "free": free,
        "slot_bytes": resident_slot_bytes(model, optimizer, PROBE_SLOT),
        "act_peak": max(act_peak, 0),
        "adapter_local_params": local_params,
        "adapter_expert_params": expert_params,
        "expert_groups_per_slot": expert_groups_per_slot(model),
    }


def expert_groups_per_slot(model) -> int:
    """Grouped-GEMM groups one slot adds on this rank: the local experts of the widest expert adapter."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRAGroupedExpertLinear

    chunks = model if isinstance(model, (list, tuple)) else [model]
    return max(
        (
            module.num_local_experts
            for chunk in chunks
            for module in chunk.modules()
            if isinstance(module, MultiLoRAGroupedExpertLinear)
        ),
        default=0,
    )


def adapter_param_counts(model, slot: int) -> tuple[int, int]:
    """(all, on MoE experts) params of one slot on this rank, by the adapter modules' names."""
    slot_ids = {id(param) for param in adapter_slot_parameters(model, slot)}
    chunks = model if isinstance(model, (list, tuple)) else [model]
    local = expert = 0
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            if id(param) not in slot_ids:
                continue
            local += param.numel()
            if ".experts." in name:
                expert += param.numel()
    return local, expert


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


async def probe_slot_capacity(args: Namespace, backend, trainer, dp_size: int = 1) -> list[RankProbe]:
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
    predicted = predicted_slot_bytes(args, probes[0], dp_size)
    if abs(probes[0].slot_bytes - predicted) > 0.2 * max(predicted, 1):
        logger.warning(
            f"measured slot bytes {probes[0].slot_bytes} diverge from predicted {predicted:.0f}: "
            "unaccounted per-slot memory; trust the measurement"
        )
    else:
        logger.info(f"measured slot bytes {probes[0].slot_bytes} agree with predicted {predicted:.0f}")
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
    """min over the binding constraints, the worst rank ruling each; the log names which one bound."""
    margin = getattr(args, "train_memory_margin_bytes", 0) or 0
    worst = min(probes, key=lambda probe: probe.capacity(margin))
    n_memory = worst.capacity(margin)
    assert n_memory >= 1, (
        f"no room for one rank-{args.lora_rank} adapter slot: a slot needs {worst.slot_bytes >> 20} MiB, "
        "free after the model, a max-size batch and the margin is "
        f"{(worst.free + worst.slot_bytes - worst.act_peak - margin) >> 20} MiB. "
        "Lower --lora-rank or --max-tokens-per-gpu."
    )
    groups = max(probe.expert_groups_per_slot for probe in probes)
    n_groups = GROUPED_MM_MAX_GROUPS // groups if groups else None
    n = n_memory if n_groups is None else min(n_memory, n_groups)
    binding = (
        "the worst trainer rank's memory"
        if n_groups is None or n_memory <= n_groups
        else f"torch._grouped_mm's {GROUPED_MM_MAX_GROUPS}-group limit ({groups} local experts per slot)"
    )
    logger.info(
        f"multi-LoRA capacity: {n} slots, bound by {binding} "
        f"(memory={n_memory}, groups={n_groups if n_groups is not None else 'n/a'}, "
        f"slot={worst.slot_bytes >> 20}MiB, act_peak={worst.act_peak >> 20}MiB, "
        f"free={worst.free >> 20}MiB, margin={margin >> 20}MiB)"
    )
    return n
