"""Slot capacity: how many resident LoRA tenants fit.

Measurement is the authority, sglang-style: load one probe slot, run one
max-size forward/backward and an optimizer step through the real executor
path, and read the deltas. The closed-form prediction only cross-checks the
measurement and explains the log line.
"""

import logging
from argparse import Namespace
from dataclasses import dataclass

logger = logging.getLogger(__name__)

AUTO_SLOT_CAPACITY = -1  # --multi-lora-n-adapters auto

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
    free_before: int  # bytes free after the base model, before the probe slot
    free_after: int  # bytes free with the probe slot resident (weights+grad+master+moments)
    act_peak: int  # transient peak of one max-size fb; shared across slots (single issue)
    adapter_local_params: int  # this rank's shard of one max-rank adapter
    adapter_full_params: int  # the unsharded adapter, for engine-side copies

    @property
    def slot_bytes(self) -> int:
        return self.free_before - self.free_after

    def capacity(self, margin_bytes: int) -> int:
        return max(int((self.free_before - self.act_peak - margin_bytes) // self.slot_bytes), 0)


def bytes_per_train_param(args: Namespace) -> int:
    """Per-slot resident bytes per LoRA param, from the precision flags."""
    weight = 2 if (args.bf16 or args.fp16) else 4
    grad = 4 if args.accumulate_allreduce_grads_in_fp32 else weight
    master = 4 if weight < 4 else 0  # mixed precision keeps an fp32 master; pure fp32 does not
    moments = 8  # per-slot torch Adam: fp32 exp_avg + exp_avg_sq (not precision-aware)
    return weight + grad + master + moments


def memory_snapshot(args: Namespace, model, phase: str) -> dict:
    """Actor-side measurement half of the probe; the orchestration lives in
    probe_slot_capacity. ``before`` resets the peak tracker, ``after`` reads it."""
    import torch

    torch.cuda.synchronize()
    if phase == "before":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, _ = torch.cuda.mem_get_info()
        return {"free": free}
    assert phase == "after", f"unknown memory_snapshot phase {phase!r}"
    free, _ = torch.cuda.mem_get_info()
    act_peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    local, full = _adapter_param_counts(args, model)
    return {"free": free, "act_peak": act_peak, "adapter_local_params": local, "adapter_full_params": full}


def _adapter_param_counts(args: Namespace, model) -> tuple[int, int]:
    """(rank-local, unsharded) params of the resident adapter. Full counts come
    from megatron's own sharding attributes, so there is no shape table to drift."""
    from miles.utils.lora import is_lora_weight_name

    tp = args.tensor_model_parallel_size
    ep = getattr(args, "expert_model_parallel_size", 1) or 1
    local = full = 0
    for chunk in model:
        for name, param in chunk.named_parameters():
            if not is_lora_weight_name(name):
                continue
            numel = param.numel()
            local += numel
            multiplier = tp if getattr(param, "tensor_model_parallel", False) else 1
            if ".experts." in name:
                multiplier *= ep  # expert adapters shard by EP; etp == 1 is validated at launch
            full += numel * multiplier
    return local, full


async def probe_slot_capacity(args: Namespace, backend, trainer) -> list[RankProbe]:
    """Load one max-rank probe slot, run one max-size fb and an optimizer step
    through the real executor path, and measure every rank's head-room. Runs
    before the rollout engines launch; the trainer side is self-contained."""
    before = await trainer.multi_lora_memory_probe("before")
    await backend.load_slot(0, args.lora_rank, float(args.lora_alpha or 2 * args.lora_rank))
    row = _probe_row(args.max_tokens_per_gpu)
    await backend.forward_backward(-1, [(0, row)], "cross_entropy", {})
    await backend.optim_step({0: _PROBE_ADAM_PARAMS})
    after = await trainer.multi_lora_memory_probe("after")
    await backend.unload_slot(0)

    probes = [
        RankProbe(
            free_before=b["free"],
            free_after=a["free"],
            act_peak=a["act_peak"],
            adapter_local_params=a["adapter_local_params"],
            adapter_full_params=a["adapter_full_params"],
        )
        for b, a in zip(before, after, strict=True)
    ]
    predicted = probes[0].adapter_local_params * bytes_per_train_param(args)
    if abs(probes[0].slot_bytes - predicted) > 0.2 * max(predicted, 1):
        logger.warning(
            f"measured slot bytes {probes[0].slot_bytes} diverge from predicted {predicted}: "
            "unaccounted per-slot memory; trust the measurement"
        )
    return probes


def _probe_row(tokens: int) -> dict:
    return {"tokens": [1] * (tokens + 1), "target_len": tokens, "weights": [1.0] * tokens}


def resolve_slot_capacity(args: Namespace, probes: list[RankProbe], keep_k: int) -> int:
    """min over the binding constraints; the log names which one bound."""
    margin = getattr(args, "train_memory_margin_bytes", 0) or 0
    n_gpu = min(probe.capacity(margin) for probe in probes)  # the worst rank rules

    host_budget = getattr(args, "engine_host_lora_budget_bytes", None)
    per_version_bytes = probes[0].adapter_full_params * 2  # engine CPU copies are bf16
    n_host = host_budget // (keep_k * per_version_bytes) if host_budget else None

    n = n_gpu if n_host is None else min(n_gpu, n_host)
    worst = min(probes, key=lambda probe: probe.capacity(margin))
    assert n >= 1, (
        f"no room for one rank-{args.lora_rank} adapter slot: a slot needs "
        f"{worst.slot_bytes >> 20} MiB, free after the model and a max-size batch is "
        f"{(worst.free_before - worst.act_peak - margin) >> 20} MiB. "
        "Lower --lora-rank or --max-tokens-per-gpu."
    )
    binding = "trainer GPU memory" if n_host is None or n_gpu <= n_host else "engine host RAM (keep-K copies)"
    logger.info(
        f"multi-LoRA capacity: {n} slots, bound by {binding} "
        f"(gpu={n_gpu}, host={n_host if n_host is not None else 'unchecked'}, "
        f"slot={worst.slot_bytes >> 20}MiB, act_peak={worst.act_peak >> 20}MiB, "
        f"adapter={per_version_bytes >> 20}MiB/version)"
    )
    return n
