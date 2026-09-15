"""Slot capacity: how many resident LoRA tenants fit.

Measurement is the authority, sglang-style: load one probe slot, run one
max-size forward/backward and an optimizer step through the real executor
path, and read the deltas. The closed-form prediction only cross-checks the
measurement and explains the log line.
"""

import logging
from argparse import Namespace
from dataclasses import dataclass

from miles.backends.megatron_utils.lora.max_capacity_for_once_fb import (
    expert_groups_per_slot,
    grouped_mm_max_groups,
    max_capacity_for_once_fb,
)

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
_DTYPE_BYTES = {"float32": 4, "fp32": 4, "float16": 2, "half": 2, "fp16": 2, "bfloat16": 2, "bf16": 2}
_QUANT_BYTES = (("fp4", 0.5), ("int4", 0.5), ("awq", 0.5), ("gptq", 0.5), ("fp8", 1), ("int8", 1))


@dataclass(frozen=True)
class RankProbe:
    free_before: int  # bytes free after the base model, before the probe slot
    free_after: int  # bytes free with the probe slot resident (weights+grad+master+moments)
    act_peak: int  # transient peak of one max-size fb; shared across slots (single issue)
    adapter_local_params: int  # this rank's shard of one max-rank adapter
    adapter_full_params: int  # the unsharded adapter, for engine-side copies
    expert_groups_per_slot: int = 0  # grouped-GEMM groups one slot adds: its local experts
    grouped_mm_max_groups: int | None = None  # groups torch._grouped_mm accepts, probed; None without the op
    gpu_total: int = 0  # bytes of one GPU, the engine-memory bound's budget base
    base_params: int = 0  # the unsharded base model, the engines' weights
    slot_scale: float = 1.0

    @property
    def slot_bytes(self) -> int:
        return int((self.free_before - self.free_after) * self.slot_scale)

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
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    act_peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    local, full = _adapter_param_counts(args, model)
    return {
        "free": free,
        "act_peak": act_peak,
        "adapter_local_params": local,
        "adapter_full_params": full,
        "gpu_total": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        "expert_groups_per_slot": expert_groups_per_slot(model),
        "grouped_mm_max_groups": grouped_mm_max_groups(),
        "base_params": _rollout_base_param_count(args, model),
    }


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


def _rollout_base_param_count(args: Namespace, model) -> int:
    from miles.utils.lora import is_lora_weight_name

    tp = args.tensor_model_parallel_size
    ep = getattr(args, "expert_model_parallel_size", 1) or 1
    full = 0
    for chunk in model:
        for name, param in chunk.named_parameters():
            if is_lora_weight_name(name):
                continue
            multiplier = tp if getattr(param, "tensor_model_parallel", False) else 1
            if ".experts." in name:
                multiplier *= ep
            full += param.numel() * multiplier
    return full


def _rollout_dtype_bytes(args: Namespace) -> float:
    return _DTYPE_BYTES.get((getattr(args, "sglang_dtype", None) or "auto").lower(), 2)


def _rollout_weight_bytes(args: Namespace) -> float:
    quantization = (getattr(args, "sglang_quantization", None) or "").lower()
    return next((size for key, size in _QUANT_BYTES if key in quantization), _rollout_dtype_bytes(args))


def _rollout_kv_bytes(args: Namespace) -> float:
    return (
        1 if "fp8" in (getattr(args, "sglang_kv_cache_dtype", None) or "auto").lower() else _rollout_dtype_bytes(args)
    )


def rollout_slot_capacity(args: Namespace, probe: RankProbe) -> tuple[int, str] | None:
    seqs = getattr(args, "multi_lora_rollout_seqs_per_slot", 8)
    if not seqs or not probe.gpu_total or not probe.base_params:
        return None
    engine_tp = args.rollout_num_gpus_per_engine
    engines = max(1, args.rollout_num_gpus // engine_tp)
    fraction = getattr(args, "sglang_mem_fraction_static", None) or 0.88
    tokens = (
        getattr(args, "multi_lora_rollout_tokens_per_seq", None)
        or getattr(args, "rollout_max_context_len", None)
        or getattr(args, "sglang_context_length", None)
        or args.seq_length
    )
    kv_heads = args.num_query_groups if getattr(args, "group_query_attention", False) else args.num_attention_heads
    kv_channels = getattr(args, "kv_channels", None) or args.hidden_size // args.num_attention_heads
    kv_token = args.num_layers * -(-kv_heads // engine_tp) * kv_channels * 2 * _rollout_kv_bytes(args)
    weights = _rollout_weight_bytes(args) * probe.base_params / engine_tp
    budget = fraction * probe.gpu_total - weights
    adapter = _rollout_dtype_bytes(args) * probe.adapter_full_params / engine_tp
    kv_per_slot = seqs * tokens * kv_token / engines
    n = int(budget // (adapter + kv_per_slot))
    detail = (
        f"{int(budget) >> 20}MiB left per engine GPU after {int(weights) >> 20}MiB of weights; "
        f"a slot holds {int(adapter) >> 20}MiB of adapter and {int(kv_per_slot) >> 20}MiB of KV "
        f"for {seqs}x{tokens} tokens over {engines} engines"
    )
    return n, detail


async def probe_slot_capacity(args: Namespace, backend, trainer) -> list[RankProbe]:
    """Warm up on one max-rank probe slot, then load a second one, run one
    max-size fb and an optimizer step through the real executor path, and
    measure every rank's head-room. Runs before the rollout engines launch;
    the trainer side is self-contained."""
    alpha = float(args.lora_alpha or 2 * args.lora_rank)
    row = _probe_row(args.max_tokens_per_gpu)
    await backend.load_slot(0, args.lora_rank, alpha)
    await backend.forward_backward(-1, [(0, row)], "cross_entropy", {})
    await backend.optim_step({0: _PROBE_ADAM_PARAMS})
    before = await trainer.multi_lora_memory_probe("before")
    await backend.load_slot(1, args.lora_rank, alpha)
    await backend.forward_backward(-1, [(1, row)], "cross_entropy", {})
    await backend.optim_step({1: _PROBE_ADAM_PARAMS})
    after = await trainer.multi_lora_memory_probe("after")
    await backend.unload_slot(1)
    await backend.unload_slot(0)

    weight = 2 if (args.bf16 or args.fp16) else 4
    scale = bytes_per_train_param(args) / (bytes_per_train_param(args) - weight)
    probes = [
        RankProbe(
            free_before=b["free"],
            free_after=a["free"],
            act_peak=a["act_peak"],
            adapter_local_params=a["adapter_local_params"],
            adapter_full_params=a["adapter_full_params"],
            expert_groups_per_slot=a["expert_groups_per_slot"],
            grouped_mm_max_groups=a["grouped_mm_max_groups"],
            gpu_total=a["gpu_total"],
            base_params=a["base_params"],
            slot_scale=scale,
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
    return {
        "tokens": [1] * (tokens + 1),
        "target_len": tokens,
        "target_tokens": [1] * tokens,
        "weights": [1.0] * tokens,
    }


def resolve_slot_capacity(args: Namespace, probes: list[RankProbe], keep_k: int) -> int:
    """min over the binding constraints; the log names which one bound."""
    margin = getattr(args, "train_memory_margin_bytes", 0) or 0
    n_gpu = min(probe.capacity(margin) for probe in probes)  # the worst rank rules

    host_budget = getattr(args, "engine_host_lora_budget_bytes", None)
    per_version_bytes = probes[0].adapter_full_params * 2  # engine CPU copies are bf16
    n_host = host_budget // (keep_k * per_version_bytes) if host_budget else None

    n = n_gpu if n_host is None else min(n_gpu, n_host)
    worst = min(probes, key=lambda probe: probe.capacity(margin))
    binding = "trainer GPU memory" if n_host is None or n_gpu <= n_host else "engine host RAM (keep-K copies)"
    once_fb = max_capacity_for_once_fb(probes)
    if once_fb is not None and once_fb[0] < n:
        n, binding = once_fb
    rollout = rollout_slot_capacity(args, worst)
    if rollout is not None and rollout[0] < n:
        n, binding = rollout[0], f"the rollout engines' memory ({rollout[1]})"
    assert n >= 1, (
        f"no room for one rank-{args.lora_rank} adapter slot: a slot needs "
        f"{worst.slot_bytes >> 20} MiB, free after the model and a max-size batch is "
        f"{(worst.free_before - worst.act_peak - margin) >> 20} MiB. "
        "Lower --lora-rank or --max-tokens-per-gpu."
    )
    logger.info(
        f"multi-LoRA capacity: {n} slots, bound by {binding} "
        f"(gpu={n_gpu}, host={n_host if n_host is not None else 'unchecked'}, "
        f"once_fb={once_fb[0] if once_fb else 'unchecked'}, rollout={rollout[0] if rollout else 'unchecked'}, "
        f"slot={worst.slot_bytes >> 20}MiB, act_peak={worst.act_peak >> 20}MiB, "
        f"adapter={per_version_bytes >> 20}MiB/version)"
    )
    return n
