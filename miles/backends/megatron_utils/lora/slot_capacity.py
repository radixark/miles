"""Multi-LoRA slot capacity: auto = min(probed trainer memory, engine memory, torch._grouped_mm's group limit)."""

import logging
from argparse import Namespace
from dataclasses import dataclass

import torch

from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

logger = logging.getLogger(__name__)

AUTO_SLOT_CAPACITY = -1  # --multi-lora-n-adapters auto
PROBE_SLOTS = 1  # pool size of the probe trainer
PROBE_SLOT = 0
_PROBE_BATCH_ID = -1
# torch._grouped_mm rejects 1024 groups and up (torch 2.13); the expert adapters run one group per (slot, local expert)
GROUPED_MM_MAX_GROUPS = 1023
# sequences one slot samples at once, for the engine bound: one group of samples per prompt
DEFAULT_ROLLOUT_SEQS_PER_SLOT = 8
# every SGLang TP-rank process keeps a host copy of each loaded adapter version (~1.6 GB at rank 16 on Qwen3-30B-A3B)
ENGINE_LOADED_VERSIONS_HEADROOM = 16

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
    expert_groups_per_slot: int = 0  # grouped-GEMM groups one slot adds: this rank's local experts (0: none)
    # unsharded totals for the rollout engines' side of the estimate (0: not reported)
    gpu_total_bytes: int = 0
    base_dense_params: int = 0
    base_expert_params: int = 0
    adapter_dense_params: int = 0
    adapter_expert_params_total: int = 0

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


def expert_data_parallel_size(args: Namespace, dp_size: int) -> int:
    """Ranks that replicate one expert's params: dp * tp * cp // (ep * etp)."""
    tp = getattr(args, "tensor_model_parallel_size", 1) or 1
    cp = getattr(args, "context_parallel_size", 1) or 1
    ep = getattr(args, "expert_model_parallel_size", 1) or 1
    etp = getattr(args, "expert_tensor_parallel_size", None) or tp
    return max(1, dp_size * tp * cp // (ep * etp))


def predicted_slot_bytes(args: Namespace, probe: RankProbe, dp_size: int) -> float:
    """Closed-form counterpart of the measurement: dense state shared over DP, expert state over expert-DP."""
    shared, state = _weight_and_grad_bytes(args), _optimizer_state_bytes(args)
    dense = probe.adapter_local_params - probe.adapter_expert_params
    return dense * (shared + state / dp_size) + probe.adapter_expert_params * (
        shared + state / expert_data_parallel_size(args, dp_size)
    )


def memory_snapshot(model, slot_optimizer, phase: str, args: Namespace | None = None) -> dict:
    """Actor-side half of the probe: ``reset`` arms the peak tracker, ``measure`` reads it and the slot's residency."""
    torch.cuda.synchronize()
    if phase == "reset":
        torch.cuda.reset_peak_memory_stats()
        return {}
    assert phase == "measure", f"unknown memory_snapshot phase {phase!r}"
    act_peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    torch.cuda.empty_cache()  # cached-but-unused blocks are head-room, not residency
    free, total = torch.cuda.mem_get_info()
    local_params, expert_params = adapter_param_counts(model, PROBE_SLOT)
    return {
        "free": free,
        "slot_bytes": resident_slot_bytes(model, slot_optimizer, PROBE_SLOT),
        "act_peak": max(act_peak, 0),
        "adapter_local_params": local_params,
        "adapter_expert_params": expert_params,
        "expert_groups_per_slot": expert_groups_per_slot(model),
        "gpu_total_bytes": total,
        **(unsharded_param_counts(args, model, PROBE_SLOT) if args is not None else {}),
    }


def unsharded_param_counts(args: Namespace, model, slot: int) -> dict:
    """Whole-model and whole-adapter param counts, dense and expert, undoing this rank's TP/ETP/EP sharding."""
    tp = getattr(args, "tensor_model_parallel_size", 1) or 1
    ep = getattr(args, "expert_model_parallel_size", 1) or 1
    etp = getattr(args, "expert_tensor_parallel_size", None) or tp
    slot_ids = {id(param) for param in adapter_slot_parameters(model, slot)}
    chunks = model if isinstance(model, (list, tuple)) else [model]
    counts = {
        "base_dense_params": 0,
        "base_expert_params": 0,
        "adapter_dense_params": 0,
        "adapter_expert_params_total": 0,
    }
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            is_adapter = ".adapters." in name
            if is_adapter and id(param) not in slot_ids:
                continue  # the pool's other slots: one adapter is what the engines hold per slot
            expert = ".experts." in name
            multiplier = (etp if expert else tp) if getattr(param, "tensor_model_parallel", False) else 1
            if expert:
                multiplier *= ep
            if is_adapter:
                key = "adapter_expert_params_total" if expert else "adapter_dense_params"
            else:
                key = "base_expert_params" if expert else "base_dense_params"
            counts[key] += param.numel() * multiplier
    return counts


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


def resident_slot_bytes(model, slot_optimizer, slot: int) -> int:
    """CUDA bytes one resident slot owns (weights, grads, fp32 masters, Adam moments), each allocation once."""
    storages: dict[tuple[int, int], int] = {}

    def record(tensor) -> None:
        if tensor is not None and getattr(tensor, "is_cuda", False):
            storage = tensor.untyped_storage()
            storages[(tensor.device.index, storage.data_ptr())] = storage.nbytes()

    for param in adapter_slot_parameters(model, slot):
        for tensor in (param, param.grad, getattr(param, "main_grad", None), getattr(param, "main_param", None)):
            record(tensor)
    for child in slot_optimizer_children(slot_optimizer):
        for param in child.get_parameters():  # the fp32 masters the mixed-precision wrapper steps
            record(param)
        for state in child.optimizer.state.values():
            for value in state.values():
                record(value)
    return sum(storages.values())


def slot_optimizer_children(slot_optimizer) -> list:
    """The LayerWise children of one slot's SlotOptimizer (each wraps a torch optimizer over fp32 masters)."""
    return list(slot_optimizer._inner.chained_optimizers) if slot_optimizer is not None else []


async def probe_slot_capacity(args: Namespace, backend, trainer, dp_size: int = 1) -> list[RankProbe]:
    """Load one max-rank probe slot, warm up, then measure one max-size step on every rank."""
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
    # the same shape encoding.build_datum produces: tokens = prompt + targets[-1:], next-token targets
    return {
        "tokens": [1] * tokens,
        "target_tokens": [1] * (tokens - 1),
        "target_len": tokens - 1,
        "weights": [1.0] * (tokens - 1),
    }


_QUANT_BYTES = (
    ("fp4", 0.5),
    ("int4", 0.5),
    ("awq", 0.5),
    ("gptq", 0.5),
    ("fp8", 1),
    ("int8", 1),
)  # --sglang-quantization
_DTYPE_BYTES = {"float32": 4, "fp32": 4, "float16": 2, "half": 2, "fp16": 2, "bfloat16": 2, "bf16": 2}


def engine_dtype_bytes(args: Namespace) -> float:
    """Bytes per element of the engines' dtype (--sglang-dtype; auto: the bf16/fp16 checkpoint)."""
    return _DTYPE_BYTES.get((getattr(args, "sglang_dtype", None) or "auto").lower(), 2)


def engine_weight_bytes(args: Namespace) -> float:
    """Bytes per base weight on the engines: --sglang-quantization when set, else the engines' dtype."""
    quantization = (getattr(args, "sglang_quantization", None) or "").lower()
    return next((size for key, size in _QUANT_BYTES if key in quantization), engine_dtype_bytes(args))


def engine_kv_bytes(args: Namespace) -> float:
    """Bytes per KV element on the engines: 1 for an fp8 --sglang-kv-cache-dtype, else the engines' dtype."""
    return 1 if "fp8" in (getattr(args, "sglang_kv_cache_dtype", None) or "auto").lower() else engine_dtype_bytes(args)


def engine_slot_capacity(args: Namespace, probe: RankProbe) -> tuple[int, dict] | None:
    """Slots every engine GPU can hold with all of them sampling at once; None when off (seqs 0) or unreported."""
    seqs = getattr(args, "multi_lora_rollout_seqs_per_slot", None)
    if seqs is None:
        seqs = DEFAULT_ROLLOUT_SEQS_PER_SLOT
    if seqs <= 0 or not probe.gpu_total_bytes or not probe.base_dense_params:
        return None
    engine_tp = args.rollout_num_gpus_per_engine
    # without --sglang-ep-size the engines shard the experts across their TP ranks, as EP = TP would
    engine_ep = getattr(args, "sglang_ep_size", None) or engine_tp
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
    kv_token = args.num_layers * -(-kv_heads // engine_tp) * kv_channels * 2 * engine_kv_bytes(args)  # K and V

    weights = engine_weight_bytes(args) * (probe.base_dense_params / engine_tp + probe.base_expert_params / engine_ep)
    budget = fraction * probe.gpu_total_bytes - weights
    adapter = engine_dtype_bytes(args) * (  # the LoRA buffers stay in the engines' dtype
        probe.adapter_dense_params / engine_tp + probe.adapter_expert_params_total / engine_ep
    )
    kv_per_slot = seqs * tokens * kv_token / engines  # one slot's sequences spread over the engines
    per_slot = adapter + kv_per_slot
    n = int(budget // per_slot) if per_slot > 0 else 0
    return n, {
        "engine_budget_mib": int(budget) >> 20,
        "engine_weights_mib": int(weights) >> 20,
        "adapter_mib": int(adapter) >> 20,
        "kv_per_slot_mib": int(kv_per_slot) >> 20,
        "seqs_per_slot": seqs,
        "tokens_per_seq": tokens,
        "engines": engines,
    }


def engine_loaded_adapter_cap(n_slots: int) -> int:
    """--sglang-max-loaded-loras when unset: every slot's current version plus a few draining predecessors."""
    return n_slots + ENGINE_LOADED_VERSIONS_HEADROOM


def resolve_slot_capacity(args: Namespace, probes: list[RankProbe]) -> int:
    """min over the trainer's memory, the grouped-GEMM limit and the engines' memory; logs the binding one."""
    margin = getattr(args, "train_memory_margin_bytes", 0) or 0
    worst = min(probes, key=lambda probe: probe.capacity(margin))
    n_memory = worst.capacity(margin)
    assert n_memory >= 1, (
        f"no room for one rank-{args.lora_rank} adapter slot: a slot needs {worst.slot_bytes >> 20} MiB, "
        "free after the model, a max-size batch and the margin is "
        f"{(worst.free + worst.slot_bytes - worst.act_peak - margin) >> 20} MiB. "
        "Lower --lora-rank or --max-tokens-per-gpu."
    )
    bounds = {"trainer memory": n_memory}
    groups = max(probe.expert_groups_per_slot for probe in probes)
    if groups:
        bounds[f"torch._grouped_mm's {GROUPED_MM_MAX_GROUPS}-group limit ({groups} local experts per slot)"] = (
            GROUPED_MM_MAX_GROUPS // groups
        )
    engine = engine_slot_capacity(args, probes[0])
    if engine is not None:
        n_engine, detail = engine
        bounds[f"the rollout engines' memory with every slot sampling at once ({detail})"] = n_engine
    binding, n = min(bounds.items(), key=lambda item: item[1])
    assert n >= 1, f"no room for one slot: {bounds}"
    summary = ", ".join(f"{name.split(' (')[0]}={count}" for name, count in bounds.items())
    logger.info(
        f"multi-LoRA capacity: {n} slots, bound by {binding} [{summary}] "
        f"(slot={worst.slot_bytes >> 20}MiB, act_peak={worst.act_peak >> 20}MiB, "
        f"free={worst.free >> 20}MiB, margin={margin >> 20}MiB)"
    )
    if binding != "trainer memory":
        logger.warning(
            f"multi-LoRA capacity is bound by {binding.split(' (')[0]}: "
            f"the trainer could hold {n_memory} slots but only {n} are usable"
        )
    return n
