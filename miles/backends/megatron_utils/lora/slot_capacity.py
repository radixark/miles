"""Slot capacity: how many resident LoRA tenants fit.

Measurement is the authority, sglang-style: load one probe slot, run one
max-size forward/backward and an optimizer step through the real executor
path, and read the deltas. The closed-form prediction only cross-checks the
measurement and explains the log line.
"""

import json
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

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
    free_before: int  # warmed runtime headroom with the measured probe-slot storage added back
    free_after: int  # bytes free with the probe slot resident (weights+grad+master+moments)
    act_peak: int  # transient peak of one max-size fb; shared across slots (single issue)
    adapter_local_params: int  # this rank's shard of one max-rank adapter
    adapter_full_params: int  # the unsharded adapter, for engine-side copies

    @property
    def slot_bytes(self) -> int:
        return self.free_before - self.free_after

    def capacity(self, margin_bytes: int) -> int:
        if self.slot_bytes <= 0:
            raise ValueError(f"probe measured a non-positive slot residency: {self.slot_bytes}")
        return max(int((self.free_before - self.act_peak - margin_bytes) // self.slot_bytes), 0)


def bytes_per_train_param(args: Namespace) -> int:
    """Per-slot resident bytes per LoRA param, from the precision flags."""
    weight = 2 if (args.bf16 or args.fp16) else 4
    grad = 4 if args.accumulate_allreduce_grads_in_fp32 else weight
    master = 4 if weight < 4 else 0  # mixed precision keeps an fp32 master; pure fp32 does not
    moments = 8  # per-slot torch Adam: fp32 exp_avg + exp_avg_sq (not precision-aware)
    return weight + grad + master + moments


def memory_snapshot(args: Namespace, model, phase: str, *, optimizer) -> dict:
    """Actor-side measurement half of the probe; the orchestration lives in
    probe_slot_capacity. ``before`` resets the peak tracker, ``after`` reads it."""
    import torch

    assert phase in ("before", "measure", "warmup", "after_fb", "after"), phase
    torch.cuda.synchronize()
    if phase in ("before", "measure"):
        torch.cuda.reset_peak_memory_stats()
    act_peak = max(0, torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated())
    torch.cuda.empty_cache()  # inactive allocator cache is reusable, not slot residency
    free, total = torch.cuda.mem_get_info()
    local, full = _adapter_param_counts(args, model)
    return {
        "data_parallel_size": args.data_parallel_size,
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
    from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

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


def _adapter_param_counts(args: Namespace, model) -> tuple[int, int]:
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


async def probe_slot_capacity(args: Namespace, backend, trainer) -> list[RankProbe]:
    """Load one max-rank probe slot, run one max-size fb and an optimizer step
    through the real executor path, and measure every rank's head-room. Runs
    before the rollout engines launch; the trainer side is self-contained."""
    before = await trainer.multi_lora_memory_probe("before")
    checkpoint_root = args.tinker_checkpoint_root or f"{args.save}/tinker"
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    # Preserve pre-forward evidence even when a backend error aborts the probe.
    # This bound excludes activation memory and is not a resolved slot count.
    start_report = {"bytes_per_train_param": bytes_per_train_param(args), "ranks": before}
    Path(checkpoint_root, "slot-probe-start.json").write_text(json.dumps(start_report, indent=2) + "\n")
    await backend.load_slot(0, args.lora_rank, float(args.lora_alpha or 2 * args.lora_rank))
    row = _probe_row(args.max_tokens_per_gpu)
    # Every data-parallel replica needs a max-size row. A single global row
    # gives empty shards when DP > 1 and cannot measure those replicas.
    dp_sizes = {snapshot["data_parallel_size"] for snapshot in before}
    assert len(dp_sizes) == 1
    slot_datums = [(0, row) for _ in range(dp_sizes.pop())]
    # Materialize process-wide workspaces and the slot's Adam state before
    # measuring steady-state activation headroom. The LR remains zero.
    await backend.forward_backward(-1, slot_datums, "cross_entropy", {})
    outcomes = await backend.optim_step({0: _PROBE_ADAM_PARAMS})
    assert "grad_norm" in outcomes[0], f"probe optimizer step did not succeed: {outcomes}"
    warmup = await trainer.multi_lora_memory_probe("warmup")
    raw_path = Path(checkpoint_root, "slot-probe-raw.json")
    raw = {"before": before, "warmup": warmup}
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    measure_before = await trainer.multi_lora_memory_probe("measure")
    await backend.forward_backward(-2, slot_datums, "cross_entropy", {})
    after_fb = await trainer.multi_lora_memory_probe("after_fb")
    outcomes = await backend.optim_step({0: _PROBE_ADAM_PARAMS})
    assert "grad_norm" in outcomes[0], f"probe optimizer step did not succeed: {outcomes}"
    after = await trainer.multi_lora_memory_probe("after")
    raw.update(measure_before=measure_before, after_fb=after_fb, after=after)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    await backend.unload_slot(0)

    probes = [
        RankProbe(
            # Count owned CUDA storages, including actual padded grad buffers.
            # Shared lazy allocations reduce headroom once, not once per slot.
            free_before=a["free"] + a["resident_slot_bytes"],
            free_after=a["free"],
            act_peak=a["act_peak"],
            adapter_local_params=a["adapter_local_params"],
            adapter_full_params=a["adapter_full_params"],
        )
        for a in after
    ]
    predicted = probes[0].adapter_local_params * bytes_per_train_param(args)
    if abs(probes[0].slot_bytes - predicted) > 0.2 * max(predicted, 1):
        logger.warning(
            f"measured slot bytes {probes[0].slot_bytes} diverge from predicted {predicted}: unaccounted per-slot memory; trust the measurement"
        )
    return probes


def _probe_row(tokens: int) -> dict:
    assert tokens >= 2
    return {"tokens": [1] * tokens, "target_len": tokens - 1, "weights": [1.0] * (tokens - 1)}


def resolve_slot_capacity(args: Namespace, probes: list[RankProbe], keep_k: int) -> int:
    """min over the binding constraints; the log names which one bound."""
    margin = getattr(args, "train_memory_margin_bytes", 0) or 0
    n_gpu = min(probe.capacity(margin) for probe in probes)  # the worst rank rules

    host_budget = getattr(args, "engine_host_lora_budget_bytes", None)
    per_version_bytes = probes[0].adapter_full_params * 2  # engine CPU copies are bf16
    n_host = host_budget // (keep_k * per_version_bytes) if host_budget else None

    n = n_gpu if n_host is None else min(n_gpu, n_host)
    worst = min(probes, key=lambda probe: probe.capacity(margin))
    assert (
        n >= 1
    ), f"no room for one rank-{args.lora_rank} adapter slot: a slot needs {worst.slot_bytes >> 20} MiB, free after the model and a max-size batch is {(worst.free_before - worst.act_peak - margin) >> 20} MiB. Lower --lora-rank or --max-tokens-per-gpu."
    binding = "trainer GPU memory" if n_host is None or n_gpu <= n_host else "engine host RAM (keep-K copies)"
    logger.info(
        f"multi-LoRA capacity: {n} slots, bound by {binding} (gpu={n_gpu}, host={n_host if n_host is not None else 'unchecked'}, slot={worst.slot_bytes >> 20}MiB, act_peak={worst.act_peak >> 20}MiB, adapter={per_version_bytes >> 20}MiB/version)"
    )
    return n
