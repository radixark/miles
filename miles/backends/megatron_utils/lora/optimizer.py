"""Per-tenant slot optimizers for multi-LoRA.

Each live slot owns an independent LayerWiseDistributedOptimizer over exactly
its adapter parameters, built at load and destroyed with the slot, so a fresh
tenant never inherits optimizer state. Requires plain DDP all-reduce
(use_distributed_optimizer OFF) so cross-batch gradient retention stays
idempotent."""

import logging
import math
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import fields

import torch
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


logger = logging.getLogger(__name__)


def validate_multi_lora_optimizer_args(args: Namespace) -> None:
    """Reject launch options the per-slot optimizers cannot honor."""
    assert not args.use_distributed_optimizer, (
        "multi-LoRA per-slot optimizers require use_distributed_optimizer=False: "
        "gradient retention relies on all-reduce idempotency, and LayerWise "
        "sharding replaces byte-level ZeRO"
    )
    assert args.bf16 and not args.fp16, "multi-LoRA per-slot optimizers require bf16 (no dynamic loss scaler)"
    assert (
        args.optimizer or ""
    ).lower() == "adam", (
        f"multi-LoRA per-slot optimizers only implement Adam semantics; got optimizer={args.optimizer!r}"
    )
    for flag in ("optimizer_cpu_offload", "stream_optimizer_state_to_disk", "rematerialize_param_from_master_weight"):
        assert not getattr(args, flag, False), f"--{flag.replace('_', '-')} is not supported with multi-LoRA slots"


def adapter_slot_parameters(model, slot: int) -> list[torch.nn.Parameter]:
    """All parameters belonging to one adapter slot, across model chunks."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    parameters = []
    seen = set()
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if not isinstance(module, MultiLoRALinear):
                continue
            for param in module.adapters[slot].parameters():
                if id(param) not in seen:
                    parameters.append(param)
                    seen.add(id(param))
    return parameters


def _adam_init_state_fn(opt, config=None):
    for group in opt.param_groups:
        for p in group["params"]:
            if len(opt.state[p]) == 0:
                opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)


@contextmanager
def _only_slot_trainable(model_chunks, slot_params: list[torch.nn.Parameter]):
    """Temporarily freeze every trainable param outside ``slot_params`` so the
    stock param-group builder sees exactly one slot (the Muon construction
    pattern from megatron's ``get_megatron_muon_optimizer``)."""
    slot_ids = {id(p) for p in slot_params}
    frozen = []
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            if param.requires_grad and id(param) not in slot_ids:
                param.requires_grad = False
                frozen.append(param)
    try:
        yield
    finally:
        for param in frozen:
            param.requires_grad = True


def _build_slot_base_optimizers(config, model, slot_params, *, use_gloo_process_groups: bool) -> list:
    # create FP32 master wrappers only after LayerWise shards the base optimizers
    config.bf16 = False
    with _only_slot_trainable(model, slot_params):
        chained = get_megatron_optimizer(config, list(model), use_gloo_process_groups=use_gloo_process_groups)
    optimizers = [
        child.optimizer
        for child in chained.chained_optimizers
        if getattr(child, "optimizer", None) is not None and child.get_parameters()
    ]
    config.bf16 = True
    return optimizers


class SlotOptimizer:
    """One tenant's optimizer over one adapter slot.

    Wraps a per-slot LayerWiseDistributedOptimizer, so every method touches
    only this slot's parameters and state: reloads cannot round other tenants'
    FP32 masters, and the parameter all-gather moves only this slot."""

    def __init__(self, args: Namespace, model, slot: int) -> None:
        self.slot = slot
        self._model = model
        slot_params = adapter_slot_parameters(model, slot)
        assert slot_params, f"adapter slot {slot} has no parameters; is this a multi-LoRA model?"

        config = OptimizerConfig(
            **{f.name: getattr(args, f.name) for f in fields(OptimizerConfig) if hasattr(args, f.name)}
        )
        config.timers = None
        base_optimizers = _build_slot_base_optimizers(
            config, model, slot_params, use_gloo_process_groups=args.use_gloo_process_groups
        )
        assert base_optimizers, f"adapter slot {slot} produced no optimizer children"
        self._inner = LayerWiseDistributedOptimizer(
            base_optimizers,
            config,
            ProcessGroupCollection.use_mpu_process_groups(),
            init_state_fn_list=[_adam_init_state_fn] * len(base_optimizers),
        )
        # params are scattered whole across DP ranks; per-child norm/clip reductions must span the world
        for child in self._inner.chained_optimizers:
            child.grad_stats_parallel_group = None

    def apply_adam_params(self, adam_params: dict) -> None:
        for child in self._inner.chained_optimizers:
            for group in child.param_groups:
                group["lr"] = adam_params["learning_rate"]
                group["betas"] = (adam_params["beta1"], adam_params["beta2"])
                group["eps"] = adam_params["eps"]
                group["weight_decay"] = adam_params["weight_decay"]

    def prepare_grads(self) -> None:
        for child in self._inner.chained_optimizers:
            child.prepare_grads()

    def clip_and_step(self, clip_grad: float) -> dict:
        """-> {"grad_norm": x} stepped, {"skipped_nonfinite": 1.0} dropped.
        Collective: the slot's grad norm is all-reduced over the world."""
        grads_for_norm = []
        slot_params = []
        for child in self._inner.chained_optimizers:
            grads_for_norm += child.get_grads_for_grad_norm()
            slot_params += child.get_parameters()
        slot_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        # BF16 has no grad scaler; the all-reduced norm detects inf/nan on every rank
        if not math.isfinite(slot_norm):
            return {"skipped_nonfinite": 1.0}
        if clip_grad > 0.0 and slot_params:
            clip_grad_by_total_norm_fp32(slot_params, clip_grad, slot_norm, False)
        for child in self._inner.chained_optimizers:
            child.step_with_ready_grads()
        return {"grad_norm": float(slot_norm)}

    def allgather_params(self) -> None:
        self._inner.allgather_params()

    def reload_masters(self) -> None:
        """Refresh this slot's FP32 masters from its model parameters."""
        self._inner.reload_model_params()

    def sharded_state(self, model_sharded_state_dict: dict, *, is_loading: bool) -> dict:
        """torch_dist form: every state tensor carries its param's global coordinates."""
        return self._inner.sharded_state_dict(model_sharded_state_dict, is_loading=is_loading)

    def load_sharded_state(self, loaded: dict) -> None:
        self._inner.load_state_dict(loaded)

    def zero_grads(self) -> None:
        """Zero the slot's gradients everywhere they live: the DDP ``main_grad``
        buffer views and any lingering ``grad``/``main_param.grad`` references."""
        for param in adapter_slot_parameters(self._model, self.slot):
            if (main_grad := getattr(param, "main_grad", None)) is not None:
                main_grad.zero_()
            param.grad = None
            if (main_param := getattr(param, "main_param", None)) is not None:
                main_param.grad = None


def reset_grad_metadata_keep_grads(model_chunks) -> None:
    """Reset DDP bookkeeping while retaining each slot's gradient accumulation window."""
    for model_chunk in model_chunks:
        if getattr(model_chunk.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in model_chunk.params_with_grad:
                param.grad_added_to_main_grad = False
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.reset()


def step_slot_optimizers(
    slot_optimizers: dict[int, SlotOptimizer],
    adam_params_by_slot: dict[int, dict],
) -> dict[int, dict]:
    """Step slots in the same order on every rank; execution failures invalidate the cell."""
    slots = sorted(adam_params_by_slot)
    for slot in slots:
        slot_optimizers[slot].apply_adam_params(adam_params_by_slot[slot])
        slot_optimizers[slot].prepare_grads()

    outcomes = {
        slot: slot_optimizers[slot].clip_and_step(adam_params_by_slot[slot]["grad_clip_norm"]) for slot in slots
    }
    for slot in slots:
        slot_optimizers[slot].zero_grads()
    for slot in slots:
        if "grad_norm" in outcomes[slot]:
            slot_optimizers[slot].allgather_params()
    return outcomes
