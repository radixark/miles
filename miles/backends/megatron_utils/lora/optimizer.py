"""Per-tenant slot optimizers for multi-LoRA.

Each live slot owns an independent LayerWiseDistributedOptimizer over exactly
its adapter parameters, built at load and destroyed with the slot, so a fresh
tenant never inherits optimizer state. Plain DDP all-reduce keeps full gradients
available to each slot's optimizer."""

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
    assert not args.backend.use_distributed_optimizer, (
        "multi-LoRA per-slot optimizers require use_distributed_optimizer=False: "
        "LayerWise sharding requires full parameter gradients instead of byte-level ZeRO shards"
    )
    assert (
        args.backend.bf16 and not args.backend.fp16
    ), "multi-LoRA per-slot optimizers require bf16 (no dynamic loss scaler)"
    assert (
        args.backend.optimizer or ""
    ).lower() == "adam", (
        f"multi-LoRA per-slot optimizers only implement Adam semantics; got optimizer={args.backend.optimizer!r}"
    )
    for config, flag in (
        (args.backend, "optimizer_cpu_offload"),
        (args, "stream_optimizer_state_to_disk"),
        (args, "rematerialize_param_from_master_weight"),
    ):
        assert not getattr(
            config, flag, False
        ), f"--{flag.replace('_', '-')} is not supported with multi-LoRA slots"  # config-access-exempt: attribute selected at runtime from flag


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
        if getattr(child, "optimizer", None) is not None
        and child.get_parameters()  # config-access-exempt: optimizer wrappers differ in optimizer support
    ]
    config.bf16 = True
    return optimizers


class SlotOptimizer:
    """One tenant's optimizer over one adapter slot.

    Wraps a per-slot LayerWiseDistributedOptimizer, so every method touches
    only this slot's parameters and state: reloads cannot round other tenants'
    FP32 masters, and the parameter all-gather moves only this slot."""

    def __init__(self, args: Namespace, model, slot: int, slot_params: list[torch.nn.Parameter]) -> None:
        self.slot = slot
        self._slot_params = slot_params
        self._trains_mtp = any(getattr(param, "grad_norm_group", None) == "mtp" for param in slot_params)
        assert slot_params, f"adapter slot {slot} has no parameters; is this a multi-LoRA model?"

        config = OptimizerConfig(
            **{
                f.name: getattr(args.backend, f.name) for f in fields(OptimizerConfig) if hasattr(args.backend, f.name)
            }  # config-access-exempt: attribute selected at runtime from f.name
        )
        config.timers = None
        base_optimizers = _build_slot_base_optimizers(
            config, model, slot_params, use_gloo_process_groups=args.backend.use_gloo_process_groups
        )
        assert base_optimizers, f"adapter slot {slot} produced no optimizer children"
        self._inner = LayerWiseDistributedOptimizer(
            base_optimizers,
            config,
            ProcessGroupCollection.use_mpu_process_groups(),
            init_state_fn_list=[_adam_init_state_fn] * len(base_optimizers),
            model_chunks=list(model),
        )
        # params are scattered whole across DP ranks; per-child norm/clip reductions must span the world
        for child in self._inner.chained_optimizers:
            child.grad_stats_parallel_group = None

    @property
    def chained_optimizers(self) -> list:
        """The slot's Float16 children: masters in ``fp32_from_float16_groups``, Adam state in ``.optimizer``."""
        return self._inner.chained_optimizers

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
        norms_and_params = {"grad_norm": self._norm_and_parameters(None)}
        if self._trains_mtp:
            norms_and_params["mtp_grad_norm"] = self._norm_and_parameters("mtp")
        # BF16 has no grad scaler; the all-reduced norm detects inf/nan on every rank
        if any(not math.isfinite(norm) for norm, _ in norms_and_params.values()):
            return {"skipped_nonfinite": 1.0}
        for norm, params in norms_and_params.values():
            if clip_grad > 0.0 and params:
                clip_grad_by_total_norm_fp32(params, clip_grad, norm, False)
        for child in self._inner.chained_optimizers:
            child.step_with_ready_grads()
        return {name: float(norm) for name, (norm, _) in norms_and_params.items()}

    def _norm_and_parameters(self, norm_group: str | None) -> tuple[float, list]:
        grads = []
        params = []
        for child in self._inner.chained_optimizers:
            grads.extend(child.get_grads_for_grad_norm(norm_group) if norm_group else child.get_grads_for_grad_norm())
            params.extend(p for p in child.get_parameters() if getattr(p, "grad_norm_group", None) == norm_group)
        return get_grad_norm_fp32(grads, grad_stats_parallel_group=None), params

    def allgather_params(self) -> None:
        self._inner.allgather_params()

    def sharded_state(self, model_sharded_state_dict: dict, *, is_loading: bool) -> dict:
        """torch_dist form: every state tensor carries its param's global coordinates."""
        return self._inner.sharded_state_dict(model_sharded_state_dict, is_loading=is_loading)

    def load_sharded_state(self, loaded: dict) -> None:
        self._inner.load_state_dict(loaded)

    def reset_adam_state(self) -> None:
        """Drop the Adam moments and step, keeping the masters: the state of a slot that never stepped."""
        for child in self._inner.chained_optimizers:
            child.optimizer.state.clear()
            for group in child.optimizer.param_groups:
                group.pop("step", None)

    def zero_grads(self) -> None:
        """Zero the slot's gradients everywhere they live: the DDP ``main_grad``
        buffer views and any lingering ``grad``/``main_param.grad`` references."""
        for param in self._slot_params:
            if (
                main_grad := getattr(param, "main_grad", None)
            ) is not None:  # config-access-exempt: main_grad is optional backend-attached tensor metadata
                main_grad.zero_()
            param.grad = None
            if (
                main_param := getattr(param, "main_param", None)
            ) is not None:  # config-access-exempt: main_param is optional backend-attached tensor metadata
                main_param.grad = None


@contextmanager
def accumulate_gradients(model_chunks):
    # Synchronizing retained expert gradients would scale earlier requests again.
    buffers = [buffer for chunk in model_chunks for buffer in chunk.buffers + chunk.expert_parallel_buffers]
    accumulated = [buffer.grad_data.clone() for buffer in buffers]
    for chunk in model_chunks:
        chunk.zero_grad_buffer()
    try:
        yield
    except BaseException:
        for buffer, previous in zip(buffers, accumulated, strict=True):
            buffer.grad_data.copy_(previous)
        raise
    else:
        for buffer, previous in zip(buffers, accumulated, strict=True):
            buffer.grad_data.add_(previous)


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
