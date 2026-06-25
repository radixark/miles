"""Runtime for Predictive Routing Replay (PR²) — arXiv:2606.00395.

Holds:
  * ``PredictiveReplayController`` — the single control-plane object
    coordinating per-layer router state, the global predictive action
    (``RECORD`` / ``SKIP_PREDICTIVE`` / ``COMPUTE_PREDICTIVE_LOSS`` /
    ``DISABLED``), and the recorded predictive microbatch queue.
  * The ``TopKRouter.forward`` patch installed by
    ``apply_predictive_router_replay_patch``.
  * ``initialize_predictive_router_modules`` — builds the per-layer
    ``bias_predictor = nn.Linear(hidden, num_experts, bias=False)`` head
    with zero-initialized weights (paper §4.1) and registers state.

The stateless helpers (stabilization layer, loss / sample reweighting,
synthetic-loss sync) live in
``predictive_router_stabilization.py`` and ``predictive_router_loss.py``
and are re-exported here so existing callers can keep importing from
this module.
"""
import logging
import os
from contextlib import contextmanager
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn

from miles.utils.replay_base import routing_replay_manager

# The stabilization / loss helpers used to live inline in this file. They are
# now split into two stateless modules; re-export so existing imports of
# `from miles.backends.megatron_utils.predictive_router_replay import X`
# (actor.py, model.py, update_weight/common.py, every test_predictive_* file)
# keep working without churn.
from miles.backends.megatron_utils.predictive_router_loss import (  # noqa: F401
    build_synthetic_predictive_loss,
    build_topk_boundary_loss_weights,
    compute_predictive_loss,
)
from miles.backends.megatron_utils.predictive_router_stabilization import (  # noqa: F401
    PREDICTIVE_LAYER_SCALE_SCHEDULES,
    compute_predictive_bias_ratio,
    compute_predictive_layer_scale,
    compute_topk_boundary_margin,
    stabilize_predictive_delta_logits,
)

logger = logging.getLogger(__name__)


class RouterPredictiveAction(str, Enum):
    DISABLED = "disabled"
    RECORD = "record"
    SKIP_PREDICTIVE = "skip_predictive"
    COMPUTE_PREDICTIVE_LOSS = "compute_predictive_loss"


def is_predictive_router_parameter_name(name: str) -> bool:
    return "bias_predictor" in name


def calculate_topk_accuracy(
    *,
    topk: int,
    logits1: torch.Tensor | None = None,
    logits2: torch.Tensor | None = None,
    topk_indices1: torch.Tensor | None = None,
    topk_indices2: torch.Tensor | None = None,
) -> float:
    if topk_indices1 is None:
        if logits1 is None:
            raise ValueError("logits1 must be provided when topk_indices1 is None.")
        _, topk_indices1 = torch.topk(logits1, k=topk, dim=-1)
    if topk_indices2 is None:
        if logits2 is None:
            raise ValueError("logits2 must be provided when topk_indices2 is None.")
        _, topk_indices2 = torch.topk(logits2, k=topk, dim=-1)

    matches = (topk_indices1.unsqueeze(-1) == topk_indices2.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean().item()


def _ensure_bias_predictor_runtime_placement(
    *,
    bias_predictor: nn.Module,
    reference_tensor: torch.Tensor,
) -> None:
    target_device = reference_tensor.device
    target_dtype = reference_tensor.dtype

    for parameter in bias_predictor.parameters():
        if parameter.device == target_device and parameter.dtype == target_dtype:
            continue
        parameter.data = parameter.data.to(device=target_device, dtype=target_dtype)
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.to(device=target_device, dtype=target_dtype)
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is not None:
            parameter.main_grad = main_grad.to(device=target_device, dtype=target_dtype)

    for buffer_name, buffer in bias_predictor.named_buffers():
        if buffer.device == target_device and buffer.dtype == target_dtype:
            continue
        setattr(bias_predictor, buffer_name, buffer.to(device=target_device, dtype=target_dtype))


def _iter_module_chunks(model_chunks):
    if isinstance(model_chunks, torch.nn.Module):
        model_chunks = [model_chunks]
    for model_chunk in model_chunks:
        yield getattr(model_chunk, "module", model_chunk)


def _build_bias_predictor(router) -> nn.Linear:
    gating_weight = getattr(router.gating, "weight", None)
    if gating_weight is not None:
        in_features = gating_weight.shape[1]
        out_features = gating_weight.shape[0]
        device = gating_weight.device
        dtype = gating_weight.dtype
    else:
        in_features = router.config.hidden_size
        out_features = router.config.num_moe_experts
        device = None
        dtype = None

    bias_predictor = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
    nn.init.zeros_(bias_predictor.weight)
    if device is not None or dtype is not None:
        bias_predictor = bias_predictor.to(device=device, dtype=dtype)

    for param in bias_predictor.parameters():
        setattr(param, "is_bias_predictor", True)
        setattr(param, "allreduce", True)
        setattr(param, "sequence_parallel", False)
        setattr(param, "tensor_model_parallel", False)
        setattr(param, "partition_dim", 0)
        setattr(param, "partition_stride", 1)

    return bias_predictor


def initialize_predictive_router_modules(
    *,
    model_chunks,
    enabled: bool,
    loss_type: str,
    lr_mult: float,
    layer_scale_schedule: str = "none",
    layer_scale_min: float = 1.0,
    boundary_loss_max_weight: float | None = None,
    boundary_loss_min_margin: float = 1e-4,
) -> None:
    from megatron.core.transformer.moe.router import TopKRouter

    PredictiveRouterReplayState.reset_registry()

    seen_modules = set()
    for module_chunk in _iter_module_chunks(model_chunks):
        for submodule in module_chunk.modules():
            if not isinstance(submodule, TopKRouter):
                continue
            if id(submodule) in seen_modules:
                continue
            seen_modules.add(id(submodule))

            submodule.config.enable_router_bias_predictor = enabled
            submodule.config.bias_predictor_loss_type = loss_type
            submodule.config.bias_predictor_lr_mult = lr_mult
            submodule.config.predictive_layer_scale_schedule = layer_scale_schedule
            submodule.config.predictive_layer_scale_min = layer_scale_min
            submodule.config.predictive_boundary_loss_max_weight = boundary_loss_max_weight
            submodule.config.predictive_boundary_loss_min_margin = boundary_loss_min_margin

            if not enabled:
                submodule.predictive_router_replay = None
                if hasattr(submodule, "bias_predictor"):
                    submodule.bias_predictor = None
                continue

            PredictiveRouterReplayState.register_router(submodule)
            submodule.bias_predictor = _build_bias_predictor(submodule)


def predictive_debug_param_stats_enabled() -> bool:
    return os.getenv("PREDICTIVE_DEBUG_PARAM_STATS", "0") == "1"


def collect_predictive_param_stats(model_chunks) -> dict[str, float | int]:
    from megatron.core.transformer.moe.router import TopKRouter

    stats: dict[str, float | int] = {
        "num_predictor_params": 0,
        "num_predictor_params_with_grad": 0,
        "num_predictor_params_with_main_grad": 0,
        "weight_abs_sum": 0.0,
        "grad_abs_sum": 0.0,
        "main_grad_abs_sum": 0.0,
        "max_weight_abs": 0.0,
        "max_grad_abs": 0.0,
        "max_main_grad_abs": 0.0,
        "weight_l2_norm": 0.0,
        "grad_l2_norm": 0.0,
        "main_grad_l2_norm": 0.0,
    }
    weight_sq_sum = 0.0
    grad_sq_sum = 0.0
    main_grad_sq_sum = 0.0

    seen_params = set()
    for module_chunk in _iter_module_chunks(model_chunks):
        for submodule in module_chunk.modules():
            if not isinstance(submodule, TopKRouter):
                continue
            bias_predictor = getattr(submodule, "bias_predictor", None)
            if bias_predictor is None:
                continue
            for param in bias_predictor.parameters():
                if id(param) in seen_params:
                    continue
                seen_params.add(id(param))
                stats["num_predictor_params"] += 1
                detached_param = param.detach()
                weight_abs = float(detached_param.abs().sum().item())
                stats["weight_abs_sum"] += weight_abs
                stats["max_weight_abs"] = max(float(stats["max_weight_abs"]), float(detached_param.abs().max().item()))
                weight_sq_sum += float(detached_param.float().pow(2).sum().item())
                if param.grad is not None:
                    stats["num_predictor_params_with_grad"] += 1
                    grad_abs = float(param.grad.detach().abs().sum().item())
                    stats["grad_abs_sum"] += grad_abs
                    stats["max_grad_abs"] = max(float(stats["max_grad_abs"]), float(param.grad.detach().abs().max().item()))
                    grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
                main_grad = getattr(param, "main_grad", None)
                if main_grad is not None:
                    stats["num_predictor_params_with_main_grad"] += 1
                    main_grad_abs = float(main_grad.detach().abs().sum().item())
                    stats["main_grad_abs_sum"] += main_grad_abs
                    stats["max_main_grad_abs"] = max(
                        float(stats["max_main_grad_abs"]), float(main_grad.detach().abs().max().item())
                    )
                    main_grad_sq_sum += float(main_grad.detach().float().pow(2).sum().item())

    stats["weight_l2_norm"] = weight_sq_sum**0.5
    stats["grad_l2_norm"] = grad_sq_sum**0.5
    stats["main_grad_l2_norm"] = main_grad_sq_sum**0.5

    return stats


def apply_predictive_router_replay_patch() -> None:
    from megatron.core.transformer.moe.moe_utils import apply_random_logits
    from megatron.core.transformer.moe.router import TopKRouter

    if hasattr(TopKRouter, "_predictive_router_replay_patched"):
        return

    original_forward = TopKRouter.forward

    def patched_forward(self, input: torch.Tensor):
        predictive_state = getattr(self, "predictive_router_replay", None)
        bias_predictor = getattr(self, "bias_predictor", None)
        predictive_controller = get_predictive_replay_controller()

        if predictive_state is None or bias_predictor is None:
            return original_forward(self, input)

        _ensure_bias_predictor_runtime_placement(bias_predictor=bias_predictor, reference_tensor=input)

        predictive_action = predictive_state.predictive_action
        if predictive_action in {None, RouterPredictiveAction.DISABLED, RouterPredictiveAction.SKIP_PREDICTIVE}:
            return original_forward(self, input)

        self._maintain_float32_expert_bias()
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)

        if predictive_action == RouterPredictiveAction.RECORD:
            with torch.no_grad():
                predictive_state.record_predictive_data(input, logits)
                predicted_delta_logits = bias_predictor(input).detach()
                predicted_delta_logits, _ = stabilize_predictive_delta_logits(
                    predicted_delta_logits=predicted_delta_logits,
                    reference_logits=logits.detach(),
                    layer_idx=predictive_state.layer_idx,
                    num_layers=len(predictive_controller.router_states),
                    layer_scale_schedule=getattr(self.config, "predictive_layer_scale_schedule", "none"),
                    layer_scale_min=float(getattr(self.config, "predictive_layer_scale_min", 1.0)),
                )
                effective_logits = logits + predicted_delta_logits
                applied_delta_logits = predicted_delta_logits
                PredictiveRouterReplayState.record_predictive_bias_stats(
                    predictive_state.layer_idx,
                    applied_delta_logits,
                    logits.detach(),
                )
                routing_replay_manager.record_predictive_bias(
                    applied_delta_logits,
                    predictive_state.layer_idx,
                )
            return self.routing(effective_logits)

        if predictive_action != RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS:
            raise ValueError(f"Unsupported predictive router action: {predictive_action}")

        old_inputs, old_logits, valid_mask, predictive_loss_scale = predictive_state.get_predictive_data()
        if predictive_state.has_valid_predictive_data():
            old_inputs = old_inputs.to(
                device=input.device,
                dtype=input.dtype,
                non_blocking=old_inputs.device.type == "cpu",
            )
            old_logits = old_logits.to(
                device=logits.device,
                dtype=logits.dtype,
                non_blocking=old_logits.device.type == "cpu",
            )
            current_inputs = input
            current_logits = logits
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=input.device, non_blocking=valid_mask.device.type == "cpu")
                current_inputs = input[valid_mask]
                current_logits = logits[valid_mask]
            current_inputs = current_inputs.detach()
            old_logits = old_logits.detach()
            current_logits = current_logits.detach()
            with torch.enable_grad():
                raw_predicted_delta_logits = bias_predictor(old_inputs.detach())
                predicted_delta_logits, stabilizer_metrics = stabilize_predictive_delta_logits(
                    predicted_delta_logits=raw_predicted_delta_logits,
                    reference_logits=old_logits.detach(),
                    layer_idx=predictive_state.layer_idx,
                    num_layers=len(predictive_controller.router_states),
                    layer_scale_schedule=getattr(self.config, "predictive_layer_scale_schedule", "none"),
                    layer_scale_min=float(getattr(self.config, "predictive_layer_scale_min", 1.0)),
                )
                effective_logits = old_logits + predicted_delta_logits
                applied_delta_logits = predicted_delta_logits
                boundary_loss_weights, boundary_loss_metrics = build_topk_boundary_loss_weights(
                    old_logits=old_logits.detach(),
                    topk=self.topk,
                    max_boundary_loss_weight=getattr(self.config, "predictive_boundary_loss_max_weight", None),
                    min_boundary_margin=float(getattr(self.config, "predictive_boundary_loss_min_margin", 1e-4)),
                )
                sample_weights = boundary_loss_weights
                predictive_loss = compute_predictive_loss(
                    old_logits=old_logits,
                    current_logits=current_logits,
                    predicted_delta_logits=predicted_delta_logits,
                    loss_type=self.config.bias_predictor_loss_type,
                    sample_weights=sample_weights,
                )
                predictive_loss = predictive_loss * predictive_loss_scale
            PredictiveRouterReplayState.record_predictive_loss(
                predictive_state.layer_idx,
                predictive_loss.item(),
                current_logits.shape[0],
            )
            PredictiveRouterReplayState.record_predictive_topk_accuracy(
                predictive_state.layer_idx,
                calculate_topk_accuracy(
                    topk=self.topk,
                    logits1=effective_logits.detach(),
                    logits2=current_logits,
                ),
                current_logits.shape[0],
            )
            for metric_name, metric_value in stabilizer_metrics.items():
                PredictiveRouterReplayState.record_predictive_scalar_metric(
                    metric_name,
                    predictive_state.layer_idx,
                    metric_value,
                    current_logits.shape[0],
                )
            for metric_name, metric_value in boundary_loss_metrics.items():
                PredictiveRouterReplayState.record_predictive_scalar_metric(
                    metric_name,
                    predictive_state.layer_idx,
                    metric_value,
                    current_logits.shape[0],
                )
            PredictiveRouterReplayState.record_predictive_metric_tensors(
                layer_idx=predictive_state.layer_idx,
                old_inputs=old_inputs,
                current_inputs=current_inputs,
                old_logits=old_logits,
                current_logits=current_logits,
                predicted_delta_logits=applied_delta_logits.detach(),
            )
        else:
            with torch.enable_grad():
                predictive_loss = build_synthetic_predictive_loss(bias_predictor=bias_predictor, input_tensor=input)

        probs, routing_map = self.routing(logits)
        predictive_loss.backward()
        predictive_state.clear_predictive_data()
        return probs, routing_map

    TopKRouter.forward = patched_forward
    TopKRouter._predictive_router_replay_patched = True


@contextmanager
def predictive_action_scope(action: RouterPredictiveAction):
    get_predictive_replay_controller().set_global_predictive_action(action)
    try:
        yield
    finally:
        get_predictive_replay_controller().clear_global_predictive_action()


def _is_predictive_param_group(param_group: dict) -> bool:
    return any(getattr(param, "is_bias_predictor", False) for param in param_group.get("params", []))


def _iter_predictive_param_groups(optimizer):
    matched_groups = [param_group for param_group in optimizer.param_groups if _is_predictive_param_group(param_group)]
    if matched_groups:
        for param_group in matched_groups:
            yield param_group
        return

    # Megatron distributed optimizer may replace the original model/main params inside
    # optimizer.param_groups with sharded tensors that no longer carry the custom
    # is_bias_predictor attribute. Fall back to the predictor group's distinctive LR:
    # the predictor group's LR is `base_lr * args.bias_predictor_lr_mult`, and lr_mult
    # is paper-sweep 5e1–1e4, so half the multiplier is a reliable lower bound that
    # excludes the base param group(s) without depending on the exact lr_mult value.
    try:
        from megatron.training.global_vars import get_args

        lr_mult = float(getattr(get_args(), "bias_predictor_lr_mult", 0.0) or 0.0)
    except Exception:
        lr_mult = 0.0
    lr_mult_threshold = max(lr_mult * 0.5, 10.0)

    positive_group_lrs = [
        float(param_group.get("max_lr", param_group.get("lr", 0.0)))
        for param_group in optimizer.param_groups
        if float(param_group.get("max_lr", param_group.get("lr", 0.0))) > 0.0
    ]
    if not positive_group_lrs:
        return

    base_group_lr = min(positive_group_lrs)
    fallback_threshold = base_group_lr * lr_mult_threshold
    for param_group in optimizer.param_groups:
        group_max_lr = float(param_group.get("max_lr", param_group.get("lr", 0.0)))
        if group_max_lr >= fallback_threshold:
            yield param_group


def clear_predictive_optimizer_grads(optimizer) -> None:
    for param_group in _iter_predictive_param_groups(optimizer):
        for param in param_group["params"]:
            param.grad = None
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()
            main_param = getattr(param, "main_param", None)
            if main_param is not None and main_param.grad is not None:
                main_param.grad = None


def disable_predictive_param_groups(optimizer) -> list[dict[str, object]]:
    saved_group_states = []
    for param_group in _iter_predictive_param_groups(optimizer):
        saved_state = {"group": param_group}
        if "lr" in param_group:
            saved_state["lr"] = param_group["lr"]
            param_group["lr"] = 0.0
        if "weight_decay" in param_group:
            saved_state["weight_decay"] = param_group["weight_decay"]
            param_group["weight_decay"] = 0.0
        saved_group_states.append(saved_state)
    return saved_group_states


def restore_predictive_param_groups(saved_group_states: list[dict[str, object]]) -> None:
    for saved_state in saved_group_states:
        param_group = saved_state["group"]
        if "lr" in saved_state:
            param_group["lr"] = saved_state["lr"]
        if "weight_decay" in saved_state:
            param_group["weight_decay"] = saved_state["weight_decay"]


class PredictiveReplayController:
    def __init__(self) -> None:
        self.router_states: list["PredictiveRouterReplayState"] = []
        self.current_action = RouterPredictiveAction.DISABLED
        self.current_rollout_id: int | None = None
        self.current_step_id: int | None = None
        self.microbatches: list[object] = []
        self.train_index = 0
        self.used_valid_predictive_data = False
        self.predictive_loss_tracker: dict[int, tuple[float, int]] = {}
        self.predictive_bias_ratio_tracker: dict[int, tuple[float, float]] = {}
        self.predictive_topk_accuracy_tracker: dict[int, tuple[float, int]] = {}
        self.predictive_scalar_metric_trackers: dict[str, dict[int, tuple[float, int]]] = {}
        self.predictive_metric_tensor_capture_enabled = False
        self.predictive_metric_tensor_cache: dict[str, list[tuple[int, torch.Tensor]]] = {}
        self.clear_predictive_metric_tensors()

    def set_current_step_context(self, *, rollout_id: int | None, step_id: int | None) -> None:
        self.current_rollout_id = None if rollout_id is None else int(rollout_id)
        self.current_step_id = None if step_id is None else int(step_id)

    def clear_current_step_context(self) -> None:
        self.current_rollout_id = None
        self.current_step_id = None

    def get_current_rollout_id(self) -> int | None:
        return self.current_rollout_id

    def register_router(self, router, attr_name: str = "predictive_router_replay") -> "PredictiveRouterReplayState":
        state = PredictiveRouterReplayState(controller=self)
        setattr(router, attr_name, state)
        return state

    def reset_registry(self) -> None:
        self.clear_global_predictive_action()
        self.clear_global_predictive_data()
        self.clear_microbatch_buffer()
        self.reset_train_step_usage()
        self.router_states.clear()
        self.clear_predictive_metrics()
        self.disable_predictive_metric_tensor_capture()
        self.clear_predictive_metric_tensors()
        self.clear_current_step_context()

    def add_router_state(self, state: "PredictiveRouterReplayState") -> None:
        self.router_states.append(state)

    def get_router_states(self) -> list["PredictiveRouterReplayState"]:
        return list(self.router_states)

    def has_registered_routers(self) -> bool:
        return bool(self.router_states)

    def set_global_predictive_action(self, action: RouterPredictiveAction) -> None:
        self.current_action = action
        for router in self.router_states:
            router.set_predictive_action(action)

    def clear_global_predictive_action(self) -> None:
        self.current_action = RouterPredictiveAction.DISABLED
        for router in self.router_states:
            router.clear_predictive_action()

    def get_global_predictive_action(self) -> RouterPredictiveAction:
        return self.current_action

    def clear_global_predictive_data(self) -> None:
        for router in self.router_states:
            router.clear_predictive_data()

    def set_global_predictive_data(
        self,
        *,
        old_inputs_concat: torch.Tensor | None,
        old_logits_concat: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        loss_scale: float = 1.0,
    ) -> None:
        if old_inputs_concat is None or old_logits_concat is None:
            self.clear_global_predictive_data()
            return

        if old_inputs_concat.ndim != 3 or old_logits_concat.ndim != 3:
            raise ValueError("Predictive tensors must have shape [num_tokens, num_layers, hidden_or_experts].")
        if old_inputs_concat.shape[:2] != old_logits_concat.shape[:2]:
            raise ValueError(
                f"Predictive tensor shape mismatch: inputs={old_inputs_concat.shape}, logits={old_logits_concat.shape}"
            )
        if old_inputs_concat.shape[1] != len(self.router_states):
            raise ValueError(
                f"Predictive tensor layer count {old_inputs_concat.shape[1]} does not match "
                f"registered routers {len(self.router_states)}."
            )

        for layer_idx, router in enumerate(self.router_states):
            router.set_predictive_data(
                inputs=old_inputs_concat[:, layer_idx : layer_idx + 1, :],
                logits=old_logits_concat[:, layer_idx : layer_idx + 1, :],
                valid_mask=valid_mask,
                loss_scale=loss_scale,
            )

    def clear_microbatch_buffer(self) -> None:
        self.microbatches.clear()
        self.train_index = 0

    def append_microbatch(self, microbatch_data) -> None:
        self.microbatches.append(microbatch_data)

    def reset_microbatch_cursor(self) -> None:
        self.train_index = 0

    def pop_next_microbatch(self):
        if self.train_index >= len(self.microbatches):
            raise IndexError(
                f"Predictive replay buffer underflow: train_index={self.train_index}, buffered={len(self.microbatches)}"
            )
        microbatch_data = self.microbatches[self.train_index]
        self.train_index += 1
        return microbatch_data

    def buffered_microbatch_count(self) -> int:
        return len(self.microbatches)

    def remaining_microbatch_count(self) -> int:
        return len(self.microbatches) - self.train_index

    def reset_train_step_usage(self) -> None:
        self.used_valid_predictive_data = False

    def mark_train_step_used(self) -> None:
        self.used_valid_predictive_data = True

    def apply_predictive_train_mode(self, predictive_train_mode: str, *, consume_microbatch: bool = False) -> None:
        if predictive_train_mode == "compute":
            predictive_microbatch = self.pop_next_microbatch()
            if predictive_microbatch.has_valid_samples:
                self.set_global_predictive_data(
                    old_inputs_concat=predictive_microbatch.old_inputs_concat,
                    old_logits_concat=predictive_microbatch.old_logits_concat,
                    valid_mask=predictive_microbatch.valid_mask,
                    loss_scale=predictive_microbatch.predictive_loss_scale,
                )
                self.set_global_predictive_action(RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS)
                self.mark_train_step_used()
                return

            # Align with VERL's all_none branch: keep the compute action active even when this
            # rank has no local predictive samples, so the router patch builds a synthetic
            # zero-loss through bias_predictor for collective synchronization.
            self.clear_global_predictive_data()
            self.set_global_predictive_action(RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS)
            self.mark_train_step_used()
            return

        if predictive_train_mode == "skip":
            if consume_microbatch:
                self.pop_next_microbatch()
            self.clear_global_predictive_data()
            self.set_global_predictive_action(RouterPredictiveAction.SKIP_PREDICTIVE)
            return

        raise ValueError(f"Unsupported predictive_train_mode: {predictive_train_mode}")

    def clear_predictive_metrics(self) -> None:
        self.predictive_loss_tracker.clear()
        self.predictive_bias_ratio_tracker.clear()
        self.predictive_topk_accuracy_tracker.clear()
        self.predictive_scalar_metric_trackers.clear()

    def record_predictive_loss(self, layer_idx: int, loss_value: float, token_count: int = 1) -> None:
        weighted_sum, total_count = self.predictive_loss_tracker.get(layer_idx, (0.0, 0))
        self.predictive_loss_tracker[layer_idx] = (
            weighted_sum + float(loss_value) * int(token_count),
            total_count + int(token_count),
        )

    def record_predictive_bias_stats(
        self,
        layer_idx: int,
        predicted_delta_logits: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> None:
        numerator = torch.abs(predicted_delta_logits).sum().item()
        denominator = torch.abs(reference_logits).sum().item() + 1e-10
        prev_num, prev_den = self.predictive_bias_ratio_tracker.get(layer_idx, (0.0, 0.0))
        self.predictive_bias_ratio_tracker[layer_idx] = (prev_num + numerator, prev_den + denominator)

    def record_predictive_bias_ratio(self, layer_idx: int, ratio_value: float) -> None:
        prev_num, prev_den = self.predictive_bias_ratio_tracker.get(layer_idx, (0.0, 0.0))
        self.predictive_bias_ratio_tracker[layer_idx] = (prev_num + float(ratio_value), prev_den + 1.0)

    def record_predictive_topk_accuracy(self, layer_idx: int, accuracy_value: float, token_count: int = 1) -> None:
        weighted_sum, total_count = self.predictive_topk_accuracy_tracker.get(layer_idx, (0.0, 0))
        self.predictive_topk_accuracy_tracker[layer_idx] = (
            weighted_sum + float(accuracy_value) * int(token_count),
            total_count + int(token_count),
        )

    def record_predictive_scalar_metric(
        self,
        metric_name: str,
        layer_idx: int,
        metric_value: float,
        token_count: int = 1,
    ) -> None:
        tracker = self.predictive_scalar_metric_trackers.setdefault(metric_name, {})
        weighted_sum, total_count = tracker.get(layer_idx, (0.0, 0))
        tracker[layer_idx] = (
            weighted_sum + float(metric_value) * int(token_count),
            total_count + int(token_count),
        )

    def enable_predictive_metric_tensor_capture(self) -> None:
        self.predictive_metric_tensor_capture_enabled = True
        self.clear_predictive_metric_tensors()

    def disable_predictive_metric_tensor_capture(self) -> None:
        self.predictive_metric_tensor_capture_enabled = False

    def clear_predictive_metric_tensors(self) -> None:
        self.predictive_metric_tensor_cache = {
            "old_inputs": [],
            "current_inputs": [],
            "old_logits": [],
            "current_logits": [],
            "predicted_delta_logits": [],
        }

    def record_predictive_metric_tensors(
        self,
        *,
        layer_idx: int,
        old_inputs: torch.Tensor,
        current_inputs: torch.Tensor,
        old_logits: torch.Tensor,
        current_logits: torch.Tensor,
        predicted_delta_logits: torch.Tensor,
    ) -> None:
        if not self.predictive_metric_tensor_capture_enabled:
            return
        self.predictive_metric_tensor_cache["old_inputs"].append((layer_idx, old_inputs.detach().cpu().contiguous()))
        self.predictive_metric_tensor_cache["current_inputs"].append(
            (layer_idx, current_inputs.detach().cpu().contiguous())
        )
        self.predictive_metric_tensor_cache["old_logits"].append((layer_idx, old_logits.detach().cpu().contiguous()))
        self.predictive_metric_tensor_cache["current_logits"].append(
            (layer_idx, current_logits.detach().cpu().contiguous())
        )
        self.predictive_metric_tensor_cache["predicted_delta_logits"].append(
            (layer_idx, predicted_delta_logits.detach().cpu().contiguous())
        )

    def get_and_clear_predictive_metric_tensors(self) -> dict[str, list[tuple[int, torch.Tensor]]]:
        tensor_cache = {
            key: [(layer_idx, tensor) for layer_idx, tensor in values]
            for key, values in self.predictive_metric_tensor_cache.items()
        }
        self.clear_predictive_metric_tensors()
        return tensor_cache

    @staticmethod
    def _get_data_parallel_group():
        if not dist.is_initialized():
            return None
        try:
            from megatron.core import parallel_state as mpu

            return mpu.get_data_parallel_group(with_context_parallel=False)
        except Exception:
            return None

    def _all_reduce_weighted_metric_tracker(
        self,
        tracker: dict[int, tuple[float, int]],
    ) -> dict[str, float]:
        if not tracker:
            return {}

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        layer_indices = sorted(tracker)
        local_stats = torch.tensor(
            [[tracker[layer_idx][0], float(tracker[layer_idx][1])] for layer_idx in layer_indices],
            dtype=torch.float64,
            device=device,
        )
        dp_group = self._get_data_parallel_group()
        if dp_group is not None:
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM, group=dp_group)
        return {
            str(layer_idx): float(weighted_sum / max(total_count, 1.0))
            for layer_idx, (weighted_sum, total_count) in zip(layer_indices, local_stats.tolist(), strict=True)
        }

    def _all_reduce_ratio_metric_tracker(
        self,
        tracker: dict[int, tuple[float, float]],
    ) -> dict[str, float]:
        if not tracker:
            return {}

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        layer_indices = sorted(tracker)
        local_stats = torch.tensor(
            [[tracker[layer_idx][0], tracker[layer_idx][1]] for layer_idx in layer_indices],
            dtype=torch.float64,
            device=device,
        )
        dp_group = self._get_data_parallel_group()
        if dp_group is not None:
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM, group=dp_group)
        return {
            str(layer_idx): float(numerator / max(denominator, 1e-10))
            for layer_idx, (numerator, denominator) in zip(layer_indices, local_stats.tolist(), strict=True)
        }

    def get_and_clear_predictive_metrics_with_details(self) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        metrics = {}
        details: dict[str, dict[str, float]] = {}

        if self.predictive_loss_tracker:
            details["predictive_loss"] = self._all_reduce_weighted_metric_tracker(self.predictive_loss_tracker)
            metrics["predictive_loss"] = sum(details["predictive_loss"].values()) / len(details["predictive_loss"])

        if self.predictive_bias_ratio_tracker:
            details["predictive_bias_to_logits_ratio"] = self._all_reduce_ratio_metric_tracker(
                self.predictive_bias_ratio_tracker
            )
            metrics["predictive_bias_to_logits_ratio"] = sum(details["predictive_bias_to_logits_ratio"].values()) / len(
                details["predictive_bias_to_logits_ratio"]
            )

        if self.predictive_topk_accuracy_tracker:
            details["predictive_topk_accuracy"] = self._all_reduce_weighted_metric_tracker(
                self.predictive_topk_accuracy_tracker
            )
            metrics["predictive_topk_accuracy"] = sum(details["predictive_topk_accuracy"].values()) / len(
                details["predictive_topk_accuracy"]
            )

        for metric_name in sorted(self.predictive_scalar_metric_trackers):
            tracker = self.predictive_scalar_metric_trackers[metric_name]
            if not tracker:
                continue
            details[metric_name] = self._all_reduce_weighted_metric_tracker(tracker)
            metrics[metric_name] = sum(details[metric_name].values()) / len(details[metric_name])

        self.clear_predictive_metrics()
        return metrics, details

    def get_and_clear_predictive_metrics(self) -> dict[str, float]:
        metrics, _ = self.get_and_clear_predictive_metrics_with_details()
        return metrics


_PREDICTIVE_REPLAY_CONTROLLER = PredictiveReplayController()


def get_predictive_replay_controller() -> PredictiveReplayController:
    return _PREDICTIVE_REPLAY_CONTROLLER


class PredictiveRouterReplayState:
    def __init__(
        self,
        layer_idx: int | None = None,
        controller: PredictiveReplayController | None = None,
    ):
        self.controller = controller or get_predictive_replay_controller()
        self.layer_idx = len(self.controller.router_states) if layer_idx is None else layer_idx
        self.predictive_action = RouterPredictiveAction.DISABLED
        self.recorded_old_inputs: torch.Tensor | None = None
        self.recorded_old_logits: torch.Tensor | None = None
        self.predictive_valid_mask: torch.Tensor | None = None
        self.predictive_loss_scale: float = 1.0
        self.controller.add_router_state(self)

    @staticmethod
    def _squeeze_router_dim(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim >= 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor

    @classmethod
    def register_router(cls, router, attr_name: str = "predictive_router_replay") -> "PredictiveRouterReplayState":
        return get_predictive_replay_controller().register_router(router, attr_name=attr_name)

    @classmethod
    def reset_registry(cls) -> None:
        get_predictive_replay_controller().reset_registry()

    @classmethod
    def get_router_instances(cls) -> list["PredictiveRouterReplayState"]:
        return get_predictive_replay_controller().get_router_states()

    def has_valid_predictive_data(self) -> bool:
        return (
            self.recorded_old_inputs is not None
            and self.recorded_old_logits is not None
            and self.recorded_old_inputs.shape[0] > 0
            and self.recorded_old_logits.shape[0] > 0
        )

    def record_predictive_data(self, inputs: torch.Tensor, logits: torch.Tensor) -> None:
        self.recorded_old_inputs = self._squeeze_router_dim(inputs).detach().contiguous()
        self.recorded_old_logits = self._squeeze_router_dim(logits).detach().contiguous()
        self.predictive_valid_mask = None
        self.predictive_loss_scale = 1.0

    def set_predictive_data(
        self,
        *,
        inputs: torch.Tensor | None,
        logits: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
        loss_scale: float = 1.0,
    ) -> None:
        self.recorded_old_inputs = inputs.detach().contiguous() if inputs is not None else None
        self.recorded_old_logits = logits.detach().contiguous() if logits is not None else None
        self.predictive_valid_mask = valid_mask.detach().contiguous() if valid_mask is not None else None
        self.predictive_loss_scale = float(loss_scale)

    def get_predictive_data(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float]:
        return self.recorded_old_inputs, self.recorded_old_logits, self.predictive_valid_mask, self.predictive_loss_scale

    def clear_predictive_data(self) -> None:
        self.recorded_old_inputs = None
        self.recorded_old_logits = None
        self.predictive_valid_mask = None
        self.predictive_loss_scale = 1.0

    def set_predictive_action(self, action: RouterPredictiveAction) -> None:
        self.predictive_action = action

    def clear_predictive_action(self) -> None:
        self.predictive_action = RouterPredictiveAction.DISABLED

    @classmethod
    def set_global_predictive_action(cls, action: RouterPredictiveAction) -> None:
        get_predictive_replay_controller().set_global_predictive_action(action)

    @classmethod
    def clear_global_predictive_action(cls) -> None:
        get_predictive_replay_controller().clear_global_predictive_action()

    @classmethod
    def get_global_predictive_action(cls) -> RouterPredictiveAction:
        return get_predictive_replay_controller().get_global_predictive_action()

    @classmethod
    def clear_global_predictive_data(cls) -> None:
        get_predictive_replay_controller().clear_global_predictive_data()

    @classmethod
    def set_global_predictive_data(
        cls,
        *,
        old_inputs_concat: torch.Tensor | None,
        old_logits_concat: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        loss_scale: float = 1.0,
    ) -> None:
        get_predictive_replay_controller().set_global_predictive_data(
            old_inputs_concat=old_inputs_concat,
            old_logits_concat=old_logits_concat,
            valid_mask=valid_mask,
            loss_scale=loss_scale,
        )

    @classmethod
    def clear_predictive_metrics(cls) -> None:
        get_predictive_replay_controller().clear_predictive_metrics()

    @classmethod
    def record_predictive_loss(cls, layer_idx: int, loss_value: float, token_count: int = 1) -> None:
        get_predictive_replay_controller().record_predictive_loss(layer_idx, loss_value, token_count)

    @classmethod
    def record_predictive_bias_stats(
        cls,
        layer_idx: int,
        predicted_delta_logits: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> None:
        get_predictive_replay_controller().record_predictive_bias_stats(
            layer_idx,
            predicted_delta_logits,
            reference_logits,
        )

    @classmethod
    def record_predictive_bias_ratio(cls, layer_idx: int, ratio_value: float) -> None:
        get_predictive_replay_controller().record_predictive_bias_ratio(layer_idx, ratio_value)

    @classmethod
    def record_predictive_topk_accuracy(cls, layer_idx: int, accuracy_value: float, token_count: int = 1) -> None:
        get_predictive_replay_controller().record_predictive_topk_accuracy(layer_idx, accuracy_value, token_count)

    @classmethod
    def record_predictive_scalar_metric(
        cls,
        metric_name: str,
        layer_idx: int,
        metric_value: float,
        token_count: int = 1,
    ) -> None:
        get_predictive_replay_controller().record_predictive_scalar_metric(
            metric_name,
            layer_idx,
            metric_value,
            token_count,
        )

    @classmethod
    def enable_predictive_metric_tensor_capture(cls) -> None:
        get_predictive_replay_controller().enable_predictive_metric_tensor_capture()

    @classmethod
    def disable_predictive_metric_tensor_capture(cls) -> None:
        get_predictive_replay_controller().disable_predictive_metric_tensor_capture()

    @classmethod
    def clear_predictive_metric_tensors(cls) -> None:
        get_predictive_replay_controller().clear_predictive_metric_tensors()

    @classmethod
    def record_predictive_metric_tensors(
        cls,
        *,
        layer_idx: int,
        old_inputs: torch.Tensor,
        current_inputs: torch.Tensor,
        old_logits: torch.Tensor,
        current_logits: torch.Tensor,
        predicted_delta_logits: torch.Tensor,
    ) -> None:
        get_predictive_replay_controller().record_predictive_metric_tensors(
            layer_idx=layer_idx,
            old_inputs=old_inputs,
            current_inputs=current_inputs,
            old_logits=old_logits,
            current_logits=current_logits,
            predicted_delta_logits=predicted_delta_logits,
        )

    @classmethod
    def get_and_clear_predictive_metric_tensors(cls) -> dict[str, list[tuple[int, torch.Tensor]]]:
        return get_predictive_replay_controller().get_and_clear_predictive_metric_tensors()

    @classmethod
    def get_and_clear_predictive_metrics_with_details(cls) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        return get_predictive_replay_controller().get_and_clear_predictive_metrics_with_details()

    @classmethod
    def get_and_clear_predictive_metrics(cls) -> dict[str, float]:
        return get_predictive_replay_controller().get_and_clear_predictive_metrics()


