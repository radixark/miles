import copy
import dataclasses
import gc
import inspect
import logging
import math
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
try:
    from megatron.core.optimizer import OptimizerConfig, ParamKey, get_megatron_optimizer
except ImportError:
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    ParamKey = None
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

from miles.utils.memory_utils import clear_memory
from miles.utils.replay_base import (
    RouterLogitsCacheAction,
    apply_routing_replay_patch,
    register_routing_replay_modules,
    routing_replay_manager,
)

from ..training_utils.ci_utils import check_grad_norm, check_kl
from ..training_utils.data import DataIterator, get_batch
from ..training_utils.log_utils import aggregate_forward_results, aggregate_train_losses, log_train_step
from ..training_utils.loss import loss_function
from ..training_utils.parallel import ParallelState
from .checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_with_lora
from .ci_utils import (
    check_model_hashes,
    check_peak_gpu_memory_after_load,
    compute_model_hashes_by_layer,
    save_model_hashes,
)
from .initialize import is_megatron_main_rank
from .lora_utils import is_lora_enabled, is_lora_model
from .model_provider import get_model_provider_func
from .parallel import get_packed_seq_params
from .predictive_router_replay import (
    collect_predictive_param_stats,
    PredictiveRouterReplayState,
    RouterPredictiveAction,
    apply_predictive_router_replay_patch,
    clear_predictive_optimizer_grads,
    disable_predictive_param_groups,
    get_predictive_replay_controller,
    predictive_debug_param_stats_enabled,
    restore_predictive_param_groups,
)
from .predictive_train_schedule import get_predictive_train_mode_for_step
from .predictive_router_utils import pack_recorded_predictive_microbatch
from .router_replay_saver import RouterReplayLogitsSaver

logger = logging.getLogger(__name__)


def _maybe_log_predictive_param_stats(
    *,
    stage: str,
    model: Sequence[DDP],
    rollout_id: int,
    step_id: int,
    predictive_train_mode: str,
) -> None:
    if not predictive_debug_param_stats_enabled():
        return
    if not is_megatron_main_rank():
        return
    stats = collect_predictive_param_stats(model)
    logger.info(
        "[Predictive Routing Replay][debug] stage=%s rollout=%s step=%s mode=%s stats=%s",
        stage,
        rollout_id,
        step_id,
        predictive_train_mode,
        stats,
    )


def _validate_predictive_main_grads(*, args: Namespace, role: str, model: Sequence[DDP]) -> None:
    if role != "actor" or not getattr(args, "enable_predictive_routing_replay", False):
        return
    if not getattr(args, "use_distributed_optimizer", False):
        return

    stats = collect_predictive_param_stats(model)
    if stats["num_predictor_params"] == 0:
        raise RuntimeError(
            "Predictive routing replay is enabled but no bias_predictor parameters were registered on the actor model."
        )
    if stats["num_predictor_params_with_main_grad"] != stats["num_predictor_params"]:
        raise RuntimeError(
            "Predictive routing replay bias_predictor parameters are missing main_grad after DDP construction. "
            "This means they were not integrated into Megatron distributed optimizer state and will not train. "
            f"stats={stats}"
        )


from .bridge_lora_helpers import _ensure_model_list, _setup_lora_model_via_bridge  # noqa: F401
from .lora_utils import save_lora_checkpoint


def _build_optimizer_config_overrides(args: Namespace, config: OptimizerConfig, role: str) -> dict | None:
    if role != "actor" or not getattr(args, "enable_predictive_routing_replay", False):
        return None
    if ParamKey is None:
        raise RuntimeError("Predictive routing replay requires megatron.core.optimizer.ParamKey support.")

    bias_predictor_optim_config = copy.deepcopy(config)
    bias_predictor_optim_config.lr = config.lr * args.bias_predictor_lr_mult
    logger.info(
        "[Predictive Routing Replay] Bias predictor optimizer override enabled: base_lr=%s lr_mult=%s predictor_lr=%s",
        config.lr,
        args.bias_predictor_lr_mult,
        bias_predictor_optim_config.lr,
    )
    return {ParamKey(attr="is_bias_predictor"): bias_predictor_optim_config}


def get_optimizer_param_scheduler(
    args: Namespace,
    optimizer: MegatronOptimizer,
    role: str = "actor",
) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


# ---------------------------------------------------------------------------
# Model + Optimizer setup
# ---------------------------------------------------------------------------


def _is_muon_optimizer(optimizer: str | None) -> bool:
    return optimizer is not None and "muon" in optimizer.lower()


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    apply_routing_replay_patch()

    if role == "actor" and getattr(args, "enable_predictive_routing_replay", False):
        apply_predictive_router_replay_patch()

    if is_lora_enabled(args) and role == "actor" and args.megatron_to_hf_mode == "bridge":
        model = _setup_lora_model_via_bridge(args)
    else:
        model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    registered_router_modules = register_routing_replay_modules(model)
    if registered_router_modules > 0:
        logger.info("Registered routing replay state on %s TopKRouter modules", registered_router_modules)
    _validate_predictive_main_grads(args=args, role=role, model=model)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    if _is_muon_optimizer(config.optimizer):
        optimizer = get_megatron_muon_optimizer(
            config=config,
            model_chunks=model,
            use_gloo_process_groups=args.enable_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in config.optimizer.lower(),
        )
    else:
        optimizer_kwargs = {
            "config": config,
            "model_chunks": model,
            "use_gloo_process_groups": args.enable_gloo_process_groups,
        }
        config_overrides = _build_optimizer_config_overrides(args, config, role)
        if config_overrides is not None:
            optimizer_signature = inspect.signature(get_megatron_optimizer)
            if "config_overrides" not in optimizer_signature.parameters:
                raise RuntimeError(
                    "Predictive routing replay requires Megatron get_megatron_optimizer to support config_overrides."
                )
            optimizer_kwargs["config_overrides"] = config_overrides
        optimizer = get_megatron_optimizer(**optimizer_kwargs)
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer, role=role)
    return model, optimizer, opt_param_scheduler


# ---------------------------------------------------------------------------
# Forward pre-hook helpers
# ---------------------------------------------------------------------------


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def _predictive_router_enabled(args: Namespace, model: Sequence[DDP]) -> bool:
    return (
        getattr(args, "enable_predictive_routing_replay", False)
        and getattr(model[0], "role", "actor") == "actor"
        and get_predictive_replay_controller().has_registered_routers()
    )


def _collect_recorded_predictive_microbatch(
    args: Namespace,
    parallel_state: ParallelState,
    batch: dict[str, torch.Tensor | list[torch.Tensor] | None],
) -> None:
    predictive_controller = get_predictive_replay_controller()
    router_states = predictive_controller.get_router_states()
    if not router_states:
        return

    recorded_old_inputs = []
    recorded_old_logits = []
    for router_state in router_states:
        old_inputs, old_logits, _, _ = router_state.get_predictive_data()
        if old_inputs is None or old_logits is None:
            raise RuntimeError("Predictive RECORD mode did not capture router inputs/logits on every local router.")
        recorded_old_inputs.append(old_inputs)
        recorded_old_logits.append(old_logits)
        router_state.clear_predictive_data()

    predictive_controller.append_microbatch(
        pack_recorded_predictive_microbatch(
            recorded_old_inputs=recorded_old_inputs,
            recorded_old_logits=recorded_old_logits,
            total_lengths=batch["total_lengths"],
            parallel_state=parallel_state,
            qkv_format=args.qkv_format,
            max_seq_lens=batch.get("max_seq_lens", None),
            allgather_cp=args.allgather_cp,
            downsample_batch_size=getattr(args, "predictive_downsample_batch_size", None),
            max_len_limit=getattr(args, "predictive_downsample_max_len_limit", None),
            storage_dtype=getattr(args, "predictive_storage_dtype", "fp32"),
        )
    )


def _apply_predictive_train_mode(
    predictive_train_mode: str,
    *,
    consume_predictive_microbatch: bool = False,
) -> None:
    get_predictive_replay_controller().apply_predictive_train_mode(
        predictive_train_mode,
        consume_microbatch=consume_predictive_microbatch,
    )


def _maybe_record_global_token_ids(batch: dict[str, torch.Tensor | list[torch.Tensor] | None]) -> None:
    global_token_ids = batch.get("global_token_ids")
    if global_token_ids is None:
        routing_replay_manager.record_global_token_ids()
        return

    valid_ids = global_token_ids[global_token_ids >= 0]
    routing_replay_manager.record_global_token_ids(valid_ids)


def _record_router_weights(model: Sequence[DDP]) -> None:
    from megatron.core.transformer.moe.router import TopKRouter

    for model_chunk in model:
        module = getattr(model_chunk, "module", model_chunk)
        for layer in module.modules():
            if not isinstance(layer, TopKRouter):
                continue
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                raise RuntimeError("TopKRouter.layer_number is required for Verl-aligned router_weights saving.")
            if not hasattr(layer, "weight"):
                raise RuntimeError("TopKRouter.weight is required for Verl-aligned router_weights saving.")
            routing_replay_manager.logits_cache["router_weights"][layer_number] = layer.weight.detach().cpu().contiguous()


def _save_router_logits_cache(
    *,
    router_logits_saver: RouterReplayLogitsSaver | None,
    step_name: str,
) -> None:
    if router_logits_saver is None:
        return

    logits_cache = routing_replay_manager.get_and_clear_logits_cache()
    cache_counts = {
        "compute_log_prob": len(logits_cache["compute_log_prob"]),
        "training": len(logits_cache["training"]),
        "router_weights": len(logits_cache["router_weights"]),
        "global_token_ids": len(logits_cache["global_token_ids"]),
        "predictive_bias": len(logits_cache["predictive_bias"]),
    }
    if not logits_cache["compute_log_prob"] and not logits_cache["training"]:
        if any(cache_counts.values()):
            logger.warning("Router logits cache for %s had no logits tensors: %s", step_name, cache_counts)
        return

    if mpu.get_tensor_model_parallel_world_size() > 1:
        logits_cache = RouterReplayLogitsSaver.gather_logits_from_tp_group(logits_cache)

    if mpu.get_tensor_model_parallel_rank() == 0:
        if mpu.get_data_parallel_world_size() > 1:
            logits_cache = RouterReplayLogitsSaver.gather_logits_from_dp_group(
                logits_cache,
                max_tokens=router_logits_saver.max_tokens,
            )
        if mpu.get_data_parallel_rank() == 0:
            logger.info("Saving router logits cache for %s with counts=%s", step_name, cache_counts)
            router_logits_saver.save_logits_async(logits_cache, step_name)


# ---------------------------------------------------------------------------
# Forward-only inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    parallel_state: ParallelState,
    store_prefix: str = "",
    record_router_logits: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f: Post-forward callback used to compute and package outputs to collect.
        args: Runtime arguments.
        model: Sequence of DDP-wrapped model chunks.
        data_iterator: Iterable(s) yielding batches for inference.
        num_microbatches: Number of microbatches per rollout step.
        store_prefix: Prefix to prepend to stored output keys.

    Returns:
        Aggregated outputs keyed by ``store_prefix + key``.
    """
    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])
    predictive_router_enabled = _predictive_router_enabled(args, model)

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch_keys = [
            "tokens",
            "loss_masks",
            "multimodal_train_inputs",
            "total_lengths",
            "response_lengths",
            "max_seq_lens",
        ]
        if record_router_logits:
            batch_keys.append("global_token_ids")
            batch_keys.append("sample_indices")
        batch = get_batch(
            data_iterator,
            batch_keys,
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = get_packed_seq_params(batch, args)
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )
        if record_router_logits:
            _maybe_record_global_token_ids(batch)

        if (
            predictive_router_enabled
            and PredictiveRouterReplayState.get_global_predictive_action() == RouterPredictiveAction.RECORD
        ):
            _collect_recorded_predictive_microbatch(args, parallel_state, batch)

        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from miles.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        # collect_non_loss_data
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        aggregated = aggregate_forward_results(forward_data_store, data_iterator[0], args, store_prefix="")
        for key, value in aggregated.items():
            rollout_data[f"{store_prefix}{key}"] = value
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    parallel_state: ParallelState,
    predictive_train_mode: str = "compute",
    router_logits_saver: RouterReplayLogitsSaver | None = None,
    router_logits_step_name: str | None = None,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args: Runtime arguments.
        rollout_id: Rollout identifier.
        step_id: Step index within the current rollout.
        data_iterator: Iterable(s) yielding training batches.
        model: Sequence of DDP-wrapped model chunks.
        optimizer: Optimizer instance.
        opt_param_scheduler: LR/WD scheduler.
        num_microbatches: Number of microbatches to process.

    Returns:
        Reduced loss dictionary (last stage only) and gradient norm for logging.
    """
    args = get_args()
    predictive_router_enabled = _predictive_router_enabled(args, model)
    if predictive_router_enabled:
        get_predictive_replay_controller().reset_train_step_usage()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from miles.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch_keys = [
            "tokens",
            "multimodal_train_inputs",
            "packed_seq_params",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "log_probs",
            "ref_log_probs",
            "values",
            "advantages",
            "returns",
            "rollout_log_probs",
            "max_seq_lens",
        ]
        if router_logits_saver is not None:
            batch_keys.append("global_token_ids")
            batch_keys.append("sample_indices")
        batch = get_batch(
            data_iterator,
            batch_keys,
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=args.allgather_cp,
        )

        from miles.utils.replay_base import all_replay_managers

        old_stages = [m.stage for m in all_replay_managers]
        for m in all_replay_managers:
            m.stage = "replay_forward"

        if predictive_router_enabled and not return_schedule_plan:
            _apply_predictive_train_mode(
                predictive_train_mode,
                consume_predictive_microbatch=predictive_train_mode == "skip",
            )

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=get_packed_seq_params(batch, args),
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": get_packed_seq_params(batch, args),
                "loss_mask": batch["full_loss_masks"],
            }

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            if (x := batch["multimodal_train_inputs"]) is not None:
                forward_kwargs.update(x)

            try:
                output_tensor = model(**forward_kwargs)
                if router_logits_saver is not None:
                    _maybe_record_global_token_ids(batch)
            finally:
                if predictive_router_enabled:
                    predictive_controller = get_predictive_replay_controller()
                    predictive_controller.clear_global_predictive_action()
                    predictive_controller.clear_global_predictive_data()

        for m, old_stage in zip(all_replay_managers, old_stages, strict=True):
            m.stage = old_stage

        return output_tensor, partial(
            loss_function, args, batch, num_microbatches, apply_megatron_loss_scaling=True
        )

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    saved_router_logits = False
    if router_logits_saver is not None:
        routing_replay_manager.set_cache_action(RouterLogitsCacheAction.TRAINING)
    try:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
        valid_step = True
        if not getattr(args, "check_for_nan_in_loss_and_grad", True):
            found_inf_flag = optimizer.prepare_grads()
            if found_inf_flag:
                valid_step = False
            else:
                grad_norm = optimizer.get_grad_norm()
                if isinstance(grad_norm, torch.Tensor):
                    valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
                else:
                    valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

        # CI check: verify only MTP parameters have non-zero gradients when truncation happens
        # This check must happen before optimizer.step() as gradients may be modified during step
        if args.ci_test and args.enable_mtp_training and args.rollout_max_response_len <= 128:
            # under response length <= 128, all outputs are truncated and loss mask is all zeros, so only MTP parameters have non-zero gradients
            from miles.backends.megatron_utils.ci_utils import check_mtp_only_grad

            check_mtp_only_grad(model, step_id)

        if valid_step:
            if predictive_router_enabled:
                _maybe_log_predictive_param_stats(
                    stage="before_optimizer_step",
                    model=model,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    predictive_train_mode=predictive_train_mode,
                )
            # Update parameters.
            disabled_predictive_groups = []
            if predictive_router_enabled and not get_predictive_replay_controller().used_valid_predictive_data:
                clear_predictive_optimizer_grads(optimizer)
                disabled_predictive_groups = disable_predictive_param_groups(optimizer)
            try:
                update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
                if router_logits_saver is not None:
                    _record_router_weights(model)

                # Update learning rate.
                assert update_successful
                opt_param_scheduler.step(increment=args.global_batch_size)
                if predictive_router_enabled:
                    _maybe_log_predictive_param_stats(
                        stage="after_optimizer_step",
                        model=model,
                        rollout_id=rollout_id,
                        step_id=step_id,
                        predictive_train_mode=predictive_train_mode,
                    )
            finally:
                if disabled_predictive_groups:
                    restore_predictive_param_groups(disabled_predictive_groups)

        # release grad
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        if router_logits_saver is not None:
            if router_logits_step_name is None:
                raise RuntimeError("router_logits_step_name must be set when router logits saving is enabled.")
            _save_router_logits_cache(
                router_logits_saver=router_logits_saver,
                step_name=router_logits_step_name,
            )
            saved_router_logits = True

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = aggregate_train_losses(losses_reduced)
            return loss_reduced, grad_norm
        return {}, grad_norm
    finally:
        if router_logits_saver is not None:
            if not saved_router_logits:
                routing_replay_manager.get_and_clear_logits_cache()
            routing_replay_manager.clear_cache_action()
        else:
            routing_replay_manager.clear_cache_action()


def finalize_model_grads_with_empty_cache(*args, **kwargs):
    # TODO: this is an ad-hoc method and we should figure out why the oom happens in the first place.
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    if free / total < 0.1:
        clear_memory()
    return finalize_model_grads(*args, **kwargs)


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    parallel_state: ParallelState,
    predictive_train_mode: str | None = None,
    router_logits_saver: RouterReplayLogitsSaver | None = None,
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
    """
    args = get_args()

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if is_megatron_main_rank():
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    role = getattr(model[0], "role", "actor")
    predictive_router_enabled = _predictive_router_enabled(args, model)
    router_topk = None
    if router_logits_saver is not None:
        for model_module in model:
            module = getattr(model_module, "module", model_module)
            for submodule in module.modules():
                if hasattr(submodule, "topk"):
                    router_topk = int(submodule.topk)
                    break
            if router_topk is not None:
                break

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):
        predictive_controller = get_predictive_replay_controller()
        # Publish current rollout/step so the stabilization layer's cross-
        # rollout anneal (--predictive-topk-margin-ratio-anneal-*) can resolve
        # its progress; without this, every annealing-related flag silently
        # no-ops because resolve_predictive_topk_margin_ratio sees rollout_id=None.
        predictive_controller.set_current_step_context(rollout_id=rollout_id, step_id=step_id)
        resolved_predictive_train_mode = (
            predictive_train_mode
            if predictive_train_mode is not None
            else get_predictive_train_mode_for_step(
                role=role,
                predictive_enabled=predictive_router_enabled,
                step_id=step_id,
            )
        )
        router_logits_step_name = (
            f"training_{rollout_id}_mini{step_id}"
            if router_logits_saver is not None
            else None
        )
        capture_predictive_metric_tensors = (
            router_logits_saver is not None
            and resolved_predictive_train_mode == "compute"
            and predictive_router_enabled
        )
        if capture_predictive_metric_tensors:
            predictive_controller.enable_predictive_metric_tensor_capture()
        else:
            predictive_controller.disable_predictive_metric_tensor_capture()
            predictive_controller.clear_predictive_metric_tensors()

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            parallel_state,
            predictive_train_mode=resolved_predictive_train_mode,
            router_logits_saver=router_logits_saver,
            router_logits_step_name=router_logits_step_name,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if (x := tracker.get("reduce_group")) is not None:
                    torch.distributed.all_reduce(values, group=x)
                if (x := tracker.get("avg_group")) is not None:
                    torch.distributed.all_reduce(values, group=x, op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from miles.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        predictive_metrics = {}
        predictive_metric_details = {}
        if predictive_router_enabled:
            predictive_metrics, predictive_metric_details = predictive_controller.get_and_clear_predictive_metrics_with_details()

        predictive_metric_tensors = predictive_controller.get_and_clear_predictive_metric_tensors()
        predictive_controller.disable_predictive_metric_tensor_capture()
        if (
            router_logits_saver is not None
            and router_logits_step_name is not None
            and predictive_metric_details
        ):
            if mpu.get_data_parallel_world_size() > 1:
                predictive_metric_tensors = RouterReplayLogitsSaver.gather_predictive_metric_tensors_from_dp_group(
                    predictive_metric_tensors,
                    max_tokens=router_logits_saver.max_tokens,
                )
            if mpu.get_data_parallel_rank() == 0:
                router_logits_saver.save_predictive_metrics_async(predictive_metric_details, router_logits_step_name)
                if predictive_metric_tensors and any(predictive_metric_tensors.values()):
                    router_logits_saver.save_predictive_metric_tensors_async(
                        predictive_metric_tensors,
                        router_logits_step_name,
                        topk=router_topk,
                    )

        # per train step log.
        if is_megatron_main_rank():
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role_tag = "" if role == "actor" else f"{role}-"

            extra_metrics = {}
            if args.enable_mtp_training:
                extra_metrics["mtp_loss"] = mtp_losses
            extra_metrics.update(predictive_metrics)

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                extra_metrics[f"lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            log_dict = log_train_step(
                args=args,
                loss_dict=loss_dict,
                grad_norm=grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                num_steps_per_rollout=num_steps_per_rollout,
                role=role,
                extra_metrics=extra_metrics,
                should_log=True,
            )

            if args.ci_test and not args.ci_disable_kl_checker:
                check_kl(args, log_dict, step_id, accumulated_step_id)

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_test:
                check_grad_norm(
                    args=args,
                    grad_norm=grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    role=role,
                    rank=mpu.get_data_parallel_rank(),
                )

    # Drop rollout/step context now that the loop is done; if the next caller
    # forgets to set it again we want resolve_predictive_topk_margin_ratio to
    # see None (= disabled) rather than a stale id from the previous rollout.
    get_predictive_replay_controller().clear_current_step_context()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int, model: Sequence[DDP], optimizer: MegatronOptimizer, opt_param_scheduler: OptimizerParamScheduler
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    hashes = None
    if args.ci_test and args.ci_save_model_hash:
        hashes = compute_model_hashes_by_layer(model)
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    if is_lora_model(model):
        save_checkpoint_with_lora(iteration, model, optimizer, opt_param_scheduler)
    else:
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
        )

    if hashes is not None:
        save_model_hashes(args, model, iteration, hashes)
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def save_hf_model(args, rollout_id: int, model: Sequence[DDP]) -> None:
    """Save Megatron model in HuggingFace format.

    For LoRA models this saves both:
    - A **merged** HF model (adapter weights folded into base) at ``{path}/``
      so it can be loaded directly with ``AutoModelForCausalLM.from_pretrained``.
    - An **adapter-only** HF PEFT checkpoint at ``{path}/adapter/``
      so it can be loaded with ``PeftModel.from_pretrained``.

    This function is collective — all ranks must call it.

    Args:
        args: Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

        import miles_plugins.megatron_bridge  # noqa: F401

        from miles.utils.megatron_bridge_utils import patch_megatron_model

        path = Path(args.save_hf.format(rollout_id=rollout_id))

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

        path.mkdir(parents=True, exist_ok=True)

        with patch_megatron_model(model):
            # For LoRA models, merge_adapter_weights=True (default) merges
            # adapter weights into base weights for a standalone HF model.
            bridge.save_hf_pretrained(model, path=path)

        if should_log:
            logger.info(f"Successfully saved merged HuggingFace model to {path}")
    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")

    # Additionally save adapter-only checkpoint for LoRA models
    if is_lora_model(model):
        try:
            adapter_path = Path(args.save_hf.format(rollout_id=rollout_id)) / "adapter"
            if should_log:
                logger.info(f"Saving LoRA adapter (HF PEFT format) to {adapter_path}")
            save_lora_checkpoint(model, args, str(adapter_path))
            if should_log:
                logger.info(f"Successfully saved LoRA adapter to {adapter_path}")
        except Exception as e:
            if should_log:
                logger.error(f"Failed to save LoRA adapter: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role

    # Snapshot per-param-group max_lr right after construction (config_overrides
    # in setup_model_and_optimizer placed the bias-predictor lr_mult correctly
    # on the right group via Megatron's ParamKey(is_bias_predictor) filter).
    # load_checkpoint + --override-opt_param-scheduler is observed to clobber
    # max_lr across groups (job 313141: actor pg_0 came up at 2e-4 instead of
    # 2e-6, blew up grad_norm to 49 in 2 steps). The Megatron distributed
    # optimizer reshards bias_predictor params at construction and the
    # is_bias_predictor attr is lost on the sharded versions, so the predictive
    # group cannot be re-identified by attribute post-load. We snapshot the
    # construction-time max_lrs and force-restore them after the optimizer
    # state is loaded but before scheduler.step() reads them.
    _construction_max_lrs = [pg.get("max_lr") for pg in optimizer.param_groups]

    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    check_peak_gpu_memory_after_load(args)
    clear_memory()

    check_model_hashes(args, model, iteration)

    if len(optimizer.param_groups) == len(_construction_max_lrs):
        for pg, expected_max_lr in zip(optimizer.param_groups, _construction_max_lrs, strict=True):
            if expected_max_lr is None:
                continue
            if pg.get("max_lr") != expected_max_lr:
                logger.info(
                    "[Predictive Routing Replay] Restoring param-group max_lr: was=%s, expected=%s",
                    pg.get("max_lr"),
                    expected_max_lr,
                )
                pg["max_lr"] = expected_max_lr

    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    return model, optimizer, opt_param_scheduler, iteration
