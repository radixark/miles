import dataclasses
import gc
import logging
import math
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

from miles.utils.memory_utils import clear_memory

from ..training_utils.ci_utils import check_grad_norm, check_kl
from ..training_utils.data import DataIterator, get_batch
from ..training_utils.log_utils import aggregate_forward_results, aggregate_train_losses, log_train_step
from ..training_utils.loss import loss_function
from ..training_utils.parallel import ParallelState
from .checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_with_lora
from .lora_utils import create_lora_instance, is_lora_enabled, is_lora_model
from .model_provider import get_model_provider_func
from .parallel import get_packed_seq_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bridge / LoRA model setup helpers
# ---------------------------------------------------------------------------


@dataclass
class _BridgeWrapperConfig:
    """Configuration for Megatron-Bridge module wrapping."""

    is_value_model: bool = False
    wrap_with_ddp: bool = True
    use_distributed_optimizer: bool = True


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _make_value_model_hook(hidden_size: int, sequence_parallel: bool):
    """Create a pre-wrap hook that replaces the output layer with a value head."""
    from megatron.core import parallel_state

    from .model_provider import LinearForLastLayer

    def hook(model):
        model_post_process = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i))
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        model_list = _ensure_model_list(model)
        assert len(model_post_process) == len(model_list), "Model list length and post process list length must match."

        for index, model_chunk in enumerate(model_list):
            if not model_post_process[index]:
                continue
            model_chunk.output_layer = LinearForLastLayer(
                input_size=hidden_size,
                output_size=1,
                sequence_parallel=sequence_parallel,
            )

    return hook


def _get_model_config_from_wrapped(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def _setup_lora_model_via_bridge(args: Namespace) -> list:
    """Build Megatron model with LoRA using Megatron-Bridge.

    This handles:
    1. Creating the Bridge and Provider
    2. Creating and registering the LoRA pre-wrap hook
    3. Registering value-model hooks if needed
    4. Building the DDP-wrapped model

    Args:
        args: Training arguments.

    Returns:
        List of DDP-wrapped model chunks with LoRA applied.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig

    # Build Bridge and Provider
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    # Set parallel configs on the provider
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.finalize()

    # Create and register LoRA hook (must be applied BEFORE DDP wrapping so that
    # DDP and the distributed optimizer only track trainable adapter parameters)
    lora = create_lora_instance(args)

    def apply_lora_hook(model_chunks):
        transformed = lora(model_chunks, training=True)
        lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_lora_hook)

    # Register value-model hook if needed
    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    # Build DDP config and create model
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    ddp_config.finalize()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    return model


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
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

    if is_lora_enabled(args) and role == "actor" and args.megatron_to_hf_mode == "bridge":
        model = _setup_lora_model_via_bridge(args)
    else:
        model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
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
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            parallel_state,
            args.data_pad_size_multiplier,
            args.qkv_format,
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

        return output_tensor, partial(
            f,
            args=args,
            parallel_state=parallel_state,
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
        batch = get_batch(
            data_iterator,
            [
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
            ],
            parallel_state,
            args.data_pad_size_multiplier,
            args.qkv_format,
        )

        from miles.utils.replay_base import all_replay_managers

        old_stages = [m.stage for m in all_replay_managers]
        for m in all_replay_managers:
            m.stage = "replay_forward"

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

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            output_tensor = model(**forward_kwargs)

        for m, old_stage in zip(all_replay_managers, old_stages, strict=True):
            m.stage = old_stage

        return output_tensor, partial(
            loss_function, args, parallel_state, batch, num_microbatches, apply_megatron_loss_scaling=True
        )

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
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
    if args.ci_test and args.enable_mtp_training:
        from miles.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate.
        assert update_successful
        opt_param_scheduler.step(increment=args.global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = aggregate_train_losses(losses_reduced, parallel_state)
        return loss_reduced, grad_norm
    return {}, grad_norm


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
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
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

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

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
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from miles.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"

            extra_metrics = {}
            if args.enable_mtp_training:
                extra_metrics["mtp_loss"] = mtp_losses

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

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------


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
    from .lora_utils import is_lora_model, save_lora_checkpoint

    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

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


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


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
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    clear_memory()

    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    return model, optimizer, opt_param_scheduler, iteration
