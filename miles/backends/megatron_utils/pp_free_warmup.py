"""One-time, pipeline-free Megatron forward/backward warmup."""

import logging
from contextlib import nullcontext

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
from megatron.core.fp8_utils import get_fp8_recipe
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.random import _fork_rng
from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from transformer_engine.pytorch.graph import restore_fp8_tensors, save_fp8_tensors

from miles.backends.megatron_utils.parallel import get_packed_seq_params
from miles.backends.training_utils.data import get_batch

logger = logging.getLogger(__name__)

_WARMED_ATTR = "_miles_pp_free_warmed"
_BATCH_KEYS = [
    "tokens",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "max_seq_lens",
    "witness_ids",
]


def _zero_grad(model, optimizer):
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    if optimizer is not None:
        optimizer.zero_grad()


def _clear_deferred_embedding_buffers(config, model_chunk):
    if not getattr(config, "defer_embedding_wgrad_compute", False):
        return

    output_model = get_attr_wrapped_model(model_chunk, "post_process", return_model_obj=True)
    if output_model.post_process:
        output_model.embedding_activation_buffer.clear()
        output_model.grad_output_buffer.clear()


def _clear_loss_trackers(args):
    clear_aux_losses_tracker()
    if args.enable_mtp_training:
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

        if "values" in MTPLossLoggingHelper.tracker:
            MTPLossLoggingHelper.clean_loss_in_tracker()

    try:
        from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
    except ImportError:
        pass
    else:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def _tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def _zero_backward_seed(value):
    tensors = [tensor for tensor in _tensor_leaves(value) if tensor.requires_grad]
    if not tensors:
        raise RuntimeError("The PP-free warmup model output has no differentiable tensors")
    loss = tensors[0].float().sum() * 0.0
    for tensor in tensors[1:]:
        loss = loss + tensor.float().sum() * 0.0
    return loss


def _get_local_input_shape(tokens, config):
    # Miles uses variable sequence lengths, so MCore's pipeline shape helper returns ().
    # The batch tokens are already CP-local; only sequence parallelism further shards them.
    seq_length = tokens.shape[-1]
    if config.sequence_parallel:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if seq_length % tp_size != 0:
            raise RuntimeError(f"Local sequence length {seq_length} is not divisible by TP size {tp_size}")
        seq_length //= tp_size
    if getattr(config, "dsv4_mode", False):
        return (seq_length, tokens.shape[0], config.dsv4_hc_mult, config.hidden_size)
    return (seq_length, tokens.shape[0], config.hidden_size)


def run_pp_free_warmup(args, rollout_id, model, optimizer, data_iterator):
    """Run one PP-free forward/backward on every pipeline stage."""
    if len(model) != len(data_iterator):
        raise RuntimeError(
            f"The PP-free warmup needs one data iterator per model chunk; got {len(data_iterator)} "
            f"iterators for {len(model)} chunks"
        )

    if all(getattr(model_chunk, _WARMED_ATTR, False) for model_chunk in model):
        return

    pp_group = mpu.get_pipeline_model_parallel_group()
    if pp_group.size() == 1:
        if dist.get_rank() == 0:
            logger.info("Skipping PP-free warmup because pipeline parallelism is disabled (PP size is 1)")
        for model_chunk in model:
            setattr(model_chunk, _WARMED_ATTR, True)
        return

    config = get_model_config(model[0])
    vpp_enabled = len(model) > 1
    if dist.get_rank() == 0:
        logger.info("Starting PP-free warmup before rollout %s", rollout_id)

    training = [model_chunk.training for model_chunk in model]
    local_inputs = [None] * len(model)
    buffer_state = [
        (buffer, buffer.detach().clone())
        for model_chunk in model
        for _, buffer in model_chunk.named_buffers()
    ]
    fp8_state = (
        save_fp8_tensors(model, get_fp8_recipe(config))
        if getattr(config, "fp8", None) is not None
        else None
    )
    virtual_rank = mpu.get_virtual_pipeline_model_parallel_rank() if vpp_enabled else None

    try:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            for model_chunk in model:
                model_chunk.disable_forward_pre_hook(param_sync=False)

        for vp_stage, (model_chunk, iterator) in enumerate(zip(model, data_iterator, strict=True)):
            if vpp_enabled:
                mpu.set_virtual_pipeline_model_parallel_rank(vp_stage)

            chunk_config = get_model_config(model_chunk)
            batch = get_batch(
                iterator,
                _BATCH_KEYS,
                args.data_pad_size_multiplier,
                args.qkv_format,
                allgather_cp=args.allgather_cp,
            )
            tokens = batch["tokens"]

            if not get_attr_wrapped_model(model_chunk, "pre_process"):
                local_inputs[vp_stage] = torch.zeros(
                    _get_local_input_shape(tokens, chunk_config),
                    dtype=chunk_config.pipeline_dtype or chunk_config.params_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=True,
                )

            forward_kwargs = {
                "input_ids": tokens,
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": get_packed_seq_params(batch, args),
                "loss_mask": batch["full_loss_masks"],
            }
            if args.enable_witness:
                forward_kwargs["witness_ids"] = batch["witness_ids"]
            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": tokens}
            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            model_chunk.train()
            if hasattr(model_chunk, "set_is_first_microbatch"):
                model_chunk.set_is_first_microbatch()
            if local_inputs[vp_stage] is not None:
                get_attr_wrapped_model(model_chunk, "set_input_tensor")([local_inputs[vp_stage]])

            autocast = (
                torch.autocast("cuda", dtype=chunk_config.autocast_dtype)
                if chunk_config.enable_autocast
                else nullcontext()
            )
            with _fork_rng(), torch.enable_grad(), model_chunk.no_sync(), autocast:
                # Warmup JIT kernels concurrently without PP serialization.
                output = model_chunk(**forward_kwargs)
                _zero_backward_seed(output).backward()

        p2p = P2PCommunicator(pp_group=pp_group, config=config)
        is_pp_first_stage = mpu.is_pipeline_first_stage() and not vpp_enabled
        is_pp_last_stage = mpu.is_pipeline_last_stage() and not vpp_enabled
        p2p_buffer_shape = _get_local_input_shape(tokens, config)
        p2p_buffer = torch.empty(
            p2p_buffer_shape, dtype=config.pipeline_dtype, device=torch.cuda.current_device()
        )

        # Setup transport channels (NCCL lazily creates them)
        # Warmup respects VPP mapping if enabled.
        _, _ = p2p.send_forward_backward_recv_forward_backward(
            output_tensor=None if is_pp_last_stage else p2p_buffer,
            input_tensor_grad=None if is_pp_first_stage else p2p_buffer,
            recv_prev=not is_pp_first_stage,
            recv_next=not is_pp_last_stage,
            tensor_shape=p2p_buffer_shape,
        )
        torch.cuda.synchronize()
    finally:
        if vpp_enabled:
            mpu.set_virtual_pipeline_model_parallel_rank(virtual_rank)
        if fp8_state is not None:
            restore_fp8_tensors(model, fp8_state)
        for iterator in data_iterator:
            iterator.reset()
        for vp_stage, model_chunk in enumerate(model):
            model_chunk.train(training[vp_stage])
            if local_inputs[vp_stage] is not None:
                get_attr_wrapped_model(model_chunk, "set_input_tensor")([None])
            _clear_deferred_embedding_buffers(get_model_config(model_chunk), model_chunk)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                model_chunk.enable_forward_pre_hook()
        _clear_loss_trackers(args)
        reset_model_temporary_tensors(config, model)
        _zero_grad(model, optimizer)
        for buffer, saved_buffer in buffer_state:
            buffer.detach().copy_(saved_buffer)

    torch.cuda.synchronize()
    for model_chunk in model:
        setattr(model_chunk, _WARMED_ATTR, True)
    if dist.get_rank() == 0:
        logger.info("PP-free warmup complete on every stage")
