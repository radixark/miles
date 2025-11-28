"""
Helper routines for Miles Ray actors.

NOTE: This duplicates logic that also lives inside `model.py`/`train_one_step`.
We extracted it here to share code between gradient-accumulation helpers, but
the implementation still needs real refactoring to avoid copy/paste.
"""

import math
import os
from functools import partial
from typing import Callable, Dict, Tuple

import torch
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.global_vars import get_args

from .data import get_batch
from .loss import get_log_probs_and_entropy, loss_function
from .model import forward_only


def _build_forward_step_fn(actor, args, num_microbatches):
    """
    Build the forward_step_fn closure used by Megatron's pipeline engine.

    This is shared by the training path and the forward_backward_only helper.
    """

    def forward_step(
        iterator: "DataIterator",
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> Tuple[torch.Tensor, Callable]:
        batch = get_batch(
            iterator,
            [
                "tokens",
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
            ],
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
        else:
            old_stage = None

        def build_loss_mask_for_mtp(batch_data: dict[str, object]) -> torch.Tensor | None:
            tokens_tensor: torch.Tensor = batch_data["tokens"]

            mask_chunks: list[torch.Tensor] = []
            for total_len, response_len, resp_mask in zip(
                batch_data["total_lengths"],
                batch_data["response_lengths"],
                batch_data["loss_masks"],
            ):
                assert (
                    resp_mask.numel() == response_len
                ), f"Unexpected loss mask size {resp_mask.numel()} (expected {response_len})."
                prompt_len = total_len - response_len
                full_mask = resp_mask.new_zeros(total_len)
                full_mask[prompt_len:] = resp_mask

                from .cp_utils import slice_with_cp  # local import to avoid cycles

                mask_chunks.append(slice_with_cp(full_mask, 0.0))

            flattened_mask = torch.cat(mask_chunks, dim=0)
            seq_len = tokens_tensor.size(-1)
            assert flattened_mask.numel() <= seq_len, (
                f"MTP loss mask ({flattened_mask.numel()}) exceeds token length ({seq_len})."
            )

            loss_mask_tensor = flattened_mask.new_zeros(seq_len)
            loss_mask_tensor[: flattened_mask.numel()] = flattened_mask
            return loss_mask_tensor.unsqueeze(0)

        loss_mask = None
        mtp_kwargs = None

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should be disabled with combined 1f1b"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
            )
        else:
            if args.enable_mtp_training:
                loss_mask = build_loss_mask_for_mtp(batch)
                assert loss_mask.shape == batch["tokens"].shape, (
                    f"loss_mask shape {loss_mask.shape} mismatches token shape {batch['tokens'].shape}"
                )
                mtp_kwargs = {
                    "mtp_labels": batch["tokens"],
                }

            output_tensor = model(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=loss_mask,
                **(dict(mtp_kwargs=mtp_kwargs) if mtp_kwargs is not None else {}),
            )

        if old_stage is not None:
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches[0])

    return forward_step


def run_forward_backward_only(actor, rollout_id, data_iterator, num_microbatches, zero_grads):
    """Execute forward/backward without optimizer.step() for gradient accumulation."""
    args = get_args()

    if zero_grads:
        for model_chunk in actor.model:
            model_chunk.zero_grad_buffer()
        actor.optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from miles.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, 0, actor.model, actor.optimizer, actor.opt_param_scheduler)

    forward_step = _build_forward_step_fn(actor, args, num_microbatches)
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=actor.model,
        num_microbatches=num_microbatches[0],
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    grad_norm = None
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = actor.optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = actor.optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    loss_dict: Dict[str, float | list[torch.Tensor]] = {}
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        keys = losses_reduced[0]["keys"]
        values = None
        for item in losses_reduced:
            if values is None:
                values = item["values"]
            else:
                values += item["values"]
        assert len(keys) + 1 == values.numel()
        torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))

        values = values.tolist()
        num_samples_or_tokens = values[0]
        for key, value in zip(keys, values[1:]):
            loss_dict[key] = value * mpu.get_context_parallel_world_size() / num_samples_or_tokens

        if "log_probs" in losses_reduced[0]:
            all_log_probs = []
            for entry in losses_reduced:
                if "log_probs" in entry and entry["log_probs"]:
                    all_log_probs.extend(entry["log_probs"])
            if all_log_probs:
                loss_dict["log_probs"] = all_log_probs

    return loss_dict, grad_norm, valid_step


def run_forward_only(actor, data_iterator, num_microbatches):
    """Forward-only pass to collect log-probs/entropy for DPO-style flows."""
    rollout_data_result = forward_only(
        get_log_probs_and_entropy,
        actor.args,
        actor.model,
        data_iterator,
        num_microbatches,
        store_prefix="",
    )

    loss_dict: Dict[str, list[torch.Tensor]] = {}
    if mpu.is_pipeline_last_stage():
        if "log_probs" in rollout_data_result:
            loss_dict["log_probs"] = rollout_data_result["log_probs"]
        if "entropy" in rollout_data_result:
            loss_dict["entropy"] = rollout_data_result["entropy"]
    return loss_dict
