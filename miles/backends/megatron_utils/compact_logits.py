import torch
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region


def can_compact_actor_logits(args) -> bool:
    """Return whether compact logits support the selected built-in actor path."""
    if not getattr(args, "compact_actor_logits", False):
        return False
    if getattr(args, "loss_type", None) not in {"policy_loss", "sft_loss"}:
        return False

    unsupported_values = (
        getattr(args, "enable_mtp_training", False),
        getattr(args, "custom_model_provider_path", None),
        getattr(args, "megatron_to_hf_mode", None) == "bridge",
        getattr(args, "multi_lora_n_adapters", 0),
        getattr(args, "custom_megatron_init_path", None),
        getattr(args, "custom_megatron_before_log_prob_hook_path", None),
        getattr(args, "custom_megatron_before_train_step_hook_path", None),
        getattr(args, "custom_advantage_function_path", None),
        getattr(args, "rollout_data_postprocess_path", None),
        getattr(args, "custom_pg_loss_reducer_function_path", None),
        getattr(args, "custom_tis_function_path", None),
        getattr(args, "get_mismatch_metrics", False),
        getattr(args, "use_rollout_entropy", False),
        getattr(args, "save_debug_train_data", None),
        getattr(args, "use_rollout_logprobs", False),
        getattr(args, "custom_loss_function_path", None),
        getattr(args, "config_logger_dir", None),
        getattr(args, "cuda_graph_impl", "none") != "none",
    )
    if any(unsupported_values):
        return False

    return not (getattr(args, "advantage_estimator", None) == "ppo" and getattr(args, "kl_coef", 0) != 0)


def compact_logits_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor,
    inference_context,
    runtime_gather_output: bool | None,
    scale_logits,
    config,
    **_,
) -> torch.Tensor:
    """Project only hidden-state rows selected by the target-aligned loss mask."""
    assert labels is None
    assert inference_context is None
    assert hidden_states.ndim == 3 and hidden_states.size(1) == 1
    assert loss_mask.ndim == 2 and loss_mask.size(0) == 1
    assert hidden_states.device == loss_mask.device

    sequence_parallel = output_layer.sequence_parallel
    tp_group = output_layer.tp_group
    expected_sequence_length = hidden_states.size(0) * (tp_group.size() if sequence_parallel else 1)
    assert loss_mask.size(1) == expected_sequence_length

    projection_weight = output_weight if output_weight is not None else output_layer.weight
    if sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
            group=tp_group,
        )

    selected_hidden_states = hidden_states[:, 0, :][loss_mask[0].bool()].unsqueeze(1)
    original_sequence_parallel = output_layer.sequence_parallel
    try:
        output_layer.sequence_parallel = False
        if selected_hidden_states.size(0) == 0:
            anchor = hidden_states.reshape(-1)[:1].sum() * 0
            anchor = anchor + projection_weight.reshape(-1)[:1].sum() * 0
            logits = hidden_states.new_empty((0, 1, projection_weight.size(0))) + anchor
        else:
            logits, _ = output_layer(
                selected_hidden_states,
                weight=output_weight,
                runtime_gather_output=runtime_gather_output,
            )
    finally:
        output_layer.sequence_parallel = original_sequence_parallel

    return scale_logits(logits).transpose(0, 1).contiguous()
