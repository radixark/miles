"""Fail early when a receiver cannot retain the immutable HF baseline."""

import os

from miles.utils.gpu_delta.publication import configured_codec


def validate_load_formats(load_format, draft_load_format=None):
    allowed = {"auto", "safetensors", "fastsafetensors"}
    if load_format not in allowed:
        raise ValueError("GPU delta requires a standard local safetensors target load, not dummy loading")
    # An omitted draft format inherits the target format in SGLang.
    if draft_load_format is not None and draft_load_format not in allowed:
        raise ValueError("GPU delta keeps the MTP draft static and requires its real checkpoint weights")


def validate_gpu_delta_args(args):
    """Validate the effective launch configuration after custom overrides."""
    if args.update_weight_transfer_mode != "gpu-delta":
        return
    assert args.update_weight_delta_encoding == "xor", "GPU delta supports only --update-weight-delta-encoding xor"
    assert not args.colocate, "GPU delta requires separate training and rollout GPUs"
    assert (
        args.train_backend == "megatron" and args.megatron_to_hf_mode != "bridge"
    ), "GPU delta requires the direct Megatron exporter"
    assert args.lora_rank <= 0, "GPU delta does not support LoRA"
    assert args.prefill_num_servers is None, "GPU delta does not support PD"
    assert args.pause_generation_mode == "retract", "GPU delta requires retract pause"
    assert not args.check_weight_update_equal, "GPU delta requires an independent, non-destructive correctness check"
    assert args.update_weight_disk_dir and os.path.isdir(
        args.hf_checkpoint
    ), "GPU delta requires a shared publication directory and a local canonical HF checkpoint"
    configured_codec()
    configured_codec(initial_sync=True)
    assert not args.mtp_num_layers, "GPU delta requires a static receiver MTP draft"
    target_format = getattr(args, "sglang_load_format", "auto")
    draft_format = getattr(args, "sglang_speculative_draft_load_format", None)
    validate_load_formats(target_format, draft_format)
    validate_load_formats(
        getattr(args, "eval_sglang_load_format", None) or target_format,
        getattr(args, "eval_sglang_speculative_draft_load_format", None) or draft_format,
    )
