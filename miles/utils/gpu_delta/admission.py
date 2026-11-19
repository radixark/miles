"""Fail early when a receiver cannot retain the immutable HF baseline."""

import os

from miles.backends.sglang_utils.arguments import collect_eval_sglang_overrides
from miles.utils.gpu_delta.publication import ADMITTED_LOAD_FORMATS, configured_codec, configured_skip_payload_hash


def _validate_load_formats(load_format, draft_load_format=None):
    if load_format not in ADMITTED_LOAD_FORMATS:
        raise ValueError("GPU delta requires a standard local safetensors target load, not dummy loading")
    # An omitted draft format inherits the target format in SGLang.
    if draft_load_format is not None and draft_load_format not in ADMITTED_LOAD_FORMATS:
        raise ValueError("GPU delta keeps the MTP draft static and requires its real checkpoint weights")


def validate_gpu_delta_args(args):
    """Validate the effective launch configuration after custom overrides."""
    if args.update_weight_transfer_mode != "gpu-delta":
        return
    assert not (
        args.debug_train_only and args.eval_num_gpus > 0
    ), "Dedicated GPU-delta eval requires a live trainer snapshot publisher"
    assert args.update_weight_delta_encoding == "xor", "GPU delta supports only --update-weight-delta-encoding xor"
    assert not args.colocate, "GPU delta requires separate training and rollout GPUs"
    assert not args.offload_rollout, "GPU delta requires resident original rollout weight storage"
    assert "train" not in args.ft_components, "GPU delta requires job/checkpoint restart after sender baseline loss"
    assert (
        args.train_backend == "megatron" and args.megatron_to_hf_mode != "bridge"
    ), "GPU delta requires the direct Megatron exporter"
    assert args.lora_rank <= 0, "GPU delta does not support LoRA"
    assert args.prefill_num_servers is None, "GPU delta does not support PD"
    # Rollout syncs pause through SGLang's scheduler, which has only these two modes; abort is a tokenizer-side abort.
    assert args.pause_generation_mode in (
        "retract",
        "in_place",
    ), "GPU delta requires --pause-generation-mode retract or in_place"
    assert not args.check_weight_update_equal, "GPU delta requires an independent, non-destructive correctness check"
    assert args.update_weight_disk_dir and os.path.isdir(
        args.hf_checkpoint
    ), "GPU delta requires a shared publication directory and a local canonical HF checkpoint"
    configured_codec()
    configured_codec(initial_sync=True)
    # The writer records this policy in every publication, and requesters follow the publication.
    configured_skip_payload_hash()
    assert not args.mtp_num_layers, "GPU delta requires a static receiver MTP draft"
    target_format = args.sglang_load_format
    draft_format = args.sglang_speculative_draft_load_format
    _validate_load_formats(target_format, draft_format)
    # An absent --eval-sglang-* value inherits the --sglang-* one.
    eval_overrides = collect_eval_sglang_overrides(args)
    _validate_load_formats(
        eval_overrides.get("load_format", target_format),
        eval_overrides.get("speculative_draft_load_format", draft_format),
    )
