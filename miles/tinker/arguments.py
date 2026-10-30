from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    # the gateway never runs the RL save loop
    for flag, value in (
        ("--save", args.save),
        ("--save-interval", args.save_interval),
        ("--save-hf", args.save_hf),
        ("--save-trigger-sentinel", args.save_trigger_sentinel),
        ("--custom-megatron-post-save-hook-path", args.custom_megatron_post_save_hook_path),
    ):
        assert value is None, f"Tinker does not support {flag}; checkpoints are saved through the Tinker API"
    assert args.tinker_checkpoint_root, "Tinker requires --tinker-checkpoint-root"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
