from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


def configure_tinker_mode(args):
    training_only = args.debug_train_only or args.load_debug_rollout_data is not None
    inference_only = args.debug_rollout_only
    mode = args.tinker_mode
    if mode is None:
        mode = "training" if training_only else "inference" if inference_only else "combined"
    if mode not in ("combined", "training", "inference"):
        raise ValueError(f"Unknown --tinker-mode {mode!r}")
    if (training_only and mode != "training") or (inference_only and mode != "inference"):
        raise ValueError("--tinker-mode conflicts with existing component flags or --load-debug-rollout-data")
    if mode == "training" and args.eval_num_gpus > 0:
        raise ValueError("--tinker-mode training requires --eval-num-gpus 0; no inference engines are started")
    args.tinker_mode = mode
    # Normalize public modes into the existing placement controls after custom-config overrides.
    args.debug_train_only = mode == "training"
    args.debug_rollout_only = mode == "inference"


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True


def validate_tinker_args(args):
    """Checks on the final arguments; preprocessing runs before --custom-config-path is merged."""
    # --save-interval and the other save-loop flags already require --save in miles_validate_args
    for flag, value in (("--save", args.save), ("--save-hf", args.save_hf)):
        assert value is None, f"Tinker does not support {flag}; checkpoints are saved through the Tinker API"
    assert args.tinker_checkpoint_root, "Tinker requires --tinker-checkpoint-root"
