from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("full-training", action="store_true", help="Train all model parameters; one active model per trainer")
    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    if args.tinker_full_training:
        assert (
            not args.multi_lora_n_adapters and not args.lora_rank and args.lora_adapter_path is None
        ), "full training cannot enable LoRA"
        assert args.bf16 and not args.fp16, "Tinker full training currently requires BF16"
        assert args.loss_scale in (None, 1.0), "Tinker full training requires unit loss scaling"
        assert args.optimizer == "adam", "Tinker optim_step supplies Adam parameters"
        assert not args.calculate_per_token_loss, "Tinker loss inputs already contain client normalization"
        assert (
            args.use_dynamic_batch_size or args.micro_batch_size == 1
        ), "full training requires --use-dynamic-batch-size or --micro-batch-size 1 for arbitrary SDK batches"
        assert args.context_parallel_size == 1, "Tinker full training does not yet support context parallelism"
        assert not args.offload_train and not args.colocate, "Tinker requires resident trainer and inference GPUs"
        assert not args.overlap_param_gather, "Tinker full training does not yet support overlapping parameter gathers"
        assert not args.use_precision_aware_optimizer, "Tinker full training requires ordinary BF16 Adam"
        for flag in (
            "indep_dp",
            "debug_disable_optimizer",
            "debug_rollout_only",
            "enable_mtp_training",
            "optimizer_cpu_offload",
            "stream_optimizer_state_to_disk",
            "rematerialize_param_from_master_weight",
        ):
            assert not getattr(
                args, flag, False
            ), f"--{flag.replace('_', '-')} is not supported with Tinker full training"
        args.tinker_lora_groups = []
        return
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
