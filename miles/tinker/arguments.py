from miles.tinker.core.types import GatewayConfig
from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


def add_tinker_arguments(parser):
    # Tinker targets cover full sequences; bound the FP32 cross-entropy temporaries.
    parser.set_defaults(log_probs_chunk_size=1024)
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "batch-token-budget",
        type=int,
        default=GatewayConfig.batch_token_budget,
        help="Token budget per gateway work unit; independent of --max-tokens-per-gpu.",
    )
    add_argument(
        "pack-before-dp",
        action="store_true",
        help="Pack dynamic microbatches before distributing them across DP ranks.",
    )
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
    if args.tinker_batch_token_budget <= 0:
        raise ValueError("--tinker-batch-token-budget must be positive")
    if args.tinker_pack_before_dp:
        if not args.use_dynamic_batch_size or args.indep_dp or args.balance_by_flops:
            raise ValueError(
                "--tinker-pack-before-dp requires --use-dynamic-batch-size and does not support "
                "--indep-dp or --balance-by-flops"
            )
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
