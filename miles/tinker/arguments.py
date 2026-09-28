from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets

_GLM5_NEXT_NATIVE_SMOKE_HF_TARGETS = {
    "model.language_model.layers.*.mlp.gate_proj",
    "model.language_model.layers.*.mlp.up_proj",
    "model.language_model.layers.*.mlp.down_proj",
    "model.language_model.layers.*.mlp.shared_experts.gate_proj",
    "model.language_model.layers.*.mlp.shared_experts.up_proj",
    "model.language_model.layers.*.mlp.shared_experts.down_proj",
}


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

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
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    if "glm5_next" in (args.model_name or "").lower():
        if set(groups) == _GLM5_NEXT_NATIVE_SMOKE_HF_TARGETS:
            args.tinker_lora_groups = ["mlp"]
            args.target_modules = groups
            return
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
