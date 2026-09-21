import argparse

from miles.utils.hf_config import load_hf_config


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
    add_argument("train-attn", action=argparse.BooleanOptionalAction, default=True)
    add_argument("train-mlp", action=argparse.BooleanOptionalAction, default=True)
    add_argument("train-unembed", action=argparse.BooleanOptionalAction, default=True)
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert (
        args.target_modules is None and args.exclude_modules is None
    ), "Tinker uses --tinker-train-attn/mlp/unembed; --target-modules and --exclude-modules are not supported"
    modules = _resolve_target_modules(
        load_hf_config(args.hf_checkpoint),
        train_attn=args.tinker_train_attn,
        train_mlp=args.tinker_train_mlp,
        train_unembed=args.tinker_train_unembed,
    )
    # The common LoRA validator parses and validates this before trainer/engine initialization.
    args.target_modules = ",".join(modules)


def _resolve_target_modules(hf_config, *, train_attn, train_mlp, train_unembed):
    # Other architectures need their own complete attention/MLP mapping.
    assert hf_config.model_type in (
        "qwen3",
        "qwen3_moe",
    ), f"Tinker target layout is not defined for model_type={hf_config.model_type!r}"
    modules = []
    if train_attn:
        modules.extend(("q_proj", "k_proj", "v_proj", "o_proj"))
    if train_mlp:
        modules.extend(("gate_proj", "up_proj", "down_proj"))
    if train_unembed:
        modules.append("lm_head")
    assert modules, "Tinker requires at least one trainable LoRA module group"
    return modules
