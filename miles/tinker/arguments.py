import argparse

from miles.utils.hf_config import load_hf_config
from miles.utils.hf_lora_targets import (
    exclude_hf_lora_targets,
    get_hf_lora_targets,
    matches_hf_lora_target,
    parse_lora_targets,
    resolve_hf_lora_targets,
)


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
    hf_config = load_hf_config(args.hf_checkpoint).to_dict()
    layout = get_hf_lora_targets(hf_config)
    requested = resolve_hf_lora_targets(
        hf_config,
        target_modules=parse_lora_targets(args.target_modules),
        train_attn=args.tinker_train_attn,
        train_mlp=args.tinker_train_mlp,
        train_unembed=args.tinker_train_unembed,
    )
    available = layout.attention + layout.mlp + layout.unembed
    for target in requested:
        assert any(matches_hf_lora_target(module, target) for module in available), (
            f"Tinker target {target!r} is not an HF target of this model; use the model's HF projection names"
        )
    targets = [module for module in available if any(matches_hf_lora_target(module, target) for target in requested)]
    targets = exclude_hf_lora_targets(targets, parse_lora_targets(args.exclude_modules) or [])
    for name, group in (("attn", layout.attention), ("mlp", layout.mlp), ("unembed", layout.unembed)):
        selected = set(targets).intersection(group)
        assert not selected or selected == set(group), (
            f"Tinker targets must select the whole {name} group or none of it; "
            "the SDK cannot describe a partial training group"
        )
        setattr(args, f"tinker_train_{name}", bool(selected))
    args.target_modules = targets
    # Exclusions must be applied before advertising the SDK training groups.
    args.exclude_modules = None
