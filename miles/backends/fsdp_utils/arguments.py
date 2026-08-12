import argparse
import dataclasses
import difflib
from dataclasses import dataclass

import yaml


@dataclass
class FSDPArgs:
    # Optim
    optimizer: str = "adam"  # Optimizer type: "adam" (AdamW)
    lr: float = 2e-5
    lr_warmup_init: float = 0.0
    min_lr: float = 0.0
    lr_decay_style: str = "constant"
    lr_decay_iters: int | None = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: float | None = None
    lr_wsd_decay_iters: int | None = None
    lr_wsd_decay_style: str | None = None
    use_checkpoint_lr_scheduler: bool = True
    override_lr_scheduler: bool = False
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03

    attn_implementation: str = "flash_attention_2"

    # Logging
    wandb_project: str = "miles-fsdp"
    wandb_run_name: str | None = None

    # Precision
    gradient_checkpointing: bool = False
    fp16: bool = False
    keep_fp32_master: bool = True

    # FSDP configuration
    fsdp_state_dict_cpu_offload: bool = True  # If True, offload full state dict to CPU during collection.
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )
    # FSDP2 hybrid-shard replica count.
    dp_replicate_size: int = 1

    deterministic_mode: bool = False  # This name must be the same as Megatron's

    # The FSDP backend is pure data parallel. This knob only exists so shared argument
    # validation can reject a context-parallel run with a clear message.
    context_parallel_size: int = 1
    # Profile
    record_memory_history: bool = False
    memory_snapshot_path: str = "snapshot.pickle"
    use_pytorch_profiler: bool = False
    profile_step_start: int = 10
    profile_step_end: int = 12
    tensorboard_dir: str | None = None

    # YAML bookkeeping
    config: str | None = None


def build_fsdp_parser(extra_args_provider=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("FSDP SFT Training (miles)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(FSDPArgs):
        if f.name == "config":
            continue

        # Handle union types like int | None, str | None, etc.
        if hasattr(f.type, "__args__"):  # Check if it's a Union type
            # For T | None, use T as the type
            non_none_types = [t for t in f.type.__args__ if t is not type(None)]
            arg_type = non_none_types[0] if non_none_types else str
        else:
            arg_type = f.type

        if arg_type is bool:
            parser.add_argument(
                f"--{f.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=f.default
            )
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    return parser


def parse_fsdp_cli(extra_args_provider=None):
    return build_fsdp_parser(extra_args_provider).parse_args()


def reject_unknown_config_keys(data: dict, known: set[str]) -> None:
    unknown = sorted(set(data) - known)
    if not unknown:
        return

    described = []
    for key in unknown:
        close = difflib.get_close_matches(key, known, n=1)
        described.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
    raise ValueError(f"unknown key(s) in the YAML config: {', '.join(described)}")


def load_fsdp_args(extra_args_provider=None):
    parser = build_fsdp_parser(extra_args_provider)
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        reject_unknown_config_keys(data, set(vars(args)))
        parser.set_defaults(**data)
        args = parser.parse_args()
    args.bf16 = not args.fp16
    return args


def validate_hybrid_shard_args(args) -> None:
    """Validate that the training topology can form the requested FSDP2 mesh."""
    replicate_size = args.dp_replicate_size
    if replicate_size < 1:
        raise ValueError(f"dp_replicate_size must be at least 1, got {replicate_size}")

    world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    if world_size % replicate_size:
        raise ValueError(f"world_size({world_size}) must be divisible by dp_replicate_size({replicate_size})")
