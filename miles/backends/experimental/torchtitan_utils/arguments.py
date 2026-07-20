"""TorchTitanArgs: derived from FSDPArgs' field list (see fsdp_utils/arguments.py),
dropping fsdp-only knobs (attn_implementation, fsdp_cpu_*, deterministic_mode — titan
owns attention/sharding/precision instead) and adding titan-specific knobs.

Keeps every field shared miles code dereferences WITHOUT getattr regardless of backend:
context_parallel_size, fp16, and the TrainProfiler fields (record_memory_history,
memory_snapshot_path, use_pytorch_profiler, profile_step_start, profile_step_end,
tensorboard_dir). Deliberately omits update_weight_buffer_size: the main parser already
registers ``--update-weight-buffer-size`` via a plain add_argument (not a reset_arg),
and redefining it here would collide at parse time once extra_args_provider merges this
parser's args into the same namespace.
"""

import argparse
import dataclasses
from dataclasses import dataclass

import yaml


@dataclass
class TorchTitanArgs:
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

    # Logging
    wandb_project: str = "miles-torchtitan"
    wandb_run_name: str | None = None

    # Precision / activation checkpointing
    gradient_checkpointing: bool = False
    fp16: bool = False

    # Context Parallelism (unwired for torchtitan v1; kept so shared miles code that
    # dereferences it without getattr keeps working)
    context_parallel_size: int = 1

    # Profile (TrainProfiler reads these regardless of backend)
    record_memory_history: bool = False
    memory_snapshot_path: str = "snapshot.pickle"
    use_pytorch_profiler: bool = False
    profile_step_start: int = 10
    profile_step_end: int = 12
    tensorboard_dir: str | None = None

    # torchtitan-specific parallelism/runtime knobs
    tt_tensor_parallel_size: int = 1
    tt_expert_parallel_size: int = 1  # v1 asserts == 1; EP>1 FSDP wrap is broken on 2.11
    tt_dp_replicate: int = 1
    tt_attn_backend: str = "flex"  # varlen kwargs are nightly-era on torch 2.11
    tt_ac_mode: str = "none"  # none | selective | full | memory-budget
    tt_compile: bool = False
    tt_model_flavor: str | None = None  # override the mapper's inferred flavor

    # YAML bookkeeping
    config: str | None = None


def parse_torchtitan_cli(extra_args_provider=None):
    parser = argparse.ArgumentParser("TorchTitan Training (miles)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(TorchTitanArgs):
        if f.name == "config":
            continue

        if hasattr(f.type, "__args__"):  # Union types like int | None, str | None
            non_none_types = [t for t in f.type.__args__ if t is not type(None)]
            arg_type = non_none_types[0] if non_none_types else str
        else:
            arg_type = f.type

        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args


def load_torchtitan_args(extra_args_provider=None):
    args = parse_torchtitan_cli(extra_args_provider)
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    return args
