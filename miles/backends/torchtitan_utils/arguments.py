"""Arguments for the torchtitan backend.

Extends FSDPArgs for the common non-Megatron training options and adds only
the titan-specific knobs; torchtitan's deeper configuration stays reachable
through its own config tree rather than being mirrored into argparse.
"""

from dataclasses import dataclass

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser


@dataclass
class TorchtitanArgs(FSDPArgs):
    # torchtitan.models.<name>.model_registry(<flavor>)
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    # "flex" | "flex_flash" | "varlen"; the pinned torchtitan has no sdpa for language models
    titan_attn_backend: str = "flex"

    # sizes torchtitan's rotary tables; must cover the longest prompt + response
    titan_seq_len: int = 4096

    # keep only the first N transformer blocks (0 = all), for few-layer cutdown checkpoints
    titan_num_layers: int = 0

    # torchtitan's ParallelismConfig fields, same names, defaults and semantics
    titan_data_parallel_replicate_degree: int = 1
    titan_data_parallel_shard_degree: int = -1
    titan_tensor_parallel_degree: int = 1
    titan_pipeline_parallel_degree: int = 1
    titan_context_parallel_degree: int = 1
    titan_expert_parallel_degree: int = 1
    titan_pipeline_parallel_schedule: str = ""

    wandb_project: str = "miles-torchtitan"


def build_torchtitan_parser(extra_args_provider=None):
    return build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    import torch

    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 13):
        raise ValueError(f"the torchtitan backend needs torch>=2.13; this environment runs {torch.__version__}")
    if args.titan_attn_backend == "sdpa":
        raise ValueError("--titan-attn-backend sdpa: the pinned torchtitan removed sdpa for language models; use flex")
    if args.titan_context_parallel_degree != 1 and args.titan_model_name == "qwen3_5":
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    if args.rollout_max_context_len is not None:
        if args.titan_seq_len < args.rollout_max_context_len:
            raise ValueError(
                f"--titan-seq-len {args.titan_seq_len} is shorter than "
                f"--rollout-max-context-len {args.rollout_max_context_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.titan_seq_len <= args.rollout_max_response_len:
        raise ValueError(
            f"--titan-seq-len {args.titan_seq_len} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    if getattr(args, "ref_update_interval", None) is not None:
        raise ValueError("--ref-update-interval is not supported by the torchtitan backend")
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")
