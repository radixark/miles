from dataclasses import dataclass

import torch

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser


@dataclass
class TorchtitanArgs(FSDPArgs):
    # torchtitan.models.<name>.model_registry(<flavor>)
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    seq_length: int = 4096

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1


def build_torchtitan_parser(extra_args_provider=None):
    return build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 13):
        raise ValueError(f"the torchtitan backend needs torch>=2.13; this environment runs {torch.__version__}")
    if args.context_parallel_size != 1 and args.titan_model_name == "qwen3_5":
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    if args.rollout_max_context_len is not None:
        if args.seq_length < args.rollout_max_context_len:
            raise ValueError(
                f"--seq-length {args.seq_length} is shorter than "
                f"--rollout-max-context-len {args.rollout_max_context_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.seq_length <= args.rollout_max_response_len:
        raise ValueError(
            f"--seq-length {args.seq_length} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    if args.ref_update_interval is not None:
        raise ValueError("--ref-update-interval is not supported by the torchtitan backend")
    if args.fp16:
        raise ValueError("the torchtitan backend trains in bf16 mixed precision; --fp16 is not supported")
    if args.lr_decay_style not in ("constant", "linear", "cosine"):
        raise ValueError(
            f"torchtitan's LR schedule has no {args.lr_decay_style!r} decay; use constant, linear or cosine"
        )
    if args.lr_warmup_fraction is not None or args.lr_wsd_decay_iters is not None or args.lr_decay_iters is not None:
        raise ValueError(
            "torchtitan's LR schedule takes --lr-warmup-iters only; fraction, WSD and decay-iters are unsupported"
        )
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")
