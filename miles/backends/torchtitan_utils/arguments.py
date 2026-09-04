"""Arguments for the torchtitan backend.

Extends FSDPArgs rather than restating it: those fields are the common
non-Megatron training options (optimizer, LR schedule, precision, profiling),
not FSDP-specific ones. Only the titan-specific knobs are added here, and
torchtitan's own deeper configuration stays reachable through its config tree
rather than being mirrored into argparse.
"""

import dataclasses
from dataclasses import dataclass

from miles.backends.fsdp_utils.arguments import (
    FSDPArgs,
    build_dataclass_parser,
    load_args_from_parser,
)


@dataclass
class TorchtitanArgs(FSDPArgs):
    # Which torchtitan model to build: resolved as
    # torchtitan.models.<name>.model_registry(<flavor>).
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    # "flex" | "flex_flash" | "varlen". flex is torchtitan's default training
    # path for these models and masks document boundaries from ``positions``,
    # which is what allows packed microbatches. sdpa is not offered: the pinned
    # torchtitan removed it for language models.
    titan_attn_backend: str = "flex"

    # Sequence length titan sizes its RoPE caches for; must cover the longest
    # packed microbatch (prompt + response).
    titan_seq_len: int = 4096

    # Truncate the built model to the first N transformer blocks (0 = keep all).
    # For loading a few-layer cutdown of a large checkpoint, whose depth has to
    # match exactly. Structural validation only: per-block init scaling was
    # already computed for the full depth, which is harmless because real weights
    # overwrite it, but it makes a from-scratch run with this flag meaningless.
    titan_num_layers: int = 0

    # Parallelism, verbatim from torchtitan's ParallelismConfig: same names,
    # same defaults, same semantics. The FSDP axis is data_parallel_shard_degree
    # with -1 meaning "infer from world size / the other degrees", exactly as
    # torchtitan runs it; miles adds no renamed aliases of its own.
    titan_data_parallel_replicate_degree: int = 1
    titan_data_parallel_shard_degree: int = -1
    titan_tensor_parallel_degree: int = 1
    titan_pipeline_parallel_degree: int = 1
    titan_context_parallel_degree: int = 1
    titan_expert_parallel_degree: int = 1
    # Empty string keeps torchtitan's own default schedule.
    titan_pipeline_parallel_schedule: str = ""

    wandb_project: str = "miles-torchtitan"


def build_torchtitan_parser(extra_args_provider=None):
    return build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    import torch

    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if args.titan_attn_backend == "sdpa":
        raise ValueError(
            "--titan-attn-backend sdpa: the pinned torchtitan removed sdpa for language models; use flex"
        )
    # The two remaining backends do not share a threshold, and getting this
    # wrong is how a torch bump silently breaks one of them: varlen_attn's
    # enable_gqa= is in 2.12, but create_block_mask's separate_full_blocks
    # kwarg is only public from 2.13 (verified against the v2.12.0/v2.13.0 tags).
    needed = (2, 13) if args.titan_attn_backend.startswith("flex") else (2, 12)
    if torch_version < needed:
        raise ValueError(
            f"--titan-attn-backend {args.titan_attn_backend} needs "
            f"torch>={'.'.join(map(str, needed))}; this environment runs {torch.__version__}"
        )
    if args.titan_context_parallel_degree != 1 and args.titan_model_name == "qwen3_5":
        # Upstream rejects it: GatedDeltaNet needs the full sequence.
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    # torchtitan sizes its rotary tables from training.seq_len, and the kernels
    # assert on a position beyond them -- device-side, with a traceback pointing
    # at rope rather than at the setting that caused it. A packed microbatch
    # restarts positions per document, so the bound that matters is the longest
    # single sequence, not the packed length.
    if args.rollout_max_context_len is not None:
        if args.titan_seq_len < args.rollout_max_context_len:
            raise ValueError(
                f"--titan-seq-len {args.titan_seq_len} is shorter than "
                f"--rollout-max-context-len {args.rollout_max_context_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.titan_seq_len <= args.rollout_max_response_len:
        # Without an explicit context bound the prompt length is unknown, but a
        # sequence is at least one prompt token plus the response, so equality
        # already overflows the tables.
        raise ValueError(
            f"--titan-seq-len {args.titan_seq_len} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    # The reference model is built once from --ref-load; refreshing it mid-run
    # would need the actor-to-ref copy FSDP does, which this backend has not
    # wired up. Silently ignoring the interval would quietly train against a
    # stale reference.
    if getattr(args, "ref_update_interval", None) is not None:
        raise ValueError("--ref-update-interval is not supported by the torchtitan backend")
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")

    known = {f.name for f in dataclasses.fields(TorchtitanArgs)}
    assert "titan_model_name" in known  # guards against a silent rename
