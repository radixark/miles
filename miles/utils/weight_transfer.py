"""Weight-transfer mode selection without importing distributed backends."""

from argparse import ArgumentParser, Namespace

WEIGHT_TRANSFER_MODES = ("broadcast", "broadcast_packed", "p2p", "disk-delta")


def add_weight_transfer_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--update-weight-transfer-mode",
        choices=WEIGHT_TRANSFER_MODES,
        default="broadcast",
        help=(
            "The method to transfer weights to remote rollout engines during update weight. "
            "'broadcast' (default) broadcasts each tensor separately; 'broadcast_packed' "
            "packs each bucket into one byte broadcast. The packed mode requires Megatron "
            "non-colocated transfer and SGLang's mixed-dtype flattened-bucket API. It adds a "
            "contiguous bucket allocation on sender and receivers; atomic update units may "
            "exceed --update-weight-buffer-size. "
            "'disk-delta' diffs each sync against a CPU snapshot of the previous one and publishes "
            "only the changed bytes to --update-weight-disk-dir; each engine's /pull_weights applies "
            "them into a host-local checkpoint that the engine reloads from."
        ),
    )


def is_broadcast_mode(mode: str) -> bool:
    return mode in ("broadcast", "broadcast_packed")


def validate_weight_transfer_args(args: Namespace) -> None:
    """Validate CLI, custom YAML and direct caller selections before dispatch."""
    if hasattr(args, "update_weight_use_flattened_buckets"):
        raise ValueError(
            "update_weight_use_flattened_buckets was replaced by "
            "--update-weight-transfer-mode=broadcast_packed; use broadcast for per-tensor transfer"
        )
    mode = getattr(args, "update_weight_transfer_mode", "broadcast")
    if mode not in WEIGHT_TRANSFER_MODES:
        raise ValueError(f"Unknown --update-weight-transfer-mode {mode!r}")
    if mode == "broadcast_packed" and (
        getattr(args, "train_backend", None) != "megatron" or getattr(args, "colocate", False)
    ):
        raise ValueError("broadcast_packed requires Megatron non-colocated weight transfer")
