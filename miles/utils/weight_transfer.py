"""Weight-transfer mode selection without importing distributed backends."""

from argparse import Namespace

WEIGHT_TRANSFER_MODES = ("broadcast", "broadcast_packed", "p2p", "disk-delta")


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
