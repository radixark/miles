import hashlib
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def hash_tensor_sha256(tensor: torch.Tensor) -> str:
    """Real (cryptographic) hash: a mismatch here has to mean a bug."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


def is_verifying_transfer_checksums(args: Namespace) -> bool:
    return args.save_inference_engine_weight_checksum and args.update_weight_transfer_mode == "p2p"


def compute_send_checksums(parameters: Mapping[str, torch.Tensor], names: Sequence[str]) -> dict[str, str]:
    assert len(names) == len(set(names)), "P2P bucket contains duplicate tensor names"
    checksums = {}
    for name in names:
        tensor = parameters[name]
        assert tensor.device.type == "cpu", "Expected registered CPU send buffers"
        checksums[name] = hash_tensor_sha256(tensor)
    return checksums


def verify_transfer_checksums(
    *, sent_checksums: dict[str, str], engine_body: dict[str, Any], cell_id: str, rank: int
) -> None:
    received_checksums = _compute_received_checksums(engine_body, rank=rank)

    mismatched_names = sorted(
        name
        for name in sent_checksums.keys() | received_checksums.keys()
        if sent_checksums.get(name) != received_checksums.get(name)
    )
    if mismatched_names:
        raise RuntimeError(
            f"[P2P-Shared] {len(mismatched_names)} tensors reached rollout cell {cell_id} rank {rank} "
            f"with checksums different from what was sent: {mismatched_names}"
        )


def _compute_received_checksums(engine_body: dict[str, Any], *, rank: int) -> dict[str, str]:
    assert engine_body.get("success", False), f"check_weights engine reported failure: {engine_body!r}"
    matching = [info for info in engine_body["ranks"] if _compute_gpu_rank(info) == rank]
    assert (
        len(matching) == 1
    ), f"expected one checksum record for GPU rank {rank}, got {len(matching)}: {engine_body!r}"
    return matching[0]["checksums"]


def _compute_gpu_rank(checksum_info: dict[str, Any]) -> int:
    gpu_ranks = {role_info["rank"] for role_info in checksum_info["parallelism_info"]}
    assert len(gpu_ranks) == 1, f"expected one GPU rank across roles, got {gpu_ranks}: {checksum_info!r}"
    return next(iter(gpu_ranks))
