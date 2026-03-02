"""CI utilities for Megatron backend testing."""

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import torch
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)encoder\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),
)


def _layer_key(name: str) -> str:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return f"layer_{int(match.group(1)):04d}"
    return "non_layer"


def _hash_tensor_bytes(tensor: torch.Tensor) -> bytes:
    data = tensor.detach()
    if data.is_cuda:
        data = data.cpu()
    if not data.is_contiguous():
        data = data.contiguous()
    return data.view(torch.uint8).numpy().tobytes()


def compute_model_hashes_by_layer(model: Sequence[DDP]) -> dict[str, str]:
    """Compute per-layer SHA256 hashes over parameter bytes.

    Hash input includes parameter name, shape, dtype, and raw bytes.
    """
    hashers: dict[str, hashlib._Hash] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, param in sorted(model_chunk.named_parameters(), key=lambda x: x[0]):
            if param is None:
                continue
            full_name = f"pp{pp_idx}.{name}"
            key = _layer_key(full_name)
            hasher = hashers.setdefault(key, hashlib.sha256())
            hasher.update(full_name.encode("utf-8"))
            hasher.update(str(tuple(param.shape)).encode("utf-8"))
            hasher.update(str(param.dtype).encode("utf-8"))
            hasher.update(_hash_tensor_bytes(param))
    return {k: v.hexdigest() for k, v in sorted(hashers.items(), key=lambda x: x[0])}


def _hash_file_path(base_dir: str | Path, iteration: int) -> Path:
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    cp_rank = mpu.get_context_parallel_rank()
    base = Path(base_dir)
    iter_dir = base if base.name.startswith("iter_") else base / f"iter_{int(iteration):07d}"
    return iter_dir / f"model_hash_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}_cp{cp_rank}.json"


def save_model_hashes(args, model: Sequence[DDP], iteration: int, hashes: dict[str, str]) -> None:
    if not args.ci_test or not args.ci_save_model_hash:
        return
    path = _hash_file_path(args.save, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    logger.info(f"[CI hash] Saved model hashes to {path}")


def check_model_hashes(args, model: Sequence[DDP], iteration: int) -> None:
    if not args.ci_test or not args.ci_check_model_hash:
        return
    path = _hash_file_path(args.load, iteration)
    if not path.is_file():
        raise AssertionError(f"[CI hash] Hash file missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        expected = json.load(f)
    actual = compute_model_hashes_by_layer(model)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
        raise AssertionError(
            "[CI hash] Model hash mismatch after load. "
            f"missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]}"
        )
    logger.info(f"[CI hash] Model hashes match for iteration {iteration}.")


def check_mtp_only_grad(model: Sequence[DDP], step_id: int) -> None:
    """Check that only MTP parameters have non-zero gradients.

    This is used for CI testing to verify that when all outputs are truncated,
    only the MTP layers receive gradients (since only mtp_loss contributes).

    Args:
        model: Sequence of DDP-wrapped model chunks.
        step_id: Current step index for logging.

    Raises:
        AssertionError: If any non-MTP parameter has a non-zero gradient.
    """
    non_mtp_nonzero_grads = []
    mtp_nonzero_grads = []

    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            # Get the main_grad from the distributed optimizer if available
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue

            grad_norm = grad.abs().max().item()
            is_mtp = ".mtp." in name

            if is_mtp:
                if grad_norm > 0:
                    mtp_nonzero_grads.append((name, grad_norm))
            else:
                if grad_norm > 0:
                    non_mtp_nonzero_grads.append((name, grad_norm))

    # Log the results
    logger.info(
        f"[CI MTP Grad Check] Step {step_id}: "
        f"MTP params with non-zero grad: {len(mtp_nonzero_grads)}, "
        f"non-MTP params with non-zero grad: {len(non_mtp_nonzero_grads)}"
    )

    if non_mtp_nonzero_grads:
        # Log the first few non-MTP params with non-zero gradients for debugging
        for name, grad_norm in non_mtp_nonzero_grads[:5]:
            logger.error(f"[CI MTP Grad Check] Non-MTP param with non-zero grad: {name}, max_grad={grad_norm}")

    assert len(non_mtp_nonzero_grads) == 0, (
        f"Expected all non-MTP parameters to have zero gradients, "
        f"but found {len(non_mtp_nonzero_grads)} with non-zero gradients. "
        f"First few: {non_mtp_nonzero_grads[:5]}"
    )

    # Also verify that MTP params do have gradients (otherwise the test is not valid)
    assert len(mtp_nonzero_grads) > 0, (
        "Expected MTP parameters to have non-zero gradients, but all were zero. "
        "This may indicate the MTP loss is not being computed."
    )


def check_mtp_loss(mtp_loss: float, max_mtp_loss: float = 1.0) -> None:
    """Check that MTP loss is within expected bounds.

    Args:
        mtp_loss: The computed MTP loss value.
        max_mtp_loss: Maximum allowed MTP loss (default: 1.0).

    Raises:
        AssertionError: If MTP loss exceeds the maximum allowed value.
    """
    assert mtp_loss < max_mtp_loss, (
        f"MTP loss {mtp_loss} exceeds maximum allowed value {max_mtp_loss}. "
        "This may indicate an issue with MTP training."
    )
