from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor


_NORM_GRAD_DEBUG_COUNTER = 0


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return -1


def _tensor_stats(tensor: Tensor) -> dict[str, Any]:
    flat = tensor.detach().view(-1)
    finite = torch.isfinite(flat)
    nan = torch.isnan(flat)
    inf = torch.isinf(flat)
    stats = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": flat.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(nan.sum().item()),
        "inf": int(inf.sum().item()),
        "max_abs_finite": None,
        "min_finite": None,
        "max_finite": None,
    }
    if stats["finite"] > 0:
        finite_values = flat[finite].float()
        stats.update(
            {
                "max_abs_finite": float(finite_values.abs().max().item()),
                "min_finite": float(finite_values.min().item()),
                "max_finite": float(finite_values.max().item()),
            }
        )
    return stats


def register_norm_grad_debug(tensor: Tensor, *, layer_number: int, name: str) -> None:
    debug_dir = os.environ.get("MILES_NORM_BACKWARD_DEBUG_DIR")
    if not debug_dir or not torch.is_tensor(tensor) or not tensor.requires_grad:
        return

    debug_all = os.environ.get("MILES_NORM_BACKWARD_DEBUG_ALL") == "1"
    threshold = float(os.environ.get("MILES_NORM_BACKWARD_DEBUG_THRESHOLD", "1e20"))
    debug_key = f"{layer_number}:{name}"

    def _hook(grad: Tensor) -> Tensor:
        stats = _tensor_stats(grad)
        max_abs_finite = stats.get("max_abs_finite")
        should_dump = debug_all or stats["nan"] or stats["inf"]
        if max_abs_finite is not None and max_abs_finite >= threshold:
            should_dump = True
        if not should_dump:
            return grad

        rank = _rank()
        if not debug_all:
            seen = getattr(register_norm_grad_debug, "_seen", set())
            seen_key = (rank, debug_key)
            if seen_key in seen:
                return grad
            seen.add(seen_key)
            register_norm_grad_debug._seen = seen

        global _NORM_GRAD_DEBUG_COUNTER
        counter = _NORM_GRAD_DEBUG_COUNTER
        _NORM_GRAD_DEBUG_COUNTER += 1
        stats.update({"rank": rank, "layer_number": layer_number, "name": name})
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"rank_{rank}_layer_{layer_number}_{counter:05d}_{name}_pid_{os.getpid()}.pt")
        torch.save(stats, path)
        print(f"[MILES_NORM_BACKWARD_DEBUG] {stats} wrote {path}", flush=True)
        return grad

    tensor.register_hook(_hook)


def register_activation_grad_debug(owner: object, tensor: Tensor, *, layer_number: int, name: str) -> None:
    debug_dir = os.environ.get("MILES_ACTIVATION_GRAD_DEBUG_DIR")
    if not debug_dir or not torch.is_tensor(tensor) or not tensor.requires_grad:
        return

    debug_key = f"{layer_number}:{name}"

    def _hook(grad: Tensor) -> Tensor:
        if torch.isfinite(grad).all().item():
            return grad

        seen = getattr(owner, "_activation_grad_debug_seen", set())
        if debug_key in seen:
            return grad
        seen.add(debug_key)
        owner._activation_grad_debug_seen = seen

        rank = _rank()
        stats = _tensor_stats(grad)
        stats.update({"rank": rank, "layer_number": layer_number, "name": name})
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"rank_{rank}_layer_{layer_number}_{name}_pid_{os.getpid()}.pt")
        torch.save(stats, path)
        print(f"[MILES_ACTIVATION_GRAD_DEBUG] {stats} wrote {path}", flush=True)
        return grad

    tensor.register_hook(_hook)


def dump_param_and_grad_buffer_debug(
    bucket_group: object,
    *,
    bucket_index: int,
    grad_norm: Tensor,
    grad_debug_dir: str,
) -> None:
    bucket = bucket_group.buckets[bucket_index]
    dumped_buckets = getattr(bucket_group, "_grad_debug_dumped_buckets", set())
    dump_key = (bucket.bucket_id, bucket_index)
    if dump_key in dumped_buckets:
        return
    dumped_buckets.add(dump_key)
    bucket_group._grad_debug_dumped_buckets = dumped_buckets

    rank = _rank()
    params = []
    for param in bucket.params_list:
        start, end = bucket.param_to_index[param]
        grad_slice = bucket.grad_data.view(-1)[start:end].view(param.shape)
        params.append(
            {
                "name": bucket.param_to_name.get(param, "<unknown>"),
                "shape": tuple(param.shape),
                "requires_grad": bool(param.requires_grad),
                "grad_slice": _tensor_stats(grad_slice),
                "main_grad": (
                    _tensor_stats(param.main_grad)
                    if hasattr(param, "main_grad") and param.main_grad is not None
                    else None
                ),
            }
        )

    os.makedirs(grad_debug_dir, exist_ok=True)
    path = os.path.join(
        grad_debug_dir,
        f"rank_{rank}_bucket_{bucket.bucket_id}_idx_{bucket_index}_pid_{os.getpid()}.pt",
    )
    torch.save(
        {
            "rank": rank,
            "bucket_id": bucket.bucket_id,
            "bucket_index": bucket_index,
            "grad_norm": float(grad_norm.detach().float().cpu().item()),
            "bucket_grad": _tensor_stats(bucket.grad_data),
            "params": params,
        },
        path,
    )
    print(f"[MILES_GRAD_DEBUG] wrote nonfinite grad dump to {path}", flush=True)
