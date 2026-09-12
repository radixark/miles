"""Gather PLE rows straight out of pinned host memory.

Mirrors sglang's ``_gather_ple_embedding_from_pinned_kernel``: one program per
(token, hash-head) row, reading through a raw host pointer so the GPU pulls the
row over the coherent link instead of staging the table into HBM. The table is
51.2 B parameters / 102.4 GB, so it never goes to device -- and on GB300 the
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_ple_rows_from_pinned(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    row_start,
    row_end,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= row_start) & (global_idx < row_end)
    local_idx = tl.where(in_range, global_idx - row_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(ptr + local_idx * embedding_dim + offsets, mask=mask, other=0.0)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values.to(tl.bfloat16), 0.0),
        mask=mask,
    )


def gather_ple_rows(
    host_table: torch.Tensor,
    ids: torch.Tensor,
    row_start: int,
    row_end: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``ids`` ``[...]`` int64 on device -> ``[..., embedding_dim]`` bf16 on device."""
    assert host_table.device.type == "cpu", "PLE table must stay on the host"
    assert host_table.is_pinned(), "PLE table must be pinned for the kernel to read it"
    assert host_table.dtype == torch.bfloat16, f"expected bf16 table, got {host_table.dtype}"
    embedding_dim = host_table.shape[-1]

    shape = (*ids.shape, embedding_dim)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=ids.device)
    else:
        assert tuple(out.shape) == shape and out.dtype == torch.bfloat16

    flat = ids.reshape(-1)
    if flat.numel():
        _gather_ple_rows_from_pinned[(flat.numel(),)](
            host_table.data_ptr(),
            flat.contiguous(),
            out.view(-1, embedding_dim),
            embedding_dim=embedding_dim,
            row_start=row_start,
            row_end=row_end,
            BLOCK_D=triton.next_power_of_2(embedding_dim),
        )
    return out
