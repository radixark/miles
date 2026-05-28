import torch

# Lazy-load tilelang GPU kernels (see comment in indexer.py).
def sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, **kwargs):
    if hasattr(q, "is_npu") and q.is_npu:
        from ._npu.sparse_mla import npu_sparse_mla_fwd_interface
        return npu_sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, **kwargs)
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface as _gpu
    return _gpu(q, kv, indices, sm_scale=sm_scale, **kwargs)


def sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None, **kwargs):
    if hasattr(q, "is_npu") and q.is_npu:
        from ._npu.sparse_mla import npu_sparse_mla_bwd
        return npu_sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale, **kwargs)
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd as _gpu
    return _gpu(q, kv, o, do, indices, lse, sm_scale=sm_scale, **kwargs)


class SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

        # Save tensors for backward pass
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling)

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None
