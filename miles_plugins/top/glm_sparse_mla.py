"""GLM's sparse MLA: serving's FlashMLA values; gradients from Miles' differentiable sparse
attention (FlashMLA forward, TileLang backward) recomputed over the same selections."""

import torch


class TopSparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        from sglang.srt.layers.attention.dsa.flashmla import flashmla_sparse_bf16

        if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or kv.shape[-1] != 576:
            raise NotImplementedError("[top] GLM sparse MLA requires BF16 query/KV with latent width 576")
        ctx.save_for_backward(q, kv, indices)
        ctx.scaling = scaling
        output = flashmla_sparse_bf16(
            q.contiguous(),
            kv.contiguous(),
            512,
            indices.squeeze(1).contiguous(),
            scaling,
            device_sm_major=torch.cuda.get_device_capability(q.device)[0],
            topk_length=(indices >= 0).sum(dim=-1).reshape(-1).to(torch.int32),
        )
        return output, None

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        q, kv, indices = ctx.saved_tensors
        return (*sparse_mla_reference_grads(q, kv, indices, ctx.scaling, grad_output), None, None)


def sparse_mla_reference_grads(q, kv, indices, scaling, grad_output, kv_transform=None):
    """dq, dkv of softmax(q kv^T) kv[:512] over each row's selected rows (-1 is unselected)."""
    from miles.kernels.attention.dsa import sparse_attention

    from miles_plugins.top.kernels.surrogate import reference_mode

    qd = q.detach().requires_grad_(q.requires_grad)
    kd = kv.detach().requires_grad_(kv.requires_grad)
    idx = indices.reshape(q.shape[0], -1)
    width = (idx.shape[-1] + 63) // 64 * 64
    idx = torch.nn.functional.pad(idx, (0, width - idx.shape[-1]), value=-1).int()
    with torch.enable_grad(), reference_mode():
        k = kd.reshape(kd.shape[0], -1) if kv_transform is None else kv_transform(kd.reshape(kd.shape[0], -1))
        out = sparse_attention(qd[None], k[None, :, None], idx[None, :, None], scaling, d_v=512)[0]
        want = [t for t in (qd, kd) if t.requires_grad]
        got = torch.autograd.grad(out, want, grad_output.reshape(out.shape).to(out.dtype)) if want else ()
    got = iter(got)
    return (next(got) if qd.requires_grad else None, next(got) if kd.requires_grad else None)
