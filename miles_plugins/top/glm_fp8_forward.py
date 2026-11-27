"""GLM sparse-attention, rotary and indexer operations for FP8 forward."""

import torch


class FP8SparseMLA(torch.autograd.Function):
    """Use authoritative FP8 KV storage and the serving sparse-attention op."""

    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
        from sglang.srt.true_on_policy.dsa.flashmla_fp8 import sparse_from_fp8

        ctx.save_for_backward(q, kv, indices)
        ctx.scaling = scaling
        rows = kv.shape[0]
        padded = kv.new_zeros(((rows + 63) // 64 * 64, 1, 576))
        padded[:rows] = kv
        cache = quantize_k_cache(padded.view(-1, 64, 1, 576))
        output = sparse_from_fp8(q, cache, indices.squeeze(1), scaling, 512)
        return output, None

    @staticmethod
    def backward(ctx, grad_output, grad_lse=None):
        from miles_plugins.top.glm_sparse_mla import sparse_mla_reference_grads
        from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant

        q, kv, indices = ctx.saved_tensors

        def as_stored(k):
            # The cache stores the 512 nope dims in E4M3 (128-groups, FP32 amax scales) and the rope dims in BF16;
            # quantization is straight-through.
            nope = fp8_group_fake_quant(k[:, :512].float(), 128, scale="amax").to(k.dtype)
            return torch.cat([nope, k[:, 512:]], -1)

        return (*sparse_mla_reference_grads(q, kv, indices, ctx.scaling, grad_output, as_stored), None, None)
