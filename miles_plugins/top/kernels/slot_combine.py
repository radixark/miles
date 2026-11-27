"""Weight unrounded expert rows inside SGLang's FP32 slot-order accumulation."""

import torch
from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather

from miles_plugins.top.kernels.receipt import validate_receipt

# Slots per backward pass: four bound the FP32 temporaries to ~0.2 GB per 2.3k-token chunk at GLM width.
_SLOTS_PER_PASS = 4


class _SlotOrderCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, weights, ids, index):
        out = torch.empty(ids.shape[0], rows.shape[-1], device=rows.device, dtype=rows.dtype)
        # rows may view the transport's padded payload; saving the dense copy the gather reads frees the padding.
        rows = rows.contiguous()
        ctx.save_for_backward(rows, weights, ids, index)
        ep_gather(rows, ids, weights, index, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        rows, weights, ids, index = ctx.saved_tensors
        num_rows, num_tokens, topk = rows.shape[0], ids.shape[0], ids.shape[1]
        if num_rows == 0:
            return torch.zeros_like(rows), torch.zeros_like(weights), None, None
        # No boolean masks (each is a host sync): an inactive slot reads row 0 with weight 0 and writes the one
        # scratch row past the end. Receipt validation guarantees each live row occurs exactly once, so only the
        # discarded scratch row can take more than one write.
        active = ids >= 0
        read = torch.where(active, index.long(), 0)
        write = torch.where(active, index.long(), num_rows)
        weight = torch.where(active, weights.float(), 0.0)
        grad = grad_output.float()
        drows = rows.new_zeros(num_rows + 1, rows.shape[-1])
        dweights = torch.zeros_like(weights)
        for k0 in range(0, topk, _SLOTS_PER_PASS):
            k1 = min(k0 + _SLOTS_PER_PASS, topk)
            drows.index_copy_(
                0,
                write[:, k0:k1].reshape(-1),
                (grad[:, None, :] * weight[:, k0:k1, None]).to(rows.dtype).flatten(0, 1),
            )
            picked = rows.index_select(0, read[:, k0:k1].reshape(-1)).view(num_tokens, k1 - k0, -1).float()
            dot = torch.bmm(picked, grad[:, :, None]).squeeze(-1)
            dweights[:, k0:k1] = torch.where(active[:, k0:k1], dot, 0.0).to(weights.dtype)
        return drows[:num_rows], dweights, None, None


def combine_with_receipt(rows, receipt, *, scale=None, router_topk_weights=None):
    """Combine the delivered rows in slot order. ``scale`` (routed scaling) multiplies the router weights; the
    transport's receipts compared the delivered, unscaled weights with these bytes."""
    validate_receipt(receipt, num_rows=rows.shape[0])
    # The gradient goes through the router's own tensor, which holds the receipt's values.
    weights = router_topk_weights
    if weights is None:
        weights = receipt.topk_weights
    if scale is not None:
        weights = weights.float() * scale
    return _SlotOrderCombine.apply(rows, weights, receipt.topk_ids, receipt.input_index)
