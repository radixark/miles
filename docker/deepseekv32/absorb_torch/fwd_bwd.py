import torch
from torch.nn import functional as F
import torch.distributed as dist

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


@torch.no_grad
def eager_attn_fwd(q, k, v, attn_bias, sinks, scale, dropout):
    """Forward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q = einops.rearrange(q, 'b s h d -> b h s d')
    _k = einops.rearrange(k, 'b s h d -> b h d s')
    _v = einops.rearrange(v, 'b s h d -> b h s d')

    # Compute attention weights
    attn_w = torch.matmul(_q, _k) * scale
    attn_w = attn_w + attn_bias

    # Add sinks to attention weights
    if sinks is None:
        logits = attn_w
    else:
        _sinks = sinks.reshape(1, h, 1, 1).expand(b, -1, sq, 1)
        logits = torch.cat([attn_w, _sinks], dim=-1)

    # Compute attention scores
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink

    # Compute attention output
    attn_output = torch.matmul(attn_w, _v)
    attn_output = einops.rearrange(attn_output, 'b h s d -> b s h d')
    attn_output = attn_output.contiguous()

    return attn_output, probs


@torch.no_grad
def eager_attn_bwd(q, kv, attn_bias, sinks, scale, dim_short, dropout, attn_output, probs, grad_output):
    """Backward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    _, sk, _, _ = kv.shape
    k = kv
    v = kv[:,:,:,:dim_short]
    q_tail = q[:,:,:,dim_short:]
    _q_tail_T = einops.rearrange(q_tail, 'b s h d -> b h d s').contiguous()
    _q_T = einops.rearrange(q, 'b s h d -> b h d s')
    _k_T = einops.rearrange(k, 'b s h d -> b h s d')
    _v_T = einops.rearrange(v, 'b s h d -> b h d s')

    # Backward pass for score @ value
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink
    grad_output = einops.rearrange(grad_output, 'b s h d -> b h s d')
    attn_w_T = einops.rearrange(attn_w, ' b h sq sk -> b h sk sq')
    grad__v = torch.matmul(attn_w_T, grad_output).contiguous() # b h sk d
    grad_attn_w = torch.matmul(grad_output, _v_T).contiguous() # b h s d  || b h d sk -> b h s sk
 
    # Backward pass for softmax
    if sinks is None:
        grad_probs = grad_attn_w
    else:
        dummy = torch.zeros((b, h, sq, 1), device=q.device, dtype=q.dtype)
        grad_probs = torch.cat([grad_attn_w, dummy], dim=3)
    del grad_attn_w
    grad_logits = torch._softmax_backward_data(
        grad_probs, probs, -1, probs.dtype
    )  # [b, h, sq, sk+1]

    # Backward pass for adding sinks
    if sinks is None:
        grad_sinks = None
        grad_attn_w = grad_logits
    else:
        grad__sinks = grad_logits[:, :, :, -1]  # [b, h, sq]
        grad_sinks = einops.rearrange(grad__sinks, 'b h s -> h (b s)').sum(-1)
        grad_attn_w = grad_logits[:, :, :, :-1].contiguous()  # [b, h, sq, sk]

    # Backward pass for q @ K^T
    grad_attn_w *= scale
    grad__q = torch.matmul(grad_attn_w, _k_T).contiguous()
    grad__k = torch.matmul(_q_T, grad_attn_w).contiguous() # b h d sk

    grad__k_T = grad__k.transpose(2, 3).contiguous() # b h sk d
    grad__kv = torch.zeros((b, h, sk, d), device=q.device, dtype=q.dtype) # b h sk d
    grad__kv[:,:,:,:dim_short] = grad__v + grad__k_T[:,:,:,:dim_short]
    grad__kv[:,:,:,dim_short:] = torch.matmul(_q_tail_T, grad_attn_w).contiguous().transpose(2, 3).contiguous() # b h sk d

    # Rearrange grads to (b, s, h, d)
    grad__kv = grad__kv.transpose(1, 2).contiguous()
    grad_q = einops.rearrange(grad__q, 'b h s d -> b s h d')
    return grad_q, grad__kv, grad_sinks