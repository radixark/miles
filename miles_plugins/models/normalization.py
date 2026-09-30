import torch


def rms_norm(inputs: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Recompute the RMSNorm fused into TELayerNormColumnParallelLinear (eager, fp32 internals)."""
    x = inputs.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * gamma.float()).to(inputs.dtype)
