"""The SGLang expert activation, with an analytic FP32 backward."""

import torch
from sglang.kernels.ops.activation.activation import silu_and_mul, silu_and_mul_accurate


class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, accurate=True):
        out = torch.empty(
            gate_up.shape[0],
            gate_up.shape[1] // 2,
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
        ctx.save_for_backward(gate_up)
        # Routed experts: bitwise serving's TOP expert SiLU (decode: precise masked; prefill: the
        # accurate CUDA build). The shared expert is a dense MLP: serving's default fast build.
        (silu_and_mul_accurate if accurate else silu_and_mul)(gate_up, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (gate_up,) = ctx.saved_tensors
        gate, up = gate_up.float().chunk(2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        grad = grad_output.float()
        dgate = grad * up * sigmoid * (1 + gate * (1 - sigmoid))
        dup = grad * gate * sigmoid
        return torch.cat((dgate, dup), dim=-1).to(gate_up.dtype), None
