import torch
from .ref_kernel import act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    y, scale = act_quant(x.contiguous(), block_size, "ue8m0")
    y = y.unflatten(-1, (-1, block_size)).float() * scale.unsqueeze(-1)
    return y.flatten(-2).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None

fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
