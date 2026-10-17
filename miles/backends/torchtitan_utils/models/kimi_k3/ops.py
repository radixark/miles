"""Training operations for Kimi K3."""

from dataclasses import dataclass

import torch
from fla.modules import ShortConvolution
from fla.modules.conv.causal_conv1d import causal_conv1d
from torch import nn
from torch.distributed.tensor import DTensor, Partial
from torchtitan.models.common import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.protocols.module import Module


def linear(in_features, out_features):
    return Linear.Config(in_features=in_features, out_features=out_features).build()


def situ(gate, up, beta=4.0, linear_beta=25.0):
    gate_float, up_float = local_tensor(gate).float(), local_tensor(up).float()
    value = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    output = (value * linear_beta * torch.tanh(up_float / linear_beta)).to(gate.dtype)
    return wrap_like(output, gate)


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x):
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return value.to(x.dtype) * self.weight.to(x.dtype)


class ShortConv(ShortConvolution, Module):
    """FLA convolution with TorchTitan's parameter initialization protocol."""

    def forward(self, x, *, cu_seqlens=None, output_final_state=False, cp_context=None):
        if not isinstance(x, DTensor) and cp_context is None:
            return super().forward(x, cu_seqlens=cu_seqlens, output_final_state=output_final_state)
        if output_final_state:
            raise ValueError("K3 training does not export convolution state")
        output, _ = causal_conv1d(
            x=local_tensor(x),
            weight=local_tensor(self.weight).squeeze(1),
            bias=local_tensor(self.bias),
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            output_final_state=False,
        )
        return wrap_like(output, x), None


def local_tensor(tensor, *, partial_grad=False):
    if not isinstance(tensor, DTensor):
        return tensor
    grad_placements = None
    if partial_grad:
        grad_placements = tuple(
            Partial() if name == "tp" else placement
            for name, placement in zip(tensor.device_mesh.mesh_dim_names, tensor.placements, strict=True)
        )
    return tensor.to_local(grad_placements=grad_placements)


def wrap_like(tensor, reference):
    if isinstance(reference, DTensor):
        return DTensor.from_local(tensor, reference.device_mesh, reference.placements, run_check=False)
    return tensor


class GatedRMSNorm(RMSNorm):
    def forward(self, x, gate):
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float() * gate.float().sigmoid()).to(x.dtype)


class DenseMLP(Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.gate_proj = linear(dim, intermediate)
        self.up_proj = linear(dim, intermediate)
        self.down_proj = linear(intermediate, dim)

    def forward(self, x):
        return self.down_proj(situ(self.gate_proj(x), self.up_proj(x)))


class SituExperts(GroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def reset_parameters(self):
        for weight in self.parameters(recurse=False):
            nn.init.normal_(weight, std=0.02)

    def forward(self, x, counts):
        w1, w2, w3 = [local_tensor(w) for w in (self.w1_EFD, self.w2_EDF, self.w3_EFD)]
        offsets = counts.cumsum(0, dtype=torch.int32)
        gate = self._grouped_mm(A=x.bfloat16(), B_t=w1.bfloat16().transpose(-2, -1), offs=offsets)
        up = self._grouped_mm(A=x.bfloat16(), B_t=w3.bfloat16().transpose(-2, -1), offs=offsets)
        return self._grouped_mm(A=situ(gate, up), B_t=w2.bfloat16().transpose(-2, -1), offs=offsets).to(x.dtype)


class KimiTokenDispatcher(AllToAllTokenDispatcher):
    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    def combine(self, routed_output_RD, metadata, x_TD):
        if self.ep_mesh is not None:
            routed_output_RD = self._unpermute(routed_output_RD, metadata.input_shape, metadata.permuted_indices)
            routed_output_RD = self._combine_token_exchange(
                routed_output_RD, self.ep_mesh.get_group(), metadata.input_splits, metadata.output_splits
            )
        # K3 accumulates weighted expert outputs in FP32, then rounds once to BF16.
        weighted = routed_output_RD.float() * metadata.topk_scores_experts_sorted_N.unsqueeze(-1)
        combined = torch.zeros_like(x_TD, dtype=torch.float32)
        indices = metadata.token_indices_experts_sorted_N.unsqueeze(-1).expand_as(weighted)
        return deterministic_scatter_add(combined, indices, weighted).to(x_TD.dtype)


def attention_residual(prefix, bank, norm, proj):
    values = torch.cat((bank, prefix.unsqueeze(-2)), dim=-2)
    values_float = values.float()
    keys = values_float * torch.rsqrt(values_float.square().mean(-1, keepdim=True) + norm.eps)
    scores = (keys * (norm.weight.float() * proj.weight.squeeze(0).float())).sum(-1)
    return (scores.softmax(-1).unsqueeze(-2) @ values_float).squeeze(-2).to(values.dtype)


def packed_metadata(tokens, positions, cu_seqlens=None, *, query_indices=None):
    """Flatten batches for FLA, retaining document boundaries in packed rows."""
    batch, length = tokens.shape
    if cu_seqlens is not None:
        torch._assert_async(cu_seqlens[-1] == tokens.numel(), "K3 boundaries must include all padded tokens")
        starts = cu_seqlens[:-1].long()
    elif positions is None:
        starts = torch.arange(0, batch * length, length, device=tokens.device)
    else:
        starts = (positions.reshape(-1) == 0).nonzero().flatten()
    if cu_seqlens is None:
        cu_seqlens = torch.cat((starts, starts.new_tensor([tokens.numel()]))).to(torch.int32)
    document = torch.zeros(tokens.numel(), dtype=torch.int32, device=tokens.device)
    document[starts] = 1
    document = document.cumsum(0)
    keys = torch.arange(tokens.numel(), device=tokens.device)
    queries = keys if query_indices is None else query_indices
    causal = queries[:, None] >= keys[None, :]
    return cu_seqlens, causal & (document[queries, None] == document[None, :])
