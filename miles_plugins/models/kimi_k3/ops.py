import torch
import torch.nn as nn


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)


class KimiRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        device: torch.device | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_proj: nn.Linear,
    score_norm: KimiRMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + score_norm.eps)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    mixed = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
    return output_norm(mixed)
