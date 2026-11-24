"""miles.kernels.neuron.routed_experts against fp64 per-token SwiGLU experts, as close as bf16 inputs allow.

Run on a trn host: python -m pytest tests/manual/neuron/test_routed_experts.py
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_neuronx")

from miles.kernels.neuron.routed_experts import routed_experts  # noqa: E402

# the kernels take a multiple of 8192 tokens
NUM_TOKENS, DIM, HIDDEN, NUM_EXPERTS, TOP_K = 8192, 512, 256, 8, 2
_TOLERANCE_FACTOR = 2.5


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    got = got.detach().cpu().double()
    return ((got - want).norm() / want.norm()).item()


def _dense(x_TD, gate_up_EDTF, down_EFD, scores_TK, expert_ids_TK):
    out_TD = torch.zeros_like(x_TD)
    for k in range(TOP_K):
        experts = expert_ids_TK[:, k]
        gate_up_T2F = torch.einsum("td,tdsf->tsf", x_TD, gate_up_EDTF[experts])
        hidden_TF = F.silu(gate_up_T2F[:, 0]) * gate_up_T2F[:, 1]
        out_TD = out_TD + scores_TK[:, k : k + 1] * torch.einsum("tf,tfd->td", hidden_TF, down_EFD[experts])
    return out_TD


def test_matches_per_token_swiglu_experts():
    torch.manual_seed(0)
    x = torch.randn(NUM_TOKENS, DIM) * 0.5
    gate_up = torch.randn(NUM_EXPERTS, DIM, 2, HIDDEN) * 0.05
    down = torch.randn(NUM_EXPERTS, HIDDEN, DIM) * 0.05
    scores, expert_ids = torch.topk(torch.softmax(torch.randn(NUM_TOKENS, NUM_EXPERTS), -1), TOP_K, dim=-1)
    grad_out = torch.randn(NUM_TOKENS, DIM)

    def dense(inputs):
        out = _dense(*inputs, expert_ids)
        out.backward(grad_out.double())
        return [out, *(t.grad for t in inputs)]

    ref = dense([t.double().requires_grad_() for t in (x, gate_up, down, scores)])
    bf16_inputs = dense(
        [t.bfloat16().double().requires_grad_() for t in (x, gate_up, down)] + [scores.double().requires_grad_()]
    )

    device = torch.device("neuron", 0)
    inputs = [t.to(device, torch.bfloat16).requires_grad_() for t in (x, gate_up, down)]
    scores_d = scores.to(device).requires_grad_()
    masked_scores = torch.zeros(NUM_TOKENS, NUM_EXPERTS, device=device).scatter(1, expert_ids.to(device), scores_d)
    x_d, gate_up_d, down_d = inputs
    out = routed_experts(x_d, masked_scores, gate_up_d, down_d, TOP_K)
    out.float().backward(grad_out.to(device))
    kernel = [out, *(t.grad for t in inputs), scores_d.grad]

    for name, got, floor, want in zip(
        ["out", "dx", "dgate_up", "ddown", "dscores"], kernel, bf16_inputs, ref, strict=True
    ):
        error, floor_error = _relative_error(got, want), _relative_error(floor, want)
        assert error <= _TOLERANCE_FACTOR * floor_error, f"{name}: {error:.4%} vs bf16-input floor {floor_error:.4%}"
