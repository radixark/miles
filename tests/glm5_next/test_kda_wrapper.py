import sys

import torch

sys.path.insert(0, ".")

from miles_plugins.models.glm5_next.kda import Glm5NextKDA

HIDDEN_SIZE = 512
NUM_HEADS = 8
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
GATE_LOWER_BOUND = -5.0
RMS_NORM_EPS = 1e-5
CU_SEQLENS = [0, 96, 96 + 160, 96 + 160 + 33]


def test_gate_matches_fla_reference():
    from fla.ops.kda.gate import fused_kda_gate

    torch.manual_seed(0)
    total_tokens = 64
    f = torch.randn(1, total_tokens, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(NUM_HEADS, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(NUM_HEADS * HEAD_DIM, device="cuda", dtype=torch.float32)

    def torch_gate(x, A_log, dt_bias, lower_bound):
        a = A_log.float().exp().view(A_log.numel(), 1)
        return lower_bound * torch.sigmoid(a * (x.float() + dt_bias.float().view(A_log.numel(), -1)))

    f_ref = f.clone().requires_grad_(True)
    f_ours = f.clone().requires_grad_(True)
    ref = torch_gate(f_ref, A_log, dt_bias, GATE_LOWER_BOUND)
    ours = fused_kda_gate(f_ours, A_log, dt_bias, lower_bound=GATE_LOWER_BOUND)

    assert ref.dtype == ours.dtype == torch.float32, (ref.dtype, ours.dtype)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    ours.backward(grad)
    torch.testing.assert_close(f_ours.grad.float(), f_ref.grad.float(), rtol=4e-3, atol=4e-3)
    print("PASS gate matches fla fused_kda_gate (values + grads)")


def _run_module(seed: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    module = Glm5NextKDA(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL_SIZE,
        gate_lower_bound=GATE_LOWER_BOUND,
        rms_norm_eps=RMS_NORM_EPS,
    ).cuda()
    for name, param in module.named_parameters():
        if param.dtype == torch.float32 and name in ("A_log", "dt_bias"):
            continue
        param.data = param.data.to(torch.bfloat16)

    cu_seqlens = torch.tensor(CU_SEQLENS, device="cuda", dtype=torch.int32)
    hidden = torch.randn(1, CU_SEQLENS[-1], HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = module(hidden, cu_seqlens=cu_seqlens)
    out.float().square().mean().backward()

    grads = {name: param.grad.detach().clone() for name, param in module.named_parameters()}
    grads["__input__"] = hidden.grad.detach().clone()
    return out.detach().clone(), grads


def test_forward_backward_smoke():
    out, grads = _run_module(seed=1)
    assert torch.isfinite(out.float()).all(), "non-finite forward output"
    expected = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "conv1d.weight",
        "b_proj.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "A_log",
        "dt_bias",
        "o_norm.weight",
        "o_proj.weight",
        "__input__",
    }
    missing = expected - set(grads)
    assert not missing, f"missing grads: {missing}"
    for name, grad in grads.items():
        assert grad is not None and torch.isfinite(grad.float()).all(), f"non-finite grad: {name}"
    print("PASS fwd/bwd smoke (grads reach all params incl. A_log/dt_bias/conv1d)")


def test_determinism_double_run():
    out_a, grads_a = _run_module(seed=2)
    out_b, grads_b = _run_module(seed=2)
    assert torch.equal(out_a, out_b), "forward output not bit-identical across runs"
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), f"grad not bit-identical across runs: {name}"
    print("PASS determinism double-run (fwd + grads bit-identical)")


if __name__ == "__main__":
    test_gate_matches_fla_reference()
    test_forward_backward_smoke()
    test_determinism_double_run()
