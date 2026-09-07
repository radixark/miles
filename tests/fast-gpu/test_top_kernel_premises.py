"""The kernel-level premises true-on-policy parity rests on.

Each of these can break SILENTLY: the run still succeeds, the metric just moves off zero. They are
not regression tests for fixed bugs -- they are the assumptions that make the delegation valid.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

H = 2560
EPS = 1e-6


@pytest.fixture
def sgl_norm(monkeypatch):
    from sglang.srt.layers import layernorm
    from sglang.srt.true_on_policy import config

    monkeypatch.setattr(
        config,
        "_get_global_server_args",
        lambda: SimpleNamespace(true_on_policy_contract="true_on_policy_v1"),
    )

    def refuse_fallback(*args, **kwargs):
        raise AssertionError("TOP norm must use stock fused dispatch")

    monkeypatch.setattr(layernorm.RMSNorm, "forward_native", refuse_fallback)
    monkeypatch.setattr(layernorm, "rms_norm_batch_invariant", refuse_fallback)
    n = layernorm.RMSNorm(H, eps=EPS).cuda().to(torch.bfloat16)
    with torch.no_grad():
        n.weight.copy_(torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0)
    return n


def _out(v):
    return v[0] if isinstance(v, tuple) else v


@pytest.mark.parametrize("paired", [False, True])
def test_both_engines_resolve_the_same_norm(sgl_norm, paired):
    """The trainer has sglang's batch-invariant mode off and the rollout has it on. Under the
    contract both must still reach the SAME kernel -- that is what makes the delegation exact,
    and it is invisible if it breaks: the run succeeds and the metric moves off zero.
    """
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    assert not sgl_norm.cast_x_before_out_mul, "TOP must follow stock SGLang rounding"
    x = torch.randn(207, H, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)

    def forward():
        return sgl_norm.forward_cuda(x.clone(), r.clone()) if paired else sgl_norm.forward_cuda(x.clone())

    was_enabled = is_batch_invariant_mode_enabled()
    try:
        disable_batch_invariant_mode()
        trainer = forward()
        enable_batch_invariant_mode()
        rollout = forward()
        if paired:
            assert all(torch.equal(a, b) for a, b in zip(trainer, rollout))
        else:
            assert torch.equal(trainer, rollout)
    finally:
        if not was_enabled:
            disable_batch_invariant_mode()


@pytest.mark.parametrize("role", ["input_layernorm", "pre_mlp_layernorm", "final_layernorm"])
def test_packed_trainer_norm_matches_rollout_add_norm(sgl_norm, role):
    """Both engines pass the original operands to fused add+norm, with no early BF16 add."""
    pytest.importorskip("megatron.core")
    from miles_plugins.top.spec import TopRMSNorm

    config = SimpleNamespace(hidden_size=H, pipeline_hidden_size=2 * H)
    trainer_norm = TopRMSNorm(config, H, eps=EPS, role=role).cuda().bfloat16()
    with torch.no_grad():
        trainer_norm.weight.copy_(sgl_norm.weight)
    x = torch.randn(207, H, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(207, H, device="cuda", dtype=torch.bfloat16)

    rollout_out, rollout_residual = sgl_norm.forward_cuda(x.clone(), r.clone())
    packed = torch.cat((x, r), dim=-1)
    trainer = trainer_norm(packed)
    assert torch.equal(rollout_out, _out(trainer))
    if role != "final_layernorm":
        assert torch.equal(rollout_residual, trainer[1])
    assert torch.equal(packed, torch.cat((x, r), dim=-1))


@pytest.mark.parametrize("n_tok", [1, 8, 207])
def test_norm_is_batch_invariant(sgl_norm, n_tok):
    """A token's norm must not depend on how many tokens share the launch: the rollout sees
    prefill, decode and recompute batches; the trainer sees the whole sequence."""
    big = torch.randn(512, H, device="cuda", dtype=torch.bfloat16)
    ref = _out(sgl_norm.forward_cuda(big.clone()))
    got = _out(sgl_norm.forward_cuda(big[:n_tok].clone()))
    assert torch.equal(ref[:n_tok], got)


@pytest.mark.parametrize("splits", [2, 4, 8])
def test_matmul_tp_inv_is_split_invariant(splits):
    """The row linear gives each rank a K-slice and sums. That is only equal to the whole GEMM if
    the kernel is invariant to how K is split -- the premise of the entire row-linear delegation."""
    from sglang.srt.tp_invariant_ops import matmul_tp_inv

    torch.manual_seed(0)
    M, K, N = 512, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    full = matmul_tp_inv(a, b)
    step = K // splits
    parts = [
        matmul_tp_inv(a[:, i * step : (i + 1) * step].contiguous(), b[i * step : (i + 1) * step, :].contiguous())
        for i in range(splits)
    ]
    # summed the way the row linear sums them: the fixed pairwise tree, not left-to-right
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    assert torch.equal(full, parts[0])
