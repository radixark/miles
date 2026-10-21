"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Cake forward of the chunked KDA training step: parity with flash-linear-attention v0.5.2 ``chunk_kda``'s
forward (output and saved set), bit-determinism, the saved-set contract of the deterministic backward, the
forward -> deterministic-backward chain against fla's autograd, and the fp32 state carriers.

Needs a Blackwell (SM100a / SM103a) device with flash-linear-attention and an ``nvcc`` toolchain for the
first-use build; skips otherwise.  Tolerances are the contract's (``atol = rtol = 1e-2`` on bf16 outputs and
gradients); nothing here is relaxed to make a row pass.
"""

import pytest
import torch

fla = pytest.importorskip("fla", reason="flash-linear-attention (reference) is not installed")
from fla.modules.l2norm import l2norm_fwd  # noqa: E402
from fla.ops.common.gate import fused_beta_sigmoid_fwd  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402
from fla.ops.kda.chunk_fwd import chunk_kda_fwd  # noqa: E402
from fla.ops.utils.index import prepare_chunk_indices  # noqa: E402

from miles_plugins.models.kda_chunk_train import chunk_kda_backward, chunk_kda_forward  # noqa: E402
from miles_plugins.models.kda_chunk_train.forward import SAVED_SET  # noqa: E402

LOWER_BOUND = -5.0
CHUNK = 64
D = 128
GRADS = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "dt_bias")
TOL = dict(atol=1e-2, rtol=1e-2)


def _requires_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3)}:
        pytest.skip("SM100a / SM103a is required")


def _inputs(batch, seq_len, heads, value_heads, *, seed, packed=False):
    """``packed=<list of lengths>``: unequal packed sequences in one row (``batch`` / ``seq_len`` ignored)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    rand = lambda *shape: torch.rand(*shape, generator=gen, dtype=torch.float32, device=dev)  # noqa: E731
    randn = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32, device=dev)  # noqa: E731
    lengths = list(packed) if isinstance(packed, (list, tuple)) else None
    total, b = (sum(lengths), 1) if lengths is not None else (seq_len, batch)
    cu = None
    if lengths is not None:
        cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], dtype=torch.int32, device=dev)
    return dict(
        q=(rand(b, total, heads, D) - 0.5).to(torch.bfloat16),
        k=(rand(b, total, heads, D) - 0.5).to(torch.bfloat16),
        v=(rand(b, total, value_heads, D) - 0.5).to(torch.bfloat16),
        g=(randn(b, total, value_heads, D) * 0.1).to(torch.bfloat16),
        beta=randn(b, total, value_heads).to(torch.bfloat16),  # logits
        A_log=torch.log(rand(value_heads) + 1.0),
        dt_bias=randn(value_heads * D),
        do=randn(b, total, value_heads, D).to(torch.bfloat16),
        cu_seqlens=cu,
        cu_seqlens_cpu=None if cu is None else cu.cpu().to(torch.int64),
        scale=D**-0.5,
    )


def _fla_forward(inp):
    """fla's forward as the deterministic drop-in runs it: ``o`` and the saved set (beta = fla's fused sigmoid)."""
    with torch.no_grad():
        q_norm, q_rstd = l2norm_fwd(inp["q"])
        k_norm, k_rstd = l2norm_fwd(inp["k"])
        beta = fused_beta_sigmoid_fwd(inp["beta"], 1.0)
        cu = inp["cu_seqlens"]
        chunk_indices = None if cu is None else prepare_chunk_indices(cu, CHUNK, cu_seqlens_cpu=inp["cu_seqlens_cpu"])
        out = chunk_kda_fwd(
            q=q_norm, k=k_norm, v=inp["v"], g=inp["g"], beta=beta, scale=inp["scale"], initial_state=None,
            output_final_state=False, cu_seqlens=cu, cu_seqlens_cpu=inp["cu_seqlens_cpu"], chunk_indices=chunk_indices,
            chunk_size=CHUNK, safe_gate=True, lower_bound=LOWER_BOUND, use_gate_in_kernel=True, A_log=inp["A_log"],
            dt_bias=inp["dt_bias"], state_v_first=True,
        )
    return dict(o=out[0], q_norm=q_norm, k_norm=k_norm, q_rstd=q_rstd, k_rstd=k_rstd, beta=beta, Aqk=out[3], Akk=out[4])


def _fla_autograd(inp):
    """fla ``chunk_kda`` gradients with respect to the raw operands and the beta logits."""
    leaves = {n: inp[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")}
    out = chunk_kda(
        leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], scale=inp["scale"], output_final_state=False,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
        safe_gate=True, lower_bound=LOWER_BOUND, chunk_size=CHUNK, cu_seqlens=inp["cu_seqlens"],
        cu_seqlens_cpu=inp["cu_seqlens_cpu"], A_log=leaves["A_log"], dt_bias=leaves["dt_bias"],
    )
    out = out[0] if isinstance(out, tuple) else out
    order = ("q", "k", "v", "beta", "g", "A_log", "dt_bias")
    grads = torch.autograd.grad(out, [leaves[n] for n in order], grad_outputs=inp["do"])
    return out.detach(), dict(zip(GRADS[:3] + ("dbeta", "dg", "dA_log", "dt_bias"), [g.detach() for g in grads], strict=True))


def _cake_forward(inp, **extra):
    return chunk_kda_forward(
        q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"], beta_logits=inp["beta"], A_log=inp["A_log"], dt_bias=inp["dt_bias"],
        scale=inp["scale"], lower_bound=LOWER_BOUND, cu_seqlens=inp["cu_seqlens"], cu_seqlens_cpu=inp["cu_seqlens_cpu"], **extra,
    )


def _fla_defined_aqk(rows: int, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean ``[1, rows, 1, 64]`` mask of the Aqk entries fla's forward defines: row ``t`` of a chunk holds its 64
    chunk-local columns ``s``; the 16x16 block ``(i, j)`` with ``j > i`` (``s // 16 > t // 16``) is never stored by fla.
    Chunks are counted from each sequence's own start: with packed ``cu_seqlens`` the chunk-local row of token ``t`` of
    the sequence starting at ``bos`` is ``(t - bos) % 64`` (fla's varlen chunking, Cake's ``chunk_bos``/``chunk_len``)."""
    t = torch.arange(rows, device="cuda")
    if cu_seqlens is not None:
        cu = cu_seqlens.to(device="cuda", dtype=torch.int64)
        t = t - cu[torch.searchsorted(cu, t, right=True) - 1]
    t = t % CHUNK
    s = torch.arange(CHUNK, device="cuda")
    return ((s[None, :] // 16) <= (t[:, None] // 16))[None, :, None, :]


def _backward_on(inp, fwd):
    return chunk_kda_backward(
        q_norm=fwd["q_norm"], k_norm=fwd["k_norm"], q_rstd=fwd["q_rstd"], k_rstd=fwd["k_rstd"], v=inp["v"], g=inp["g"],
        beta=fwd["beta"], beta_logits=inp["beta"], A_log=inp["A_log"], dt_bias=inp["dt_bias"], Aqk=fwd["Aqk"], Akk=fwd["Akk"],
        do=inp["do"], scale=inp["scale"], lower_bound=LOWER_BOUND, cu_seqlens=inp["cu_seqlens"], cu_seqlens_cpu=inp["cu_seqlens_cpu"],
    )


SHAPES = [
    pytest.param(1, 2048, 12, 12, False, id="t2048_h12"),  # Kimi-K3 trainer heads per TP rank
    pytest.param(1, 1024, 6, 6, False, id="t1024_h6"),
    pytest.param(1, 512, 24, 24, False, id="t512_h24"),
    pytest.param(1, 512, 2, 4, False, id="t512_h2_hv4_gva"),
    pytest.param(2, 1000, 2, 2, False, id="b2_t1000_pad_rows"),  # partial last chunk per sequence
    pytest.param(1, 0, 2, 4, [383, 402, 128, 296], id="packed_rl4_h2_hv4"),
    pytest.param(1, 0, 2, 2, [129, 1], id="packed_129_1_one_token"),
    pytest.param(1, 0, 2, 2, [1024, 1, 2047, 64, 65, 1500, 2000, 1491], id="packed_8mix_tail_pad"),
]


@pytest.mark.parametrize("batch,seq_len,heads,value_heads,packed", SHAPES)
def test_forward_matches_fla_output_and_saved_set(batch, seq_len, heads, value_heads, packed):
    _requires_blackwell()
    inp = _inputs(batch, seq_len, heads, value_heads, seed=579, packed=packed)
    ref = _fla_forward(inp)
    out = _cake_forward(inp)
    torch.testing.assert_close(out["o"].float(), ref["o"].float(), **TOL)
    for name in ("q_norm", "k_norm", "Aqk", "Akk"):
        assert out[name].dtype == ref[name].dtype == torch.bfloat16, name
        if name == "Aqk":
            # fla allocates Aqk with torch.empty and stores only the block-lower-triangular 16x16 blocks (its diagonal
            # blocks are stored whole, zero above the diagonal): the upper blocks are UNDEFINED memory on the fla side
            # (fla 0.5.2 fla/ops/kda/chunk_intra.py: chunk_kda_fwd_intra allocation and stores). Cake's kernel writes zeros
            # there (SHA-stable saved set). Compare the defined region, pin Cake's zeros above it.
            defined = _fla_defined_aqk(out[name].shape[1], inp["cu_seqlens"])  # [1, T, 1, 64], broadcast over batch / heads
            cake, fla = out[name].float(), ref[name].float()
            torch.testing.assert_close(cake.masked_fill(~defined, 0.0), fla.masked_fill(~defined, 0.0), **TOL, msg=name)
            assert float(cake.masked_fill(defined, 0.0).abs().max()) == 0.0, "Cake Aqk: non-zero entries in the upper 16x16 blocks"
            continue
        torch.testing.assert_close(out[name].float(), ref[name].float(), **TOL, msg=name)
    for name in ("q_rstd", "k_rstd", "beta"):
        assert out[name].dtype == ref[name].dtype == torch.float32, name
        torch.testing.assert_close(out[name], ref[name], atol=1e-3, rtol=1e-2, msg=name)


@pytest.mark.parametrize("batch,seq_len,heads,value_heads,packed", [SHAPES[0], SHAPES[4], SHAPES[5], SHAPES[7]])
def test_forward_is_bit_deterministic_and_honours_the_saved_set_contract(batch, seq_len, heads, value_heads, packed):
    _requires_blackwell()
    inp = _inputs(batch, seq_len, heads, value_heads, seed=580, packed=packed)
    first, second = _cake_forward(inp), _cake_forward(inp)
    for name in ("o", "beta", *SAVED_SET):
        assert torch.equal(first[name], second[name]), f"{name} differs between identical forward calls"
    b, t = inp["q"].shape[:2]
    expect = {
        "o": (torch.bfloat16, (b, t, value_heads, D)),
        "q_norm": (torch.bfloat16, (b, t, heads, D)),
        "k_norm": (torch.bfloat16, (b, t, heads, D)),
        "q_rstd": (torch.float32, (b, t, heads)),
        "k_rstd": (torch.float32, (b, t, heads)),
        "Aqk": (torch.bfloat16, (b, t, value_heads, CHUNK)),
        "Akk": (torch.bfloat16, (b, t, value_heads, CHUNK)),
        "beta": (torch.float32, (b, t, value_heads)),
    }
    for name, (dtype, shape) in expect.items():
        assert first[name].dtype == dtype and tuple(first[name].shape) == shape and first[name].is_contiguous(), name
    assert first["ht"] is None
    grads = _backward_on(inp, first)  # the backward takes the saved set as produced
    assert grads["dq"].shape == inp["q"].shape and grads["dbeta"].dtype == torch.bfloat16


@pytest.mark.parametrize("batch,seq_len,heads,value_heads,packed", SHAPES)
def test_forward_then_deterministic_backward_matches_fla_autograd(batch, seq_len, heads, value_heads, packed):
    """Cake forward saved set -> deterministic backward == fla's own forward + backward within the contract."""
    _requires_blackwell()
    inp = _inputs(batch, seq_len, heads, value_heads, seed=581, packed=packed)
    ref_out, ref = _fla_autograd(inp)
    fwd = _cake_forward(inp)
    torch.testing.assert_close(fwd["o"].float(), ref_out.float(), **TOL)
    grads = _backward_on(inp, fwd)
    for name in GRADS:
        assert grads[name].shape == ref[name].shape, name
        torch.testing.assert_close(grads[name].float(), ref[name].float(), **TOL, msg=name)
    again = _backward_on(inp, fwd)
    for name in GRADS:
        assert torch.equal(grads[name], again[name]), name


def test_state_carriers_flag_off_equals_zero_state_and_ht_has_the_carrier_layout():
    _requires_blackwell()
    inp = _inputs(1, 0, 2, 2, seed=582, packed=[200, 300])
    base = _cake_forward(inp)
    zero_h0 = torch.zeros(2, 2, D, D, dtype=torch.float32, device="cuda")
    seeded = _cake_forward(inp, h0=zero_h0, output_final_state=True)
    assert torch.equal(seeded["o"], base["o"]) and torch.equal(seeded["Aqk"], base["Aqk"])
    ht = seeded["ht"]
    assert ht is not None and ht.dtype == torch.float32 and tuple(ht.shape) == (2, 2, D, D) and ht.is_contiguous()
    assert bool(ht.abs().sum() > 0)
    with pytest.raises(ValueError):
        _cake_forward(inp, h0=torch.zeros(3, 2, D, D, dtype=torch.float32, device="cuda"))


def test_forward_rejects_operands_outside_the_contract():
    _requires_blackwell()
    inp = _inputs(1, 256, 2, 2, seed=583)
    with pytest.raises(ValueError):
        _cake_forward({**inp, "q": inp["q"].float()})
    with pytest.raises(ValueError):
        _cake_forward({**inp, "v": inp["v"][..., :64].contiguous()})
