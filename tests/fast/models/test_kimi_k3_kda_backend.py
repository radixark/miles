"""KDA backend selection on the head-sharded linear-attention layer: ``fla`` vs the deterministic family.

``--kda-backend deterministic_fla_fwd`` keeps fla's forward and swaps in Miles' deterministic chunked
backward; ``deterministic_cake_fwd`` runs the Cake forward with it; ``deterministic`` is the alias of the
family's current default (pinned here: the Cake forward). All three are the
drop-in ``miles_plugins.models.kda_chunk_train.kda_backend.chunk_kda`` the shared ``kda_recurrence`` runs.
The GPU checks need a Blackwell (SM100a/SM103a) device with flash-linear-attention, Megatron and an
``nvcc`` toolchain for the first-use build; they skip otherwise. The routing checks run anywhere with torch.
The context-parallel checks run the CP host protocol on a real fla CP context over a
one-rank NCCL group (one GPU); the exchange between ranks is covered by the 2- and 4-GPU torchrun arms of
``tests/fast-gpu/test_linear_attn_layer.py``.
"""

import pytest
import torch

from miles_plugins.models.kda_chunk_train import kda_backend as drop_in


@pytest.mark.parametrize(
    "capability, seq_len, cu_seqlens, cp_context, expected",
    [
        ((10, 0), 2048, None, None, True),
        ((10, 3), 256, None, None, True),
        ((9, 0), 2048, None, None, False),  # Hopper: FLA
        ((10, 0), 2000, None, None, True),  # any fixed length
        ((10, 0), 512, [0, 256, 512], None, True),  # equal-length packed
        ((10, 0), 384, [0, 128, 384], None, True),  # unequal packed lengths (native cu_seqlens)
        ((10, 0), 2048, None, object(), True),  # context parallelism: the CP host protocol
    ],
)
def test_deterministic_backward_domain(capability, seq_len, cu_seqlens, cp_context, expected):
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    assert (
        drop_in.deterministic_backward_applies(
            capability=capability,
            head_dim=128,
            value_dim=128,
            num_heads=4,
            num_value_heads=4,
            seq_len=seq_len,
            cu_seqlens=cu,
            cp_context=cp_context,
        )
        is expected
    )


def test_deterministic_backward_domain_requires_head_dim_128():
    assert not drop_in.deterministic_backward_applies(
        capability=(10, 0),
        head_dim=64,
        value_dim=64,
        num_heads=4,
        num_value_heads=4,
        seq_len=256,
        cu_seqlens=None,
        cp_context=None,
    )


def test_deterministic_modes_and_the_default_alias():
    """Three ``--kda-backend`` values of the family; ``deterministic`` means the Cake forward (the alias is the one
    constant ``DETERMINISTIC_DEFAULT`` plus this pin; ``deterministic_fla_fwd`` stays the fla-forward comparator)."""
    from miles.utils.arguments import KDA_BACKENDS

    assert KDA_BACKENDS == ("fla", "deterministic", "deterministic_fla_fwd", "deterministic_cake_fwd")
    assert drop_in.DETERMINISTIC_BACKENDS == ("deterministic", "deterministic_fla_fwd", "deterministic_cake_fwd")
    assert drop_in.DETERMINISTIC_DEFAULT == "deterministic_cake_fwd"
    assert drop_in.forward_source("deterministic") == "cake"
    assert drop_in.forward_source("deterministic_fla_fwd") == "fla"
    assert drop_in.forward_source("deterministic_cake_fwd") == "cake"
    with pytest.raises(ValueError, match="not a deterministic KDA backend"):
        drop_in.forward_source("fla")
    cake = drop_in.chunk_kda_for("deterministic_cake_fwd")
    fla_fwd = drop_in.chunk_kda_for("deterministic_fla_fwd")
    assert cake.takes_beta_logits is True and fla_fwd.takes_beta_logits is False
    assert drop_in.chunk_kda_for("deterministic") is drop_in.chunk_kda_for("deterministic")
    assert drop_in.chunk_kda_for("deterministic").forward_source == "cake"
    assert drop_in.chunk_kda_for("deterministic").takes_beta_logits is True


def test_outside_contract_beta_convention_follows_the_forward_source():
    options = dict(
        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, safe_gate=True, lower_bound=-5.0,
        return_intermediate_states=False, A_log=torch.zeros(4), dt_bias=torch.zeros(512),
    )
    assert drop_in.outside_contract(**options, forward_source="fla") is None
    assert "use_beta_sigmoid_in_kernel" in drop_in.outside_contract(**options, forward_source="cake")
    logits = {**options, "use_beta_sigmoid_in_kernel": True}
    assert drop_in.outside_contract(**logits, forward_source="cake") is None
    assert "use_beta_sigmoid_in_kernel" in drop_in.outside_contract(**logits, forward_source="fla")


def test_options_outside_the_contract_are_named():
    """Forward options the deterministic backward's saved set is not defined for route to fla, with the reason."""
    options = dict(
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
        return_intermediate_states=False,
        A_log=torch.zeros(4),
        dt_bias=torch.zeros(512),
    )
    assert drop_in.outside_contract(**options) is None
    assert "use_qk_l2norm_in_kernel" in drop_in.outside_contract(**{**options, "use_qk_l2norm_in_kernel": False})
    assert "output_final_state" in drop_in.outside_contract(**{**options, "output_final_state": True})
    assert "lower_bound" in drop_in.outside_contract(**{**options, "lower_bound": None})


# ----------------------------------------------------------------------------------- GPU
_HEADS = 4
_HEAD_DIM = 128
_LOWER_BOUND = -5.0
_GRADS = ("q", "k", "v", "decay", "beta_logits", "A_log", "dt_bias")


def _kda_recurrence():
    """The shared layer's recurrence; its module imports Megatron, so skip where that is absent."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the deterministic KDA backward requires SM100a or SM103a")
    pytest.importorskip("fla.ops.kda")
    pytest.importorskip("megatron.core")
    from miles_plugins.models.linear_attn import kda_recurrence

    return kda_recurrence


def test_unknown_backend_raises_value_error():
    pytest.importorskip("megatron.core")
    from miles_plugins.models.linear_attn import kda_kernel

    with pytest.raises(ValueError, match="Unsupported KDA backend"):
        kda_kernel("nope")


def _inputs(seq_len: int, seed: int, heads: int = _HEADS) -> dict[str, torch.Tensor]:
    """The layer's operand regime: scaled bf16 q/k/v and low-rank decay, beta as its logit (the
    recurrence squashes it, or hands it to the Cake forward), A_log at the model init's decay rates in [1, 2)."""
    torch.manual_seed(seed)

    def activation() -> torch.Tensor:
        return torch.randn(1, seq_len, heads, _HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5

    return {
        "q": activation(),
        "k": activation(),
        "v": activation(),
        "decay": activation().reshape(1, seq_len, heads * _HEAD_DIM),
        "beta_logits": torch.randn(1, seq_len, heads, device="cuda", dtype=torch.bfloat16),
        "A_log": torch.log(torch.rand(heads, device="cuda", dtype=torch.float32) + 1.0),
        "dt_bias": torch.randn(heads * _HEAD_DIM, device="cuda", dtype=torch.float32),
    }


def _run(kda_recurrence, backend: str, inputs: dict[str, torch.Tensor], cu_seqlens: torch.Tensor | None):
    """One forward + backward through the layer's recurrence. Packed input travels as the layer sends it: the
    device int32 ``cu_seqlens`` plus its CPU int64 host copy ``cu_seqlens_cpu``."""
    leaves = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    cu_seqlens_cpu = None if cu_seqlens is None else torch.tensor(cu_seqlens.tolist(), dtype=torch.int64)
    output = kda_recurrence(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["beta_logits"],
        leaves["decay"],
        leaves["A_log"],
        leaves["dt_bias"],
        gate_lower_bound=_LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        cp_context=None,
        backend=backend,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    torch.manual_seed(0)
    upstream = torch.randn_like(output)
    grads = torch.autograd.grad(output, [leaves[n] for n in _GRADS], grad_outputs=upstream)
    return output.detach(), dict(zip(_GRADS, [g.detach() for g in grads], strict=True))


@pytest.mark.parametrize(
    "seq_len, cu_seqlens",
    [
        (256, None),
        (512, None),
        (512, [0, 256, 512]),
        (1000, None),  # T not a multiple of the chunk size
        (1209, [0, 383, 785, 913, 1209]),  # variable-length packed (RL batch shape)
        (640, [0, 100, 640]),  # two unequal sequences
    ],
    ids=["t256", "t512", "packed_2x256", "t1000", "packed_var4", "packed_2_unequal"],
)
def test_deterministic_backend_matches_fla(seq_len, cu_seqlens):
    """fla-forward mode: same forward bits; gradients within the bf16 tolerance of FLA's own backward (fixed and
    packed layouts). The default alias runs the Cake forward, covered by the Cake-forward checks below."""
    kda_recurrence = _kda_recurrence()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=460)
    fla_out, fla_grads = _run(kda_recurrence, "fla", inputs, cu)
    det_out, det_grads = _run(kda_recurrence, "deterministic_fla_fwd", inputs, cu)
    torch.testing.assert_close(det_out, fla_out, rtol=0, atol=0)
    for name in _GRADS:
        assert det_grads[name].dtype == fla_grads[name].dtype, name
        torch.testing.assert_close(det_grads[name].float(), fla_grads[name].float(), atol=1e-2, rtol=1e-2, msg=name)


@pytest.mark.parametrize(
    "seq_len, cu_seqlens", [(512, None), (1209, [0, 383, 785, 913, 1209])], ids=["t512", "packed_var4"]
)
def test_deterministic_backend_backward_is_bit_deterministic(seq_len, cu_seqlens):
    kda_recurrence = _kda_recurrence()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=461)
    _, first = _run(kda_recurrence, "deterministic", inputs, cu)
    _, second = _run(kda_recurrence, "deterministic", inputs, cu)
    for name in _GRADS:
        assert torch.equal(first[name], second[name]), f"{name} differs between identical backward calls"


def test_deterministic_backend_falls_back_outside_domain(monkeypatch):
    """A call the deterministic backward does not cover (here: a device outside its list) still trains, through fla
    (fla-forward mode: fla's bits; the Cake-forward mode's fallback is checked below within the contract)."""
    kda_recurrence = _kda_recurrence()
    monkeypatch.setattr(drop_in, "_BLACKWELL_CAPABILITIES", ())
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    inputs = _inputs(200, seed=462)
    with pytest.warns(UserWarning, match="fell back to fla"):
        det_out, det_grads = _run(kda_recurrence, "deterministic_fla_fwd", inputs, None)
    fla_out, fla_grads = _run(kda_recurrence, "fla", inputs, None)
    torch.testing.assert_close(det_out, fla_out, rtol=0, atol=0)
    for name in _GRADS:
        torch.testing.assert_close(det_grads[name], fla_grads[name], rtol=0, atol=0)


# ----------------------------------------------------------------------------- deterministic family (X1)
_CAKE = "deterministic_cake_fwd"


def _no_fallback():
    """Context: fail the test if the drop-in warns that it fell back to fla."""
    import contextlib
    import warnings

    @contextlib.contextmanager
    def guard():
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            yield
        hits = [w for w in seen if "fell back to fla" in str(w.message)]
        assert not hits, [str(w.message) for w in hits]

    return guard()


def test_deterministic_alias_is_bitwise_the_cake_forward_mode():
    """``deterministic`` and ``deterministic_cake_fwd`` are one code path (pins the alias: Cake forward + deterministic
    backward); the fla-forward comparator is a different forward, so its output is not the alias's bits."""
    kda_recurrence = _kda_recurrence()
    inputs = _inputs(640, seed=470)
    cu = torch.tensor([0, 100, 640], dtype=torch.int32, device="cuda")
    with _no_fallback():
        a_out, a_grads = _run(kda_recurrence, "deterministic", inputs, cu)
        b_out, b_grads = _run(kda_recurrence, _CAKE, inputs, cu)
        c_out, _ = _run(kda_recurrence, "deterministic_fla_fwd", inputs, cu)
    assert not torch.equal(a_out, c_out), "the alias still runs fla's forward"
    assert torch.equal(a_out, b_out)
    for name in _GRADS:
        assert torch.equal(a_grads[name], b_grads[name]), name


@pytest.mark.parametrize(
    "seq_len, cu_seqlens, heads",
    [
        (2048, None, 12),  # the trainer's heads per TP rank
        (1024, None, 6),
        (512, None, 24),
        (1000, None, 4),  # T not a multiple of the chunk size (pad rows)
        (1209, [0, 383, 785, 913, 1209], 4),  # variable-length packed (RL batch shape)
        (130, [0, 129, 130], 4),  # one-token sequence
        (8192, [0, 1024, 1025, 3072, 3136, 3201, 4701, 6701, 8192], 12),  # 8-mix with a tail pad pseudo-sequence
    ],
    ids=["t2048_h12", "t1024_h6", "t512_h24", "t1000_pad", "packed_var4", "packed_one_token", "packed_8mix_h12"],
)
def test_cake_forward_backend_matches_fla_within_the_contract(seq_len, cu_seqlens, heads):
    """Cake forward + deterministic backward vs fla through the layer's recurrence: output and gradients within
    the contract tolerance (the Cake forward is not fla's bits: single-rounded kb, fused sigmoid, own gate cumsum);
    no fallback warning; gradient dtypes equal fla's."""
    kda_recurrence = _kda_recurrence()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=471, heads=heads)
    fla_out, fla_grads = _run(kda_recurrence, "fla", inputs, cu)
    with _no_fallback():
        det_out, det_grads = _run(kda_recurrence, _CAKE, inputs, cu)
    assert det_out.dtype == fla_out.dtype and det_out.shape == fla_out.shape
    torch.testing.assert_close(det_out.float(), fla_out.float(), atol=1e-2, rtol=1e-2)
    for name in _GRADS:
        assert det_grads[name].dtype == fla_grads[name].dtype, name
        torch.testing.assert_close(det_grads[name].float(), fla_grads[name].float(), atol=1e-2, rtol=1e-2, msg=name)


@pytest.mark.parametrize(
    "seq_len, cu_seqlens", [(2048, None), (1209, [0, 383, 785, 913, 1209])], ids=["t2048", "packed_var4"]
)
def test_cake_forward_backend_is_bit_deterministic_forward_and_backward(seq_len, cu_seqlens):
    kda_recurrence = _kda_recurrence()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=472, heads=12)
    with _no_fallback():
        first_out, first = _run(kda_recurrence, _CAKE, inputs, cu)
        second_out, second = _run(kda_recurrence, _CAKE, inputs, cu)
    assert torch.equal(first_out, second_out), "the Cake forward output differs between identical calls"
    for name in _GRADS:
        assert torch.equal(first[name], second[name]), f"{name} differs between identical backward calls"


def test_cake_forward_backend_falls_back_outside_domain(monkeypatch):
    """Outside the kernels' domain the Cake-forward mode still trains through fla (beta logits with fla's fused
    sigmoid), with exactly the one warning, and lands within the contract of the plain fla backend."""
    kda_recurrence = _kda_recurrence()
    monkeypatch.setattr(drop_in, "_BLACKWELL_CAPABILITIES", ())
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    inputs = _inputs(200, seed=473)
    with pytest.warns(UserWarning, match="fell back to fla") as seen:
        det_out, det_grads = _run(kda_recurrence, _CAKE, inputs, None)
    assert sum("fell back to fla" in str(w.message) for w in seen) == 1
    fla_out, fla_grads = _run(kda_recurrence, "fla", inputs, None)
    torch.testing.assert_close(det_out.float(), fla_out.float(), atol=1e-2, rtol=1e-2)
    for name in _GRADS:
        torch.testing.assert_close(det_grads[name].float(), fla_grads[name].float(), atol=1e-2, rtol=1e-2, msg=name)


# ----------------------------------------------------------------------------- context parallelism
_CP_PACK = [0, 383, 785, 913, 1209]


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _cp_world1_context(cu_seqlens: torch.Tensor):
    """A real fla CP context on a one-rank NCCL group: every call of the CP host protocol runs on one GPU (the
    pre-process wrappers' ``all_gather`` over one rank, the first/last-rank roles, the compressed-state plumbing, the
    hit-path caches); the state exchange between ranks is the torchrun arms' job."""
    import atexit

    import torch.distributed as dist

    pytest.importorskip("fla.ops.cp")
    from miles_plugins.models.cp_utils import build_fla_cp_context

    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{_free_port()}", world_size=1, rank=0)
        atexit.register(dist.destroy_process_group)
    host = torch.tensor(cu_seqlens.tolist(), dtype=torch.int64)
    ctx = build_fla_cp_context(cu_seqlens, dist.group.WORLD, 4, torch.device("cuda"), cu_seqlens_cpu=host)
    return ctx, ctx.cu_seqlens_cpu.to(torch.int64)


def _run_cp(kda_recurrence, backend: str, inputs: dict[str, torch.Tensor], ctx, cu_seqlens_cpu: torch.Tensor):
    """One forward + backward through the layer's recurrence under CP: the context carries the boundaries (the layer
    passes no device ``cu_seqlens``) and ``cu_seqlens_cpu`` is the int64 host copy of the rank-local ones."""
    leaves = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    output = kda_recurrence(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["beta_logits"],
        leaves["decay"],
        leaves["A_log"],
        leaves["dt_bias"],
        gate_lower_bound=_LOWER_BOUND,
        cu_seqlens=None,
        cp_context=ctx,
        backend=backend,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    torch.manual_seed(0)
    upstream = torch.randn_like(output)
    grads = torch.autograd.grad(output, [leaves[n] for n in _GRADS], grad_outputs=upstream)
    return output.detach(), dict(zip(_GRADS, [g.detach() for g in grads], strict=True))


@pytest.mark.parametrize("backend", ["deterministic_fla_fwd", _CAKE])
def test_cp_context_runs_the_host_protocol_with_no_fallback_and_matches_fla(backend):
    """CP is in the domain: with a real fla CP context both deterministic modes run the
    protocol -- no fallback warning, the backward pre-process hook ran (``LAST_CP_STATES['dht']``), the fla-forward
    mode's output is fla-CP's bits, the Cake-forward mode's within the contract -- and the gradients land within
    1e-2 of fla's own CP autograd on the same context."""
    kda_recurrence = _kda_recurrence()
    from miles_plugins.models.kda_chunk_train import backward as det_backward

    cu = torch.tensor(_CP_PACK, dtype=torch.int32, device="cuda")
    ctx, host = _cp_world1_context(cu)
    inputs = _inputs(_CP_PACK[-1], seed=480)
    fla_out, fla_grads = _run_cp(kda_recurrence, "fla", inputs, ctx, host)
    with _no_fallback():
        det_out, det_grads = _run_cp(kda_recurrence, backend, inputs, ctx, host)
    states = det_backward.LAST_CP_STATES
    assert states["roles"]["is_first_rank"] and states["roles"]["is_last_rank"] and states["dht"] is not None
    assert states["local_lengths"] == [b - a for a, b in zip(_CP_PACK[:-1], _CP_PACK[1:], strict=True)]
    assert det_out.dtype == fla_out.dtype and det_out.shape == fla_out.shape
    if backend == "deterministic_fla_fwd":
        torch.testing.assert_close(det_out, fla_out, rtol=0, atol=0)
    else:
        torch.testing.assert_close(det_out.float(), fla_out.float(), atol=1e-2, rtol=1e-2)
    for name in _GRADS:
        assert det_grads[name].dtype == fla_grads[name].dtype, name
        torch.testing.assert_close(det_grads[name].float(), fla_grads[name].float(), atol=1e-2, rtol=1e-2, msg=name)


@pytest.mark.parametrize("backend", ["deterministic_fla_fwd", _CAKE])
def test_cp_hit_path_is_bit_deterministic_and_sync_free(backend):
    """After one warm-up on a packing, a CP forward + backward issues no synchronising CUDA call
    (``torch.cuda.set_sync_debug_mode('error')``: no ``.item()`` / ``.tolist()`` / blocking copy on the hit path --
    the pair ``cu_seqlens``, the layout tables and the state dummies are cached) and repeats the warm-up's bits."""
    kda_recurrence = _kda_recurrence()
    cu = torch.tensor(_CP_PACK, dtype=torch.int32, device="cuda")
    ctx, host = _cp_world1_context(cu)
    inputs = _inputs(_CP_PACK[-1], seed=481)
    with _no_fallback():
        first_out, first = _run_cp(kda_recurrence, backend, inputs, ctx, host)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            second_out, second = _run_cp(kda_recurrence, backend, inputs, ctx, host)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(first_out, second_out), "the CP forward output differs between identical calls"
    for name in _GRADS:
        assert torch.equal(first[name], second[name]), f"{name} differs between identical CP backward calls"
