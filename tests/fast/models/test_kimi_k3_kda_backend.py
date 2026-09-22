"""KDA backend selection for the Kimi K3 delta-rule core: ``fla`` vs ``deterministic``.

The ``deterministic`` backend keeps FLA's forward and swaps in Miles' deterministic chunked
backward (``miles_plugins.models.kda_chunk_train``). The GPU checks need a Blackwell
(SM100a/SM103a) device with flash-linear-attention and an ``nvcc`` toolchain for the
first-use build; they skip otherwise. The routing checks run anywhere.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


def load_backend_module():
    module_path = Path(__file__).resolve().parents[3] / "miles_plugins" / "models" / "kimi_k3" / "kda_backend.py"
    spec = importlib.util.spec_from_file_location("test_kimi_k3_kda_backend_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unknown_backend_raises_value_error():
    module = load_backend_module()
    with pytest.raises(ValueError, match="Unsupported KDA backend"):
        module.get_kda("nope")


@pytest.mark.parametrize(
    "capability, seq_len, cu_seqlens, cp_context, expected",
    [
        ((10, 0), 2048, None, None, True),
        ((10, 3), 256, None, None, True),
        ((9, 0), 2048, None, None, False),  # Hopper: FLA
        ((10, 0), 2000, None, None, True),  # not a multiple of 128: repacked for the backward
        ((10, 0), 512, [0, 256, 512], None, True),  # equal-length packed: direct
        ((10, 0), 384, [0, 128, 384], None, True),  # unequal packed lengths: repacked
        ((10, 0), 2048, None, object(), False),  # context parallelism
    ],
)
def test_deterministic_backward_domain(capability, seq_len, cu_seqlens, cp_context, expected):
    module = load_backend_module()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    assert (
        module.deterministic_backward_applies(
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
    module = load_backend_module()
    assert not module.deterministic_backward_applies(
        capability=(10, 0),
        head_dim=64,
        value_dim=64,
        num_heads=4,
        num_value_heads=4,
        seq_len=256,
        cu_seqlens=None,
        cp_context=None,
    )


@pytest.mark.parametrize(
    "batch, seq_len, cu_seqlens",
    [(1, 512, None), (4, 256, None), (1, 512, [0, 256, 512])],
    ids=["t512", "b4_t256", "packed_2x256"],
)
def test_plan_repack_is_direct_for_kernel_layouts(batch, seq_len, cu_seqlens):
    module = load_backend_module()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    assert module.plan_repack(cu, batch, seq_len, torch.device("cpu")) is None


@pytest.mark.parametrize(
    "batch, seq_len, cu_seqlens, slots, length",
    [
        (1, 200, None, 1, 256),
        (3, 1000, None, 3, 1024),
        (1, 1209, [0, 383, 785, 913, 1209], 4, 512),  # 383 / 402 / 128 / 296
        (1, 640, [0, 100, 640], 2, 640),  # 100 / 540: one 128-multiple slot per sequence
        (1, 300, [0, 0, 300], 2, 384),  # an empty sequence keeps its (all-pad) slot
    ],
)
def test_repack_round_trip_places_every_token_and_zero_fills_pads(batch, seq_len, cu_seqlens, slots, length):
    module = load_backend_module()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    plan = module.plan_repack(cu, batch, seq_len, torch.device("cpu"))
    assert (plan.batch, plan.length) == (slots, length)
    assert plan.index.shape == (batch * seq_len,) and plan.index.unique().numel() == batch * seq_len
    torch.manual_seed(0)
    rows = torch.randn(batch, seq_len, 2, 3) + 1.0  # no exact zeros: pads are the only zero rows
    padded = module.pad_rows(rows, plan)
    assert padded.shape == (slots, length, 2, 3)
    assert torch.equal(module.unpad_rows(padded, plan, batch, seq_len), rows)
    assert int((padded.reshape(slots * length, -1).abs().sum(-1) != 0).sum()) == batch * seq_len
    if cu is not None:
        # every sequence starts at its own 128-aligned slot and keeps its in-sequence order
        offsets = cu.tolist()
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            assert torch.equal(padded[i, : end - start], rows[0, start:end])
            assert torch.count_nonzero(padded[i, end - start :]) == 0


# ----------------------------------------------------------------------------------- GPU
_HEADS = 4
_HEAD_DIM = 128
_LOWER_BOUND = -5.0
_GRADS = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")


def _require_gpu_backends():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the deterministic KDA backward requires SM100a or SM103a")
    pytest.importorskip("fla.ops.kda")


def _inputs(seq_len: int, seed: int) -> dict[str, torch.Tensor]:
    """The layer's operand regime: scaled bf16 q/k/v/g, beta already sigmoided in fp32,
    A_log at the model init's decay rates in [1, 2)."""
    torch.manual_seed(seed)

    def activation() -> torch.Tensor:
        return torch.randn(1, seq_len, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5

    return {
        "q": activation(),
        "k": activation(),
        "v": activation(),
        "g": activation(),
        "beta": torch.rand(1, seq_len, _HEADS, device="cuda", dtype=torch.float32),
        "A_log": torch.log(torch.rand(_HEADS, device="cuda", dtype=torch.float32) + 1.0),
        "dt_bias": torch.randn(_HEADS * _HEAD_DIM, device="cuda", dtype=torch.float32),
    }


def _run(module, backend: str, inputs: dict[str, torch.Tensor], cu_seqlens: torch.Tensor | None):
    leaves = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    output = module.get_kda(backend)(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        leaves["A_log"],
        leaves["dt_bias"],
        _LOWER_BOUND,
        cu_seqlens=cu_seqlens,
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
        (1000, None),  # repacked: T not a multiple of 128
        (1209, [0, 383, 785, 913, 1209]),  # repacked: variable-length packed (RL batch shape)
        (640, [0, 100, 640]),  # repacked: two unequal sequences
    ],
    ids=["t256", "t512", "packed_2x256", "t1000_repacked", "packed_var4_repacked", "packed_2_unequal_repacked"],
)
def test_deterministic_backend_matches_fla(seq_len, cu_seqlens):
    """Same forward bits; gradients within the bf16 tolerance of FLA's own backward (direct and repacked)."""
    _require_gpu_backends()
    module = load_backend_module()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=460)
    fla_out, fla_grads = _run(module, "fla", inputs, cu)
    det_out, det_grads = _run(module, "deterministic", inputs, cu)
    torch.testing.assert_close(det_out, fla_out, rtol=0, atol=0)
    for name in _GRADS:
        assert det_grads[name].dtype == fla_grads[name].dtype, name
        torch.testing.assert_close(det_grads[name].float(), fla_grads[name].float(), atol=1e-2, rtol=1e-2, msg=name)


@pytest.mark.parametrize(
    "seq_len, cu_seqlens", [(512, None), (1209, [0, 383, 785, 913, 1209])], ids=["t512", "packed_var4_repacked"]
)
def test_deterministic_backend_backward_is_bit_deterministic(seq_len, cu_seqlens):
    _require_gpu_backends()
    module = load_backend_module()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=461)
    _, first = _run(module, "deterministic", inputs, cu)
    _, second = _run(module, "deterministic", inputs, cu)
    for name in _GRADS:
        assert torch.equal(first[name], second[name]), f"{name} differs between identical backward calls"


def test_deterministic_backend_falls_back_outside_domain(monkeypatch):
    """A call the deterministic backward does not cover (here: a device outside its list) still trains, through FLA."""
    _require_gpu_backends()
    module = load_backend_module()
    monkeypatch.setattr(module, "_BLACKWELL_CAPABILITIES", ())
    inputs = _inputs(200, seed=462)
    with pytest.warns(UserWarning, match="fell back to FLA"):
        det_out, det_grads = _run(module, "deterministic", inputs, None)
    fla_out, fla_grads = _run(module, "fla", inputs, None)
    torch.testing.assert_close(det_out, fla_out, rtol=0, atol=0)
    for name in _GRADS:
        torch.testing.assert_close(det_grads[name], fla_grads[name], rtol=0, atol=0)
