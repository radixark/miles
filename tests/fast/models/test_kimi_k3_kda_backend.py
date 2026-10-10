"""KDA backend selection on the head-sharded linear-attention layer: ``fla`` vs ``deterministic``.

``--kda-backend deterministic`` keeps fla's forward and swaps in Miles' deterministic chunked
backward (``miles_plugins.models.kda_chunk_train.kda_backend.chunk_kda``, a drop-in the shared
``kda_recurrence`` runs). The GPU checks need a Blackwell (SM100a/SM103a) device with
flash-linear-attention, Megatron and an ``nvcc`` toolchain for the first-use build; they skip
otherwise. The routing checks run anywhere with torch.
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
        ((10, 0), 2048, None, object(), False),  # context parallelism
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


def _inputs(seq_len: int, seed: int) -> dict[str, torch.Tensor]:
    """The layer's operand regime: scaled bf16 q/k/v and low-rank decay, beta as its logit (the
    recurrence squashes it), A_log at the model init's decay rates in [1, 2)."""
    torch.manual_seed(seed)

    def activation() -> torch.Tensor:
        return torch.randn(1, seq_len, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5

    return {
        "q": activation(),
        "k": activation(),
        "v": activation(),
        "decay": activation().reshape(1, seq_len, _HEADS * _HEAD_DIM),
        "beta_logits": torch.randn(1, seq_len, _HEADS, device="cuda", dtype=torch.bfloat16),
        "A_log": torch.log(torch.rand(_HEADS, device="cuda", dtype=torch.float32) + 1.0),
        "dt_bias": torch.randn(_HEADS * _HEAD_DIM, device="cuda", dtype=torch.float32),
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
    """Same forward bits; gradients within the bf16 tolerance of FLA's own backward (fixed and packed layouts)."""
    kda_recurrence = _kda_recurrence()
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    inputs = _inputs(seq_len, seed=460)
    fla_out, fla_grads = _run(kda_recurrence, "fla", inputs, cu)
    det_out, det_grads = _run(kda_recurrence, "deterministic", inputs, cu)
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
    """A call the deterministic backward does not cover (here: a device outside its list) still trains, through fla."""
    kda_recurrence = _kda_recurrence()
    monkeypatch.setattr(drop_in, "_BLACKWELL_CAPABILITIES", ())
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    inputs = _inputs(200, seed=462)
    with pytest.warns(UserWarning, match="fell back to fla"):
        det_out, det_grads = _run(kda_recurrence, "deterministic", inputs, None)
    fla_out, fla_grads = _run(kda_recurrence, "fla", inputs, None)
    torch.testing.assert_close(det_out, fla_out, rtol=0, atol=0)
    for name in _GRADS:
        torch.testing.assert_close(det_grads[name], fla_grads[name], rtol=0, atol=0)
