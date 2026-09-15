"""Boundary and aliasing contracts of the Kimi K3 KDA delta-rule core.

Both cases guard failure modes that are silent in training: the run keeps
learning, only worse, so nothing short of a numerical check catches them.
"""

import pytest
import torch

from miles_plugins.models.kimi_k3.ops import kda


# KDA kernel construction, validated on H200: q/k/v/g scaled bf16, beta uniform on
# [0, 1] because it is the delta rule's step size and falls outside the operator's
# domain when negative or above 1. (The "KDA goes non-finite with random weights"
# result is model-level -- random *projections* produce out-of-distribution q/k/v/g
# that compound across layers -- and does not apply to the kernel in isolation.)
_KDA_HEADS = 4
_KDA_HEAD_DIM = 128
_KDA_LOWER_BOUND = -5.0


def _kda_inputs(seq_len: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)

    def activation() -> torch.Tensor:
        return torch.randn(1, seq_len, _KDA_HEADS, _KDA_HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5

    return {
        "q": activation(),
        "k": activation(),
        "v": activation(),
        "g": activation(),
        "beta": torch.rand(1, seq_len, _KDA_HEADS, device="cuda", dtype=torch.float32),
        "A_log": torch.randn(_KDA_HEADS, device="cuda", dtype=torch.float32),
        "dt_bias": torch.randn(_KDA_HEADS * _KDA_HEAD_DIM, device="cuda", dtype=torch.float32),
    }


def _run_kda(inputs: dict[str, torch.Tensor], cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    return kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        _KDA_LOWER_BOUND,
        cu_seqlens=cu_seqlens,
    )


def _require_kda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required to run the KDA kernels")
    pytest.importorskip("fla.ops.kda")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).float().norm() / expected.float().norm()).item()


def test_kda_applies_packed_sequence_boundaries() -> None:
    """A packed batch must reproduce, per sequence, what that sequence produces alone.

    ``kda`` routes boundaries through one of two mutually exclusive channels
    (``cu_seqlens`` off CP, ``cp_context`` under CP). A caller that selects the CP
    channel and drops ``cu_seqlens`` -- the shape the CP-only helper had before the
    two kernel paths were merged -- leaves the recurrence with no boundaries at all,
    so the whole packed microbatch becomes one sequence and every sample inherits its
    predecessors' state. Nothing raises.

    Thresholds are relative because the assertion must not be satisfiable by the
    output being small: these activations sit around 1e-3, so any absolute tolerance
    worth the name swallows the entire signal. Measured on H200: the per-sequence
    equality is exact, and dropping the boundaries moves the second sequence by 0.40
    while leaving the first at exactly 0 -- the first sequence has no predecessor to
    inherit from, which is what makes this leakage rather than noise.
    """
    _require_kda()
    first_len, second_len = 96, 160
    total = first_len + second_len
    packed = _kda_inputs(total, seed=0)
    packed_output = _run_kda(packed, torch.tensor([0, first_len, total], dtype=torch.int32, device="cuda"))

    for offset, length in ((0, first_len), (first_len, second_len)):
        alone = {
            name: tensor[:, offset : offset + length] if tensor.dim() > 1 else tensor
            for name, tensor in packed.items()
        }
        alone_output = _run_kda(alone, torch.tensor([0, length], dtype=torch.int32, device="cuda"))
        torch.testing.assert_close(packed_output[:, offset : offset + length], alone_output, rtol=0, atol=0)

    unbounded_output = _run_kda(packed, None)
    assert _relative_l2(unbounded_output[:, first_len:], packed_output[:, first_len:]) > 1e-1, (
        "dropping cu_seqlens barely moved the second sequence, so boundaries are not reaching the kernel "
        "and the equality above passed vacuously"
    )
    torch.testing.assert_close(unbounded_output[:, :first_len], packed_output[:, :first_len], rtol=0, atol=0)


def test_kda_does_not_mutate_its_inputs() -> None:
    """The forward must leave q/k/v/g/beta untouched.

    SGLang's vendored ``chunk_kda`` overwrote its ``v`` buffer with a WY-representation
    intermediate. Any caller that reads an input back afterwards -- a manual
    ``save_for_backward`` re-derivation being the case that bit us -- then
    differentiates at a point the forward never evaluated, which put the q/k/g/beta
    gradients ~110x low and near-orthogonal. Exact comparison: aliasing is a
    mutation, not a rounding.
    """
    _require_kda()
    seq_len = 256
    inputs = _kda_inputs(seq_len, seed=1)
    before = {name: tensor.clone() for name, tensor in inputs.items()}

    _run_kda(inputs, torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"))

    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0, msg=f"KDA forward mutated input {name}")
