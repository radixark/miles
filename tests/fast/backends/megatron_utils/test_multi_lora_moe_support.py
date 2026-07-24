"""MoE multi-LoRA support guards (CI-registered):

* ``slice_lora_to_rank`` trims the rank axis correctly for the 3-D grouped MoE-expert
  layout (the previous hard-coded axis sliced the expert axis and corrupted the tensor).
  The 2-D dense layout is covered canonically in ``test_slice_lora_to_rank.py``.
* ``validate_multi_lora_args`` loudly rejects expert target modules until MoE expert
  multi-LoRA is wired end-to-end, instead of silently dropping the expert adapters.
"""

from argparse import Namespace

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

from miles.backends.megatron_utils.multi_lora_utils import slice_lora_to_rank
from miles.utils.multi_lora import validate_multi_lora_args


MAXR, R, IN, OUT, E = 8, 5, 6, 7, 3


def test_slice_3d_expert_lora_a_slices_rank_not_expert_axis():
    # [num_experts, rank, in]: rank axis is dim 1; the expert axis must be preserved.
    t = torch.randn(E, MAXR, IN)
    t[:, R:, :] = 0
    out = slice_lora_to_rank("x.experts.gate_proj.lora_A.weight", t, R)
    assert out.shape == (E, R, IN)
    assert torch.equal(out, t[:, :R, :])


def test_slice_3d_expert_lora_b_slices_rank_not_expert_axis():
    # [num_experts, out, rank]: rank axis is dim 2.
    t = torch.randn(E, OUT, MAXR)
    t[:, :, R:] = 0
    out = slice_lora_to_rank("x.experts.down_proj.lora_B.weight", t, R)
    assert out.shape == (E, OUT, R)
    assert torch.equal(out, t[:, :, :R])


def test_slice_3d_nonzero_padding_raises():
    t = torch.randn(E, MAXR, IN)  # padded rank slots NOT zeroed -> corruption guard fires
    with pytest.raises(AssertionError, match="padded rank slots are non-zero"):
        slice_lora_to_rank("x.experts.gate_proj.lora_A.weight", t, R)


def _guard_ns(target_modules):
    # Only the attributes touched up to the MoE guard are required.
    return Namespace(
        multi_lora_n_adapters=2,
        rollout_function_path="custom.fn",
        data_source_path="custom.ds",
        lora_rank=8,
        target_modules=target_modules,
    )


def test_validate_rejects_expert_target_modules():
    with pytest.raises(AssertionError, match="MoE expert target modules"):
        validate_multi_lora_args(_guard_ns(["*.mlp.experts.linear_fc1", "linear_qkv"]))


def test_validate_allows_non_expert_target_modules_past_guard():
    # A dense/attention target must pass the MoE guard (it may still trip a later,
    # unrelated check on this minimal Namespace — we only assert the guard did not
    # false-positive on non-expert names).
    try:
        validate_multi_lora_args(_guard_ns(["linear_qkv", "linear_proj"]))
    except (AssertionError, AttributeError) as exc:
        assert "MoE expert target modules" not in str(exc)
