"""slice_lora_to_rank trims max-rank-padded LoRA tensors to the adapter's real rank;
used by weight-sync and HF PEFT export (PEFT rejects tensors padded past the declared rank)."""

import pytest
import torch

from miles.backends.megatron_utils.multi_lora_utils import slice_lora_to_rank


def _padded(shape, live_rows=None, live_cols=None):
    t = torch.zeros(shape)
    if live_rows is not None:
        t[:live_rows] = 1.0
    if live_cols is not None:
        t[:, :live_cols] = 1.0
    return t


def test_lora_a_is_sliced_on_the_rank_dim():
    tensor = _padded((32, 8), live_rows=16)
    out = slice_lora_to_rank("base_model.q_proj.lora_A.weight", tensor, 16)
    assert out.shape == (16, 8)
    assert torch.equal(out, tensor[:16])


def test_lora_b_is_sliced_on_the_rank_dim():
    tensor = _padded((8, 32), live_cols=16)
    out = slice_lora_to_rank("base_model.q_proj.lora_B.weight", tensor, 16)
    assert out.shape == (8, 16)
    assert torch.equal(out, tensor[:, :16])


def test_nonzero_padding_is_rejected():
    # Live values beyond the adapter's rank mean the pad rows were trained —
    # slicing would silently drop signal, so it must hard-fail instead.
    tensor = torch.ones(32, 8)
    with pytest.raises(AssertionError, match="padded dims are non-zero"):
        slice_lora_to_rank("x.lora_A.weight", tensor, 16)


def test_full_rank_tensor_passes_through():
    tensor = torch.ones(16, 8)
    assert slice_lora_to_rank("x.lora_A.weight", tensor, 16) is tensor


def test_non_lora_names_pass_through():
    tensor = torch.ones(32, 8)
    assert slice_lora_to_rank("x.some_other.weight", tensor, 16) is tensor
