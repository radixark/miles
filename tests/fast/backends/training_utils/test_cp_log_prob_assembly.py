"""`assemble_log_prob_from_cp` must invert the CP split exactly."""

import pytest
import torch

from miles.backends.training_utils.cp_utils import assemble_log_prob_from_cp, get_logits_and_tokens_offset_with_cp


def _split(
    values: torch.Tensor,
    total_length: int,
    response_length: int,
    cp_size: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> dict[int, torch.Tensor]:
    """What each cp rank keeps, mirroring `slice_log_prob_with_cp` without a process group."""
    prompt_length = total_length - response_length
    chunks = {}
    for cp_rank in range(cp_size):
        _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
            total_length, response_length, qkv_format, max_seq_len, cp_rank=cp_rank, cp_size=cp_size
        )
        parts = [values[lo - (prompt_length - 1) : hi - (prompt_length - 1)] for lo, hi in logits_offset]
        chunks[cp_rank] = torch.cat(parts, dim=0)
    return chunks


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("total_length,response_length", [(34017, 33789), (2630, 2402), (1024, 512), (97, 96)])
def test_split_then_assemble_is_identity_thd(cp_size, total_length, response_length):
    values = torch.arange(response_length, dtype=torch.float32)
    chunks = _split(values, total_length, response_length, cp_size)
    assert sum(len(c) for c in chunks.values()) == response_length, "the slices must tile the response exactly"

    restored = assemble_log_prob_from_cp(chunks, total_length, response_length, cp_size)
    torch.testing.assert_close(restored, values)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("total_length,response_length,max_seq_len", [(1024, 512, 2048), (700, 640, 1024)])
def test_split_then_assemble_is_identity_bshd(cp_size, total_length, response_length, max_seq_len):
    """bshd partitions the padded maximum, not the sequence, so the chunk size
    comes from `max_seq_len` and the two sides must agree on which one."""
    values = torch.arange(response_length, dtype=torch.float32)
    chunks = _split(values, total_length, response_length, cp_size, qkv_format="bshd", max_seq_len=max_seq_len)
    restored = assemble_log_prob_from_cp(
        chunks, total_length, response_length, cp_size, qkv_format="bshd", max_seq_len=max_seq_len
    )
    torch.testing.assert_close(restored, values)


def test_partial_group_is_rejected():
    """A missing rank leaves holes that are unknown, not zero."""
    values = torch.arange(512, dtype=torch.float32)
    chunks = _split(values, 640, 512, 4)
    del chunks[2]
    with pytest.raises(AssertionError, match="missing"):
        assemble_log_prob_from_cp(chunks, 640, 512, 4)
