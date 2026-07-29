import pytest

pytest.importorskip("megatron.core.distributed")

import torch

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes


def test_hash_tensor_bytes_contiguous_float32_returns_raw_buffer_bytes() -> None:
    """A contiguous 2D float32 tensor hashes to its exact raw little-endian buffer bytes."""
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = _hash_tensor_bytes(tensor)
    assert isinstance(result, bytes)
    assert len(result) == 6 * 4
    assert result == tensor.reshape(-1).contiguous().numpy().tobytes()


def test_hash_tensor_bytes_int64_byte_length_is_eight_per_element() -> None:
    """An int64 tensor yields 8 bytes per element with bit-exact buffer contents."""
    tensor = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    result = _hash_tensor_bytes(tensor)
    assert len(result) == 6 * 8
    assert result == tensor.reshape(-1).contiguous().numpy().tobytes()


def test_hash_tensor_bytes_noncontiguous_transpose_does_not_raise() -> None:
    """A non-contiguous transposed view is hashed without raising (regression guard)."""
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    transposed = base.t()
    assert not transposed.is_contiguous()
    result = _hash_tensor_bytes(transposed)
    assert isinstance(result, bytes)
    assert len(result) == 6 * 4


def test_hash_tensor_bytes_noncontiguous_matches_contiguous_row_major_bytes() -> None:
    """The non-contiguous path returns the contiguous row-major bytes of the transposed values."""
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    transposed = base.t()
    result = _hash_tensor_bytes(transposed)
    assert result == transposed.contiguous().reshape(-1).numpy().tobytes()


def test_hash_tensor_bytes_distinct_values_produce_distinct_bytes() -> None:
    """Different tensor contents of the same shape and dtype produce different bytes."""
    a = torch.zeros(2, 3, dtype=torch.float32)
    b = torch.ones(2, 3, dtype=torch.float32)
    assert _hash_tensor_bytes(a) != _hash_tensor_bytes(b)
    assert len(_hash_tensor_bytes(a)) == len(_hash_tensor_bytes(b)) == 6 * 4
