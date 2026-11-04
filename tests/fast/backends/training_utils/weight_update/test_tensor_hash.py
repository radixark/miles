import hashlib

import pytest
import torch

from miles.backends.training_utils.weight_update.checksum_utils import hash_tensor_sha256


class TestHashTensorSha256:
    def test_uint8_values_match_the_standard_sha256_digest(self) -> None:
        """The shared checksum hashes tensor bytes using the interoperable SHA256 algorithm."""
        tensor = torch.tensor([0, 1, 127, 128, 255], dtype=torch.uint8)

        assert hash_tensor_sha256(tensor) == hashlib.sha256(bytes([0, 1, 127, 128, 255])).hexdigest()

    def test_noncontiguous_views_hash_logical_values_without_storage_gaps(self) -> None:
        """A strided view must not hash padding or unrelated elements from its backing storage."""
        storage = torch.tensor([1, 99, 2, 98, 3, 97], dtype=torch.uint8)

        assert hash_tensor_sha256(storage[::2]) == hashlib.sha256(bytes([1, 2, 3])).hexdigest()

    @pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
    def test_scalars_and_reshapes_with_the_same_bytes_have_the_same_hash(self, shape: tuple[int, ...]) -> None:
        """Tensor shape does not change the wire-byte checksum."""
        tensor = torch.tensor([42], dtype=torch.uint8).reshape(shape)

        assert hash_tensor_sha256(tensor) == hashlib.sha256(bytes([42])).hexdigest()

    def test_empty_tensors_hash_the_empty_byte_sequence(self) -> None:
        """An empty shard has the standard empty checksum rather than raising or hashing metadata."""
        assert hash_tensor_sha256(torch.empty((0, 3))) == hashlib.sha256(b"").hexdigest()

    def test_grad_tracking_is_preserved_without_becoming_part_of_the_hash(self) -> None:
        """Hashing must detach its read without mutating the trainable tensor."""
        tensor = torch.tensor([1.0, -2.0], requires_grad=True)
        before = tensor.detach().clone()

        assert hash_tensor_sha256(tensor) == hash_tensor_sha256(before)
        assert tensor.requires_grad
        assert tensor.grad is None
        assert torch.equal(tensor, before)
