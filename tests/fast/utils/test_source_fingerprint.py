import numpy
import pytest
import torch
from PIL import Image

from miles.utils.source_fingerprint import canonical_source_digest


def test_canonical_source_digest_ignores_mapping_order() -> None:
    first = {"prompt": "inspect", "metadata": {"left": 1, "right": 2}}
    second = {"metadata": {"right": 2, "left": 1}, "prompt": "inspect"}

    assert canonical_source_digest(first) == canonical_source_digest(second)


def test_canonical_source_digest_distinguishes_value_types() -> None:
    assert canonical_source_digest(True) != canonical_source_digest(1)
    assert canonical_source_digest(b"1") != canonical_source_digest("1")
    assert canonical_source_digest([1]) != canonical_source_digest((1,))


def test_canonical_source_digest_normalizes_tensor_views() -> None:
    tensor = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    noncontiguous = tensor.t().contiguous().t()

    assert not noncontiguous.is_contiguous()
    assert canonical_source_digest(tensor) == canonical_source_digest(noncontiguous)


def test_canonical_source_digest_includes_image_payload() -> None:
    first = Image.fromarray(numpy.array([[0, 1], [2, 3]], dtype=numpy.uint8))
    second = Image.fromarray(numpy.array([[0, 1], [2, 4]], dtype=numpy.uint8))

    assert canonical_source_digest(first) != canonical_source_digest(second)


def test_canonical_source_digest_includes_image_metadata() -> None:
    first = Image.new("RGB", (2, 1), color="red")
    second = first.copy()
    first.getexif()[274] = 1
    second.getexif()[274] = 6
    assert canonical_source_digest(first) != canonical_source_digest(second)

    second.getexif()[274] = 1
    first.info["icc_profile"] = b"first"
    second.info["icc_profile"] = b"second"
    assert canonical_source_digest(first) != canonical_source_digest(second)


def test_canonical_source_digest_normalizes_object_arrays() -> None:
    first = numpy.array([{"left": 1, "right": 2}, ["value"]], dtype=object)
    second = numpy.array([{"right": 2, "left": 1}, ["value"]], dtype=object)

    assert canonical_source_digest(first) == canonical_source_digest(second)


def test_canonical_source_digest_includes_structured_dtype() -> None:
    first = numpy.array([(1, 2)], dtype=[("left", "<i4"), ("right", "<i4")])
    second = numpy.array([(1, 2)], dtype=[("right", "<i4"), ("left", "<i4")])

    assert canonical_source_digest(first) != canonical_source_digest(second)


def test_canonical_source_digest_rejects_quantized_tensors() -> None:
    tensor = torch.quantize_per_tensor(torch.tensor([1.0]), scale=0.1, zero_point=0, dtype=torch.qint8)

    with pytest.raises(TypeError, match="Cannot fingerprint quantized tensors."):
        canonical_source_digest(tensor)


def test_canonical_source_digest_rejects_unstable_values() -> None:
    with pytest.raises(
        TypeError,
        match="Cannot fingerprint rollout source value of type builtins.object.",
    ):
        canonical_source_digest(object())
