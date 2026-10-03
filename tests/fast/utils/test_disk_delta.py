import numpy as np
import pytest
import safetensors.numpy

from miles.utils.disk_delta import make_tensor_reader


def test_tensor_reader_validates_declared_layout(tmp_path):
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)
    safetensors.numpy.save_file({"weight": expected}, tmp_path / "model.safetensors")
    read = make_tensor_reader(str(tmp_path))

    actual = read("weight", expected_dtype="F16", expected_shape=(2, 3))
    np.testing.assert_array_equal(actual, expected.view(np.uint8).reshape(-1))

    with pytest.raises(ValueError, match="dtype=F16, shape=\\(2, 3\\)"):
        read("weight", expected_dtype="BF16", expected_shape=(2, 3))
    with pytest.raises(ValueError, match="dtype=F16, shape=\\(2, 3\\)"):
        read("weight", expected_dtype="F16", expected_shape=(3, 2))
