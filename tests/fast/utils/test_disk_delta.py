import numpy as np
import pytest
import safetensors.numpy

from miles.utils import disk_delta
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


def test_available_cpus_use_quota_then_affinity(tmp_path, monkeypatch) -> None:
    cpu_max = tmp_path / "cpu.max"
    monkeypatch.setattr(disk_delta, "_CPU_MAX_PATH", str(cpu_max))
    monkeypatch.setattr(disk_delta.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)

    assert disk_delta.get_available_cpus() == 6
    cpu_max.write_text("max 100000\n")
    assert disk_delta.get_available_cpus() == 6
    cpu_max.write_text("850000 100000\n")
    assert disk_delta.get_available_cpus() == 8
    cpu_max.write_text("50000 100000\n")
    assert disk_delta.get_available_cpus() == 1
