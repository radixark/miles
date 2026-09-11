"""The NVFP4 export scale must not read a tensor value back into Python."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from miles.utils.nvfp4 import nvfp4_global_decode_scale_te, nvfp4_global_encode_scale_te


class NoHostScalarRead(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, "Scale conversion read a device scalar"
        assert func != torch.ops.aten.lift_fresh.default, "Scale conversion created a host scalar tensor"
        return func(*args, **(kwargs or {}))


def legacy_encode_scale(amax, e4m3_max):
    # Frozen arithmetic from the old path, evaluated on CPU as a value oracle.
    fp4_max = torch.tensor(6.0, dtype=torch.float32)
    fp8_max = torch.tensor(float(e4m3_max), dtype=torch.float32)
    result = torch.div(fp8_max * fp4_max, amax.to(torch.float32))
    result = torch.min(result, torch.tensor(torch.finfo(torch.float32).max))
    if result.numel() == 1:
        if result == torch.tensor(0.0):
            result = torch.tensor(1.0)
    else:
        result = torch.where(result == 0.0, torch.ones_like(result), result)
    return result.reshape(amax.shape)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("e4m3_max", [256, 448])
@pytest.mark.parametrize("shape", [(), (1,), (8,), (2, 4)])
def test_global_scale_exact_without_host_scalar_read(device, e4m3_max, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Includes overflow clamping, underflow-to-zero repair and NaN propagation.
    values = [
        0.0, 1.0, 3.5, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max,
        float("inf"), -float("inf"), float("nan"),
    ]
    if torch.Size(shape).numel() == 1:
        cases = [torch.tensor(value).reshape(shape) for value in values]
    else:
        cases = [torch.tensor(values).reshape(shape)]
    for cpu_amax in cases:
        amax = cpu_amax.to(device)
        with NoHostScalarRead():
            encoded = nvfp4_global_encode_scale_te(amax, e4m3_max)
            decoded = nvfp4_global_decode_scale_te(amax, e4m3_max)
        expected = legacy_encode_scale(cpu_amax, e4m3_max)
        assert encoded.device == amax.device and encoded.shape == amax.shape
        torch.testing.assert_close(encoded.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(decoded.cpu(), torch.div(1.0, expected), rtol=0, atol=0, equal_nan=True)
