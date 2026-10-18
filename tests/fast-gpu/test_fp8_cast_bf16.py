"""tools/fp8_cast_bf16.py: block-scaled FP8 dequantization (2-D weights and packed expert tensors) and the
BF16 config it writes."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

import importlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the Triton kernel")

# The tool imports its sibling module by bare name, as it does when run as a script.
_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture(scope="module")
def cast_tool():
    sys.path.insert(0, str(_TOOLS_DIR))
    try:
        yield importlib.import_module("fp8_cast_bf16")
    finally:
        sys.path.remove(str(_TOOLS_DIR))


def _reference(x: torch.Tensor, s: torch.Tensor, block: int) -> torch.Tensor:
    n, k = x.shape[-2:]
    scale = s.repeat_interleave(block, dim=-2)[..., :n, :].repeat_interleave(block, dim=-1)[..., :k]
    return (x.float() * scale).to(torch.bfloat16)


@pytest.mark.parametrize("leading", [(), (3,), (2, 2)])
def test_weight_dequant_matches_reference(cast_tool, leading):
    torch.manual_seed(0)
    block, n, k = 128, 300, 260  # not multiples of the block, so the edge masks are exercised
    x = torch.randn(*leading, n, k, device="cuda").to(torch.float8_e4m3fn)
    s = torch.rand(*leading, -(-n // block), -(-k // block), device="cuda") + 0.5

    torch.set_default_dtype(torch.bfloat16)
    try:
        y = cast_tool.weight_dequant(x, s, block_size=block)
    finally:
        torch.set_default_dtype(torch.float32)

    assert y.shape == x.shape and y.dtype == torch.bfloat16
    torch.testing.assert_close(y, _reference(x, s, block), rtol=0, atol=0)


def test_bf16_config_drops_quantization_config(cast_tool):
    fp8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    config = {
        "architectures": ["Step5"],
        "quantization_config": fp8,
        "text_config": {"hidden_size": 4096, "quantization_config": fp8},
    }

    assert cast_tool.bf16_config(config) == {"architectures": ["Step5"], "text_config": {"hidden_size": 4096}}
    assert config["quantization_config"] == fp8 and config["text_config"]["quantization_config"] == fp8
