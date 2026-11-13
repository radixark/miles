import pytest
import torch

from miles import platforms
from miles.platforms.xpu import XpuPlatform


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda or tpu would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda, tpu or xpu"):
        platforms._select_platform()


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper


def test_xpu_is_selected_by_name(monkeypatch):
    """MILES_ACCELERATOR=xpu picks the XPU platform."""
    monkeypatch.setenv("MILES_ACCELERATOR", "xpu")
    assert isinstance(platforms._select_platform(), XpuPlatform)


@pytest.mark.parametrize(("accelerator", "expected"), [(torch.device("xpu"), "xpu"), (None, "cuda")])
def test_without_the_env_var_the_platform_follows_torch(monkeypatch, accelerator, expected):
    """Each process detects its own device. Without an accelerator CUDA is kept as the default."""
    monkeypatch.delenv("MILES_ACCELERATOR", raising=False)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: accelerator)
    assert platforms._select_platform().device_type == expected
