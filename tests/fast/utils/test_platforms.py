import pytest

from miles import platforms


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda or tpu would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda or tpu"):
        platforms._select_platform()


def test_sail_is_probed_behind_the_cuda_default(monkeypatch):
    """SAIL runs on torch.cuda, so only the cuda-family probe can tell it apart from NVIDIA."""
    monkeypatch.delenv("MILES_ACCELERATOR", raising=False)
    monkeypatch.setattr(platforms, "_is_sail", lambda: True)
    platform = platforms._select_platform()
    assert type(platform).__name__ == "SailPlatform"
    assert platform.device_type == "cuda"
    monkeypatch.setattr(platforms, "_is_sail", lambda: False)
    assert type(platforms._select_platform()) is platforms.Platform


def test_is_sail_probes_the_device_node_and_zw_device_name(monkeypatch):
    monkeypatch.setattr(platforms.os.path, "exists", lambda p: p == "/dev/alixpu")
    monkeypatch.setattr(platforms.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(platforms.torch.cuda, "get_device_name", lambda: "ZW-M890P")
    assert platforms._is_sail() is True
    monkeypatch.setattr(platforms.torch.cuda, "get_device_name", lambda: "NVIDIA H200")
    assert platforms._is_sail() is False
    monkeypatch.setattr(platforms.torch.cuda, "is_available", lambda: False)
    assert platforms._is_sail() is False
    monkeypatch.setattr(platforms.os.path, "exists", lambda p: False)
    assert platforms._is_sail() is False


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper
