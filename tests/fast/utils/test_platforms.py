import pytest

from miles import platforms


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda or tpu would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda or tpu"):
        platforms._select_platform()


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper
