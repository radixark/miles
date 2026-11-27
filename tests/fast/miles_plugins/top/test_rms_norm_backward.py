import pytest

rms_norm_backward = pytest.importorskip("miles_plugins.top.kernels.rms_norm_backward")


def test_element_count_is_a_runtime_argument():
    """A constexpr element count compiled the residual kernel again for every microbatch token count."""
    params = {param.name: param for param in rms_norm_backward._finish_residual.params}
    assert not params["N"].is_constexpr
    assert params["N"].do_not_specialize
