from argparse import Namespace

import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles import platforms
from miles.backends.fsdp_utils.arguments import validate_platform_args
from miles.platforms.xpu import XpuPlatform


@pytest.fixture
def on_xpu(monkeypatch):
    """The process platform takes the XpuPlatform facts the startup check reads."""
    for fact in ("triton_kernels", "weight_transfer_modes"):
        monkeypatch.setattr(platforms.current_platform, fact, getattr(XpuPlatform, fact))


def _args(**overrides) -> Namespace:
    base = dict(log_probs_backend="torch", colocate=False, update_weight_transfer_mode="broadcast")
    return Namespace(**{**base, **overrides})


@pytest.mark.parametrize(
    "unsupported",
    [
        dict(update_weight_transfer_mode="p2p"),
        dict(update_weight_transfer_mode="disk-delta"),
        dict(colocate=True),
    ],
)
def test_xpu_refuses_the_weight_sync_modes_with_a_cuda_call_on_their_path(on_xpu, unsupported):
    """p2p stages its replica on a CUDA device, disk-delta waits on a CUDA stream, colocate shares CUDA IPC handles."""
    with pytest.raises(ValueError, match="update-weight-transfer-mode broadcast or http-lora"):
        validate_platform_args(_args(**unsupported))


@pytest.mark.parametrize("mode", ["broadcast", "http-lora"])
def test_xpu_takes_the_weight_sync_modes_that_cross_no_cuda_call(on_xpu, mode):
    validate_platform_args(_args(update_weight_transfer_mode=mode))


def test_xpu_refuses_the_triton_log_prob_kernel(on_xpu):
    with pytest.raises(ValueError, match="on .* use --log-probs-backend torch"):
        validate_platform_args(_args(log_probs_backend="fused"))


def test_cuda_takes_every_mode_the_check_refuses_on_xpu():
    """The base platform sets no limit, so this check changes nothing for CUDA and ROCm runs."""
    validate_platform_args(_args(log_probs_backend="fused", colocate=True, update_weight_transfer_mode="p2p"))


def test_an_xpu_fsdp_launch_is_refused_while_its_arguments_are_parsed(on_xpu):
    """The FSDP backend runs the check itself; before, only torchtitan read the platform's limits."""
    with pytest.raises(ValueError, match="update-weight-transfer-mode broadcast or http-lora"):
        parse_fsdp_test_config("--colocate")
    with pytest.raises(ValueError, match="on .* use --log-probs-backend torch"):
        parse_fsdp_test_config("--log-probs-backend", "fused")
    parse_fsdp_test_config()
