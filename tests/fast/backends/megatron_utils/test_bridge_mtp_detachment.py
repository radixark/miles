"""CPU regression tests for MTP detachment on bridge-built providers."""

import argparse
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize(
    ("detach", "initial_detach"),
    [(True, False), (True, True), (False, False), (False, True)],
)
def test_bridge_mtp_detachment(
    apply_bridge_runtime_config: Callable,
    runtime_args: argparse.Namespace,
    detach: bool,
    initial_detach: bool,
) -> None:
    """The provider takes --mtp-detach-heads from args, whatever the bridge's HF config set."""
    runtime_args.backend.mtp_detach_heads = detach
    provider = SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=initial_detach)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_detach_heads is detach
    assert provider.mtp_num_layers == 1


@pytest.mark.parametrize(
    ("mtp_args", "expected_layers"),
    [({}, 1), ({"mtp_num_layers": None}, None), ({"mtp_num_layers": 2}, 2)],
    ids=["absent", "disabled", "enabled"],
)
def test_bridge_mtp_runtime_depth(apply_bridge_runtime_config, runtime_args, mtp_args, expected_layers):
    vars(runtime_args.backend).update(mtp_args)
    provider = SimpleNamespace(mtp_num_layers=1)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_num_layers == expected_layers
