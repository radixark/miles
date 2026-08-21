"""Locks the arch table against the suite inventory it describes."""

import pytest
from tests.ci.ci_register import HWBackend, register_cpu_ci
from tests.ci.hardware import CUDA_STAGE_ARCH, KNOWN_ARCHES, auto_arch
from tests.ci.run_suite import CI_SUITES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def test_every_cuda_suite_declares_an_arch():
    # Drift here is what let a suite name sit in test files for months with no
    # workflow job behind it.
    assert set(CUDA_STAGE_ARCH) == set(CI_SUITES[HWBackend.CUDA])


def test_every_declared_arch_is_known():
    assert set(CUDA_STAGE_ARCH.values()) <= set(KNOWN_ARCHES)


def test_hopper_is_the_default_arch():
    # One B200 host against four-plus Hopper hosts: a test that could run on
    # either must not land on the scarce one by default.
    assert KNOWN_ARCHES[0] == "hopper"


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        (["hopper"], "hopper"),
        (["blackwell"], "blackwell"),
        (["hopper", "blackwell"], "hopper"),
        (["blackwell", "hopper"], "hopper"),
    ],
)
def test_auto_arch_follows_preference_not_declaration_order(hardware, expected):
    assert auto_arch(hardware) == expected


def test_auto_arch_rejects_an_unknown_set():
    with pytest.raises(ValueError, match="no known arch"):
        auto_arch(["ampere"])
