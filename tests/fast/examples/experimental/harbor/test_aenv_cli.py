"""`preflight_aenv` must stop an agentenv run whose `aenv` cannot build task Dockerfiles."""

import aenv_cli
import pytest

from miles.utils.retry_utils import NonRetryableError


@pytest.mark.parametrize(
    ("version_output", "error"),
    [
        (None, "not on PATH"),
        ("aenv 0.2.2", "is aenv 0.2.2"),
        ("AgentENV CLI dev build", "cannot read a version"),
    ],
    ids=["missing", "too_old", "no_version"],
)
def test_an_unusable_aenv_is_refused_with_the_install_command(fake_aenv, version_output, error):
    if version_output is not None:
        fake_aenv(version_output)
    with pytest.raises(NonRetryableError, match=error) as excinfo:
        aenv_cli.preflight_aenv("agentenv")
    assert "install-cli.sh | bash" in str(excinfo.value)


@pytest.mark.parametrize("version_output", ["aenv 0.2.3", "aenv 0.10.0"], ids=["floor", "two_digit_minor"])
def test_a_new_enough_aenv_passes(fake_aenv, version_output):
    fake_aenv(version_output)
    aenv_cli.preflight_aenv("agentenv")


def test_other_environment_types_need_no_aenv(fake_aenv):
    aenv_cli.preflight_aenv("e2b")


@pytest.mark.parametrize(
    ("script", "error_type"),
    [
        ("#!/bin/sh\nexit 1\n", NonRetryableError),
        ("not a program\n", NonRetryableError),
        ("#!/bin/sh\nkill -9 $$\n", RuntimeError),
        ("#!/bin/sh\nexec /bin/sleep 5\n", RuntimeError),
    ],
    ids=["nonzero_exit", "cannot_exec", "killed_by_signal", "timeout"],
)
def test_a_failed_version_call_stops_the_run_only_when_it_would_fail_again(fake_aenv, monkeypatch, script, error_type):
    """A broken binary must stop the run; a passing failure must cost only its trial."""
    monkeypatch.setattr(aenv_cli, "_VERSION_TIMEOUT_S", 1)
    fake_aenv(script=script)
    with pytest.raises((RuntimeError, NonRetryableError), match="could not run") as excinfo:
        aenv_cli.preflight_aenv("agentenv")
    assert type(excinfo.value) is error_type
