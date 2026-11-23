"""`preflight_aenv` must stop an agentenv launch whose `aenv` cannot build task Dockerfiles."""

import aenv_cli
import pytest


@pytest.mark.parametrize(
    ("version_output", "exit_code", "error"),
    [
        (None, 0, "not on PATH"),
        ("aenv 0.2.2", 0, "is aenv 0.2.2"),
        ("AgentENV CLI dev build", 0, "cannot read a version"),
        ("aenv 0.2.5", 1, "could not run"),
    ],
    ids=["missing", "too_old", "no_version", "fails_to_run"],
)
def test_an_unusable_aenv_is_refused_with_the_install_command(fake_aenv, version_output, exit_code, error):
    if version_output is not None:
        fake_aenv(version_output, exit_code=exit_code)
    with pytest.raises(RuntimeError, match=error) as excinfo:
        aenv_cli.preflight_aenv("agentenv")
    assert "install-cli.sh | bash" in str(excinfo.value)


@pytest.mark.parametrize("version_output", ["aenv 0.2.3", "aenv 0.10.0"], ids=["floor", "two_digit_minor"])
def test_a_new_enough_aenv_passes(fake_aenv, version_output):
    fake_aenv(version_output)
    aenv_cli.preflight_aenv("agentenv")


def test_other_environment_types_need_no_aenv(fake_aenv):
    aenv_cli.preflight_aenv("e2b")
