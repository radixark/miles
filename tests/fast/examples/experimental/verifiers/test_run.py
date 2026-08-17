import json
import shlex

import pytest
from examples.experimental.verifiers import run
from tests.fast.utils.command_recorder import record_commands

import miles.utils.external_utils.command_utils as U

LEGACY_ROLLOUT_ENV = "MILES_USE_LEGACY_ROLLOUT_V1"


def _rollout_config(submit_command: str) -> tuple[str, dict[str, str]]:
    argv = shlex.split(submit_command)
    rollout_fn = argv[argv.index("--rollout-function-path") + 1]
    runtime_env_arg = next(arg for arg in argv if arg.startswith("--runtime-env-json="))
    runtime_env = json.loads(runtime_env_arg.split("=", 1)[1])["env_vars"]
    return rollout_fn, runtime_env


@pytest.mark.parametrize(
    ("ambient_value", "extra_env_vars", "expected_rollout_fn", "expected_runtime_value"),
    [
        (None, "", "verifiers_rollout.VerifiersRolloutFn", None),
        ("1", "", "verifiers_rollout.generate_rollout", "1"),
        ("0", f"{LEGACY_ROLLOUT_ENV}=1", "verifiers_rollout.generate_rollout", "1"),
        ("1", f"{LEGACY_ROLLOUT_ENV}=0", "verifiers_rollout.VerifiersRolloutFn", "0"),
    ],
)
def test_adapter_and_ray_runtime_use_the_same_legacy_flag(
    monkeypatch,
    tmp_path,
    ambient_value,
    extra_env_vars,
    expected_rollout_fn,
    expected_runtime_value,
):
    commands = record_commands(monkeypatch)
    monkeypatch.setattr(U, "check_has_nvlink", lambda: False)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    if ambient_value is None:
        monkeypatch.delenv(LEGACY_ROLLOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(LEGACY_ROLLOUT_ENV, ambient_value)

    run.execute(
        run.ScriptArgs(
            verifiers_config=str(tmp_path / "verifiers.toml"),
            extra_env_vars=extra_env_vars,
        )
    )

    rollout_fn, runtime_env = _rollout_config(commands[-1])
    assert rollout_fn == expected_rollout_fn
    if expected_runtime_value is None:
        assert LEGACY_ROLLOUT_ENV not in runtime_env
    else:
        assert runtime_env[LEGACY_ROLLOUT_ENV] == expected_runtime_value
