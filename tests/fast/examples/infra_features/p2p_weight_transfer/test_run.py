import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
RUN_SCRIPT = REPO_ROOT / "examples" / "infra_features" / "p2p_weight_transfer" / "run.py"

_PROFILES_PINNING_A_ROTARY_BASE = ["Qwen3-235B-A22B-Instruct-2507", "Qwen3-30B-A3B", "GLM-4.5-Air"]


@pytest.fixture(scope="module")
def run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p2p_weight_transfer_run", RUN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand_model_args(run_module: ModuleType, model_name: str) -> list[str]:
    """Run the very snippet run.py hands to bash, so this keeps testing the real command."""
    command = run_module.build_model_args_command(run_module.RUN_CONFIGS[model_name])
    result = subprocess.run(
        f'{command} && printf "%s\\n" "${{MODEL_ARGS[@]}}"',
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_model_args_env_is_empty_when_the_profile_pins_no_rotary_base(run_module: ModuleType) -> None:
    """A profile without rotary_base must not inject any MODEL_ARGS_* override."""
    cfg = run_module.RUN_CONFIGS["Qwen3-4B"]
    assert cfg.rotary_base is None
    assert run_module.build_model_args_env(cfg) == {}


def test_model_args_env_carries_the_rotary_base_a_profile_pins(run_module: ModuleType) -> None:
    """A profile pinning rotary_base must surface it as the MODEL_ARGS_* name the model definition reads."""
    cfg = run_module.RUN_CONFIGS["Qwen3-235B-A22B-Instruct-2507"]
    assert cfg.rotary_base == 5000000
    assert run_module.build_model_args_env(cfg) == {"MODEL_ARGS_ROTARY_BASE": "5000000"}


@pytest.mark.parametrize("model_name", _PROFILES_PINNING_A_ROTARY_BASE)
def test_a_pinned_rotary_base_reaches_the_expanded_model_args(run_module: ModuleType, model_name: str) -> None:
    """The knob has to survive into the shell that expands MODEL_ARGS, not only into ray's runtime env."""
    tokens = expand_model_args(run_module, model_name)

    assert str(run_module.RUN_CONFIGS[model_name].rotary_base) == tokens[tokens.index("--rotary-base") + 1]


@pytest.mark.parametrize("model_name", sorted({"Qwen3-4B", "GLM-4.7-Flash", *_PROFILES_PINNING_A_ROTARY_BASE}))
def test_every_profile_expands_to_a_usable_argv(run_module: ModuleType, model_name: str) -> None:
    """A profile naming a model that no longer exists would submit a job with no architecture flags."""
    tokens = expand_model_args(run_module, model_name)

    assert tokens
    assert tokens[0].startswith("--")
