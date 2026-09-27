"""scripts/amd/run_deepseek_v4.py: what the launcher emits."""

import shlex

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.fixture
def launcher(monkeypatch):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_deepseek_v4.py")
    module.recording = recording
    return module


def _train_argv(launcher, tmp_path, **overrides):
    call_entrypoint(launcher, "train", overrides, sandbox=tmp_path)
    command = launcher.recording.commands[-1]
    assert "ray job submit" in command
    return command


def _flag(command: str, name: str) -> str | None:
    tokens = shlex.split(command.split(" -- python3 ", 1)[1])
    values = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == name]
    assert len(values) <= 1, f"{name} emitted {len(values)} times"
    return values[0] if values else None


FOUR_MI355X_NODES = dict(num_nodes=4, num_gpus_per_node=8, mode="normal")


def test_the_dapo_task_is_graded_by_the_answer_format_it_asks_for(launcher, tmp_path):
    command = _train_argv(launcher, tmp_path, **FOUR_MI355X_NODES)

    assert _flag(command, "--rm-type") == "dapo"
    assert _flag(command, "--reward-key") == "score"
    assert _flag(command, "--eval-reward-key") == "acc"
    assert _flag(command, "--sglang-context-length") == "16384"
    assert int(_flag(command, "--rollout-max-response-len")) < 16384


def test_aime_eval_is_graded_on_the_boxed_answer(launcher, tmp_path):
    """AIME prompts do not ask for "Answer:", so the DAPO grader would score every sample -1."""
    command = _train_argv(launcher, tmp_path, **FOUR_MI355X_NODES)

    assert _flag(command, "--eval-prompt-data") is None
    assert _flag(command, "--eval-config").startswith("base64:")
    (config,) = [text for text in launcher.recording.pseudo_files if "aime-2024.jsonl" in text]
    assert "rm_type: dapo_boxed" in config
