"""CPU contract checks for the TMax adapter (requires the Harbor extra)."""

import subprocess

import pytest

pytest.importorskip("harbor.agents.base")
from examples.experimental.tmax_score_centering import agent


def test_shell_preserves_cwd_and_exports_after_failed_command(monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "cwd").write_text(str(tmp_path))
    (state / "env").write_text("")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(agent, "_STATE", str(state))
    first = subprocess.run(agent._wrap_command(f"cd {work}; export TMAX_TEST_VALUE=retained; false"), shell=True)
    second = subprocess.run(
        agent._wrap_command('printf "%s:%s" "$PWD" "$TMAX_TEST_VALUE"'), shell=True, capture_output=True, text=True
    )
    assert first.returncode == 1
    assert second.stdout == f"{work}:retained"


def test_observation_keeps_head_tail_and_exit_status():
    observation = agent._observation("A" * 6000 + "B" * 6000, "", 7)
    assert "2000 chars elided" in observation
    assert "A" * 5000 in observation and "B" * 5000 in observation
    assert observation.endswith("(exit_code=7)")


def test_wandb_launcher_never_serializes_credentials(monkeypatch):
    from examples.experimental.tmax_score_centering.run_qwen3_5_9b import _wandb_args

    monkeypatch.setenv("WANDB_API_KEY", "test-secret-never-in-command")
    result = _wandb_args("test-run")
    assert "--use-wandb" in result
    assert "test-secret-never-in-command" not in result
    assert "--wandb-key" not in result
