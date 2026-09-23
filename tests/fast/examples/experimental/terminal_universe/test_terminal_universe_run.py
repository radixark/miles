"""Regression checks for the Terminal Universe launch configuration."""

import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

run = import_launch_script(REPO_ROOT / "examples" / "experimental" / "terminal_universe" / "run.py")


@pytest.mark.parametrize("concurrency", [None, 256])
def test_submitted_runtime_explicitly_enables_summarization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrency: int | None) -> None:
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("HARBOR_TERMINUS_2_ENABLE_SUMMARIZE", "false")
    monkeypatch.setenv("HARBOR_TERMINUS_2_LINEAR_HISTORY", "false")
    monkeypatch.setattr(run.U, "execute_train", capture)
    args = run.ScriptArgs(
        num_nodes=4,
        run_id="260101-example",
        output_dir=str(tmp_path),
        wandb_key="",
        pause_generation_mode="in_place",
        async_max_concurrent_samples=concurrency,
    )

    run.execute(args)

    env = captured["extra_env_vars"]
    assert args.radix_raft_dir in env["PYTHONPATH"].split(":")
    assert env["HARBOR_TERMINUS_2_ENABLE_SUMMARIZE"] == "true"
    assert env["HARBOR_TERMINUS_2_LINEAR_HISTORY"] == "true"
    assert env["AGENT_MAX_INPUT_TOKENS"] == "49152"
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == "16384"
    assert env["HARBOR_MAX_SEQ_LEN"] == "65536"
    assert env["MILES_ROUTER_EXTERNAL_HOST"] == ""
    argv = shlex.split(captured["train_args"])
    assert argv[argv.index("--custom-agent-function-path") + 1] == "experiments.shi.terminal_universe.miles_agent.run"
    assert "--use-rollout-routing-replay" in argv
    assert "--use-miles-dashboard" in argv
    assert "--observe-training-entropy" in argv
    assert "--use-rollout-entropy" in argv
    assert argv[argv.index("--pause-generation-mode") + 1] == "in_place"
    assert argv[argv.index("--num-rollout") + 1] == "1000"
    assert argv[argv.index("--save-interval") + 1] == "50"
    assert argv[argv.index("--rollout-batch-size") + 1] == "8"
    assert argv[argv.index("--n-samples-per-prompt") + 1] == "16"
    assert argv[argv.index("--global-batch-size") + 1] == "128"
    if concurrency is None:
        assert "--async-max-concurrent-samples" not in argv
    else:
        assert argv[argv.index("--async-max-concurrent-samples") + 1] == str(concurrency)
    manifest = json.loads((tmp_path / args.run_id / "run_manifest.json").read_text())
    assert manifest["harness"] == {
        "enable_summarize": True,
        "linear_history": True,
        "max_input_tokens": 49152,
        "max_output_tokens": 16384,
    }


@pytest.mark.parametrize("concurrency", [0, 7])
def test_concurrency_budget_must_fit_one_prompt_group(concurrency: int) -> None:
    with pytest.raises(ValueError, match="at least one complete prompt group"):
        run.ScriptArgs(num_nodes=4, async_max_concurrent_samples=concurrency)
