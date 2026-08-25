import json
import shlex

import pytest

from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_EXAMPLE_DIR = REPO_ROOT / "examples" / "experimental" / "terminus-compaction"
run = import_launch_script(_EXAMPLE_DIR / "run.py")


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


@pytest.fixture
def args():
    return run.ScriptArgs(
        run_id="260101-example",
        model_dir="/models",
        output_dir="/output",
        session_server_ip="0.0.0.0",
        router_external_host="trainer.example",
        agent_server_url="http://agent.example:11000",
    )


def test_recipe_enables_compaction_aware_session_training(args, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = shlex.split(run._build_train_args(args))

    assert _value(argv, "--use-session-server") == "v2"
    assert _value(argv, "--session-server-ip") == "0.0.0.0"
    assert _value(argv, "--tito-model") == "glm47"
    assert _value(argv, "--rollout-batch-size") == "4"
    assert _value(argv, "--n-samples-per-prompt") == "8"
    assert _value(argv, "--global-batch-size") == "32"
    assert _value(argv, "--num-rollout") == "100"
    assert _value(argv, "--rollout-max-response-len") == "8192"


def test_recipe_records_dashboard_traces_and_honors_gpu_count(args, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = shlex.split(run._build_train_args(args))

    assert _value(argv, "--dump-details") == "/output/260101-example/details"
    assert _value(argv, "--num-gpus-per-node") == "8"
    assert "--use-miles-dashboard" in argv
    assert "--use-rollout-entropy" in argv
    assert "--observe-training-entropy" in argv
    assert "--log-multi-turn" in argv
    assert "--rollout-num-gpus" not in argv


def test_session_bind_override_is_optional(args, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args.session_server_ip = ""

    assert "--session-server-ip" not in shlex.split(run._build_train_args(args))


def test_agent_runtime_environment_reuses_harbor_adapters(args):
    env = run._extra_env_vars(args)

    assert env == {
        "PYTHONPATH": str(REPO_ROOT / "examples" / "swe-agent-harbor-docker"),
        "AGENT_SERVER_URL": "http://agent.example:11000",
        "AGENT_MODEL_NAME": "model",
        "AGENT_TRIAL_TIMEOUT": "7200",
        "MILES_ROUTER_EXTERNAL_HOST": "trainer.example",
    }


def test_agent_runtime_environment_omits_optional_hosts(args):
    args.router_external_host = ""
    args.miles_host_ip = ""

    env = run._extra_env_vars(args)

    assert "MILES_ROUTER_EXTERNAL_HOST" not in env
    assert "MILES_HOST_IP" not in env


def test_bundled_subset_contains_23_terminus_tasks():
    rows = [json.loads(line) for line in (_EXAMPLE_DIR / "tb2_23_tasks.jsonl").read_text().splitlines()]
    instance_ids = [row["metadata"]["instance_id"] for row in rows]

    assert len(rows) == 23
    assert len(set(instance_ids)) == 23
    assert all(row["prompt"] == row["metadata"]["instance_id"] for row in rows)
    assert all(row["metadata"]["agent_name"] == "terminus-2" for row in rows)
