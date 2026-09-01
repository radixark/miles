from __future__ import annotations

import json
import textwrap
from argparse import Namespace
from types import SimpleNamespace

import pytest

from examples.experimental.eval.parallel_sft.parallel_command_eval import (
    CommandResult,
    EvalCommand,
    ParallelCommandEvalFn,
)
from miles.ray.rollout.metrics import log_eval_rollout_data
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput


async def test_parallel_command_eval_expands_endpoint_and_merges_metrics(monkeypatch, tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            commands:
              - name: terminal_bench
                argv: [runner, tb, '{agent_openai_base_url}', '{litellm_model}']
                metrics_path: '{output_dir}/terminal_bench.json'
              - name: hle
                argv: [runner, hle, '{openai_base_url}', '{model}']
                metrics_path: '{output_dir}/hle.json'
            """
        )
    )
    output_root = tmp_path / "results"
    monkeypatch.setenv("MILES_PARALLEL_EVAL_CONFIG", str(config_path))
    monkeypatch.setenv("MILES_PARALLEL_EVAL_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("MILES_PARALLEL_EVAL_MODEL", "checkpoint-model")
    monkeypatch.setenv("MILES_PARALLEL_EVAL_AGENT_BASE_URL", "http://100.64.0.2:31000/v1")

    calls = []

    class FakeProcess:
        def __init__(self, benchmark, env):
            self.benchmark = benchmark
            self.env = env
            self.returncode = 0

        async def wait(self):
            output_dir = output_root / "step_200"
            if self.benchmark == "tb":
                payload = {"metrics": {"accuracy": 0.4, "completed": 5}}
                filename = "terminal_bench.json"
            else:
                payload = {"metrics": {"accuracy": 0.5}, "rewards": [1.0, 0.0]}
                filename = "hle.json"
            (output_dir / filename).write_text(json.dumps(payload))
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return FakeProcess(argv[1], kwargs["env"])

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    args = Namespace(hf_checkpoint="/models/base")
    eval_fn = ParallelCommandEvalFn(RolloutFnConstructorInput(args=args, data_source=None))
    state = SimpleNamespace(args=Namespace(sglang_router_ip="10.0.0.2", sglang_router_port=31000))

    output = await eval_fn(
        RolloutFnEvalInput(
            rollout_id=200,
            generate_state=state,
            weight_version="200",
            hf_dir="/snapshots/step_200",
        )
    )

    assert len(calls) == 2
    assert calls[0][0][2:] == ("http://100.64.0.2:31000/v1", "openai/checkpoint-model")
    assert calls[1][0][2:] == ("http://10.0.0.2:31000/v1", "checkpoint-model")
    assert calls[0][1]["MILES_EVAL_CHECKPOINT_DIR"] == "/snapshots/step_200"
    assert calls[0][1]["MILES_EVAL_AGENT_OPENAI_BASE_URL"] == "http://100.64.0.2:31000/v1"
    assert calls[0][1]["MILES_EVAL_TRAINING_STEP"] == "201"
    assert calls[0][1]["MILES_EVAL_WEIGHT_VERSION"] == "200"
    assert output.metrics["eval/terminal_bench/accuracy"] == 0.4
    assert output.metrics["eval/hle/accuracy"] == 0.5
    assert output.data == {"hle": {"rewards": [1.0, 0.0]}}

    logged = {}

    def fake_tracking_log(_args, metrics, *, step_key):
        logged.update(metrics)
        logged["step_key"] = step_key

    monkeypatch.setattr("miles.ray.rollout.metrics.tracking.log", fake_tracking_log)
    logging_args = Namespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
        wandb_always_use_train_step=False,
    )

    log_eval_rollout_data(200, logging_args, output.data, output.metrics)

    assert logged["eval/terminal_bench/accuracy"] == 0.4
    assert logged["eval/hle/accuracy"] == 0.5
    assert logged["eval/hle"] == 0.5
    assert logged["eval/step"] == 200
    assert logged["step_key"] == "eval/step"


def test_parallel_command_eval_defaults_to_sglang_model_path(monkeypatch, tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("commands:\n  - name: smoke\n    argv: [runner]\n")
    monkeypatch.setenv("MILES_PARALLEL_EVAL_CONFIG", str(config_path))
    monkeypatch.delenv("MILES_PARALLEL_EVAL_MODEL", raising=False)

    args = Namespace(hf_checkpoint="/models/Qwen3.6-35B-A3B")
    eval_fn = ParallelCommandEvalFn(RolloutFnConstructorInput(args=args, data_source=None))

    assert eval_fn._model == "/models/Qwen3.6-35B-A3B"


async def test_parallel_command_eval_respects_per_command_intervals(monkeypatch, tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            commands:
              - name: hle
                interval_steps: 50
                argv: [runner, hle]
              - name: tb21
                interval_steps: 200
                argv: [runner, tb21]
            """
        )
    )
    monkeypatch.setenv("MILES_PARALLEL_EVAL_CONFIG", str(config_path))
    monkeypatch.setenv("MILES_PARALLEL_EVAL_OUTPUT_DIR", str(tmp_path / "results"))
    calls: list[tuple[str, str]] = []

    async def fake_run_command(command, context, _output_dir):
        calls.append((command.name, context["training_step"]))
        return CommandResult(
            name=command.name,
            returncode=0,
            duration_seconds=1.0,
            metrics={},
            rewards=None,
        )

    monkeypatch.setattr(
        "examples.experimental.eval.parallel_sft.parallel_command_eval._run_command",
        fake_run_command,
    )
    eval_fn = ParallelCommandEvalFn(RolloutFnConstructorInput(args=Namespace(hf_checkpoint="/models/base"), data_source=None))
    state = SimpleNamespace(args=Namespace(sglang_router_ip="10.0.0.2", sglang_router_port=31000))

    await eval_fn(RolloutFnEvalInput(rollout_id=49, generate_state=state, hf_dir="/snapshots/step_49"))
    assert calls == [("hle", "50")]

    calls.clear()
    await eval_fn(RolloutFnEvalInput(rollout_id=199, generate_state=state, hf_dir="/snapshots/step_199"))
    assert calls == [("hle", "200"), ("tb21", "200")]


@pytest.mark.parametrize("interval_steps", [0, -1, True, "1.5"])
def test_eval_command_rejects_invalid_intervals(interval_steps):
    with pytest.raises(ValueError, match="interval_steps must be a positive integer"):
        EvalCommand.parse({"name": "bad", "argv": ["runner"], "interval_steps": interval_steps})
