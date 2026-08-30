from __future__ import annotations

import json
import textwrap
from argparse import Namespace
from types import SimpleNamespace

from examples.experimental.eval.parallel_sft.parallel_command_eval import ParallelCommandEvalFn
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput


async def test_parallel_command_eval_expands_endpoint_and_merges_metrics(monkeypatch, tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            commands:
              - name: terminal_bench
                argv: [runner, tb, '{openai_base_url}', '{litellm_model}']
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

    calls = []

    class FakeProcess:
        def __init__(self, benchmark, env):
            self.benchmark = benchmark
            self.env = env
            self.returncode = 0

        async def wait(self):
            output_dir = output_root / "step_200"
            if self.benchmark == "tb":
                payload = {"metrics": {"pass_rate": 0.4, "completed": 5}}
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
    assert calls[0][0][2:] == ("http://10.0.0.2:31000/v1", "openai/checkpoint-model")
    assert calls[1][0][2:] == ("http://10.0.0.2:31000/v1", "checkpoint-model")
    assert calls[0][1]["MILES_EVAL_CHECKPOINT_DIR"] == "/snapshots/step_200"
    assert calls[0][1]["MILES_EVAL_WEIGHT_VERSION"] == "200"
    assert output.metrics["eval/terminal_bench/pass_rate"] == 0.4
    assert output.metrics["eval/hle/accuracy"] == 0.5
    assert output.data == {"hle": {"rewards": [1.0, 0.0]}}
