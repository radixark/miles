"""CPU benchmark of real rollout scheduling, agentic generation and sample collection.

The agent and inference backend are synthetic separate processes. Session server,
TITO, collect/decode, reward dispatch and the rollout function run production code.
Model weights and GPUs are not needed; only the public tokenizer is loaded.

Run: python -m tests.manual.session.bench_agentic_rollout --json-out result.json
The first complete rollout is warmup. Timings exclude startup and validation.
"""

import argparse
import base64
import gc
import json
import multiprocessing
import os
import signal
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from tests.manual.session import _rollout_benchmark_agent as agent
from tests.manual.session.bench_session_server_overhead import _build_turn_specs, _positive_int, _terminate_proc

NUM_LAYERS = 48
TOPK = 8


def build_args(bench, prompt_path, backend_port):
    from miles.utils.arguments import parse_args

    argv = [
        "bench_agentic_rollout",
        "--train-backend",
        "fsdp",
        "--ci-test",
        "--hf-checkpoint",
        bench.hf_checkpoint,
        "--prompt-data",
        str(prompt_path),
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--rollout-batch-size",
        str(bench.sessions),
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        str(bench.repetitions + 1),
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-server-concurrency",
        str(bench.sessions),
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(backend_port),
        "--rollout-max-response-len",
        "65536",
        "--custom-generate-function-path",
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path",
        "tests.manual.session._rollout_benchmark_agent.run_agent",
        "--custom-agent-function-mode",
        "inline",
        "--custom-rm-path",
        "tests.manual.session._rollout_benchmark_agent.reward",
        "--use-session-server",
        "v2",
        "--session-server-workers",
        "1",
        "--tito-model",
        "qwen3",
        "--use-rollout-routing-replay",
        "--pause-generation-mode",
        "in_place",
    ]
    with patch.object(sys, "argv", argv):
        args = parse_args()
    args.num_layers = NUM_LAYERS
    args.moe_router_topk = TOPK
    return args


def validate_samples(samples, specs, sessions):
    import numpy as np

    from miles.utils.types import Sample

    assert len(samples) == sessions, (len(samples), sessions)
    expected_tokens = specs[-1].expected_prompt_token_ids + [
        pair[1] for pair in json.loads(specs[-1].response_body)["choices"][0]["meta_info"]["output_token_logprobs"]
    ]
    expected_shape = (len(expected_tokens) - 1, NUM_LAYERS, TOPK)
    expected_r3 = np.frombuffer(
        b"".join(
            base64.b64decode(json.loads(spec.response_body)["choices"][0]["meta_info"]["routed_experts"])
            for spec in specs
        ),
        dtype=np.int32,
    ).reshape(expected_shape)
    for sample in samples:
        assert sample.status == Sample.Status.COMPLETED, sample.status
        assert sample.metadata["agent_metrics"]["turns_ok"] == len(specs)
        assert sample.tokens == expected_tokens
        assert sample.reward == 1.0
        assert sample.rollout_routed_experts.shape == expected_shape
        assert sample.rollout_routed_experts.dtype == np.int32
        np.testing.assert_array_equal(sample.rollout_routed_experts, expected_r3)
        assert len(sample.rollout_log_probs) == sample.response_length
        assert len(sample.loss_mask) == sample.response_length
    return sum(sample.rollout_routed_experts.nbytes for sample in samples)


def run_benchmark(bench):
    import psutil

    from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
    from miles.rollout.data_source import RolloutDataSourceWithBuffer
    from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
    from miles.rollout.session.config import compute_session_server_config
    from miles.rollout.session.server import run_session_server
    from miles.rollout.session.types import SessionServerInstance
    from miles.utils import http_utils
    from miles.utils.async_utils import run
    from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
    from miles.utils.http_utils import find_available_port, wait_for_server_ready
    from miles.utils.processing_utils import load_tokenizer

    template, template_kwargs = resolve_fixed_chat_template("qwen3")
    tokenizer = load_tokenizer(bench.hf_checkpoint, chat_template_path=template, trust_remote_code=True)
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen3", chat_template_kwargs=template_kwargs)
    specs = _build_turn_specs(
        tokenizer,
        tito,
        turns=bench.turns,
        input_tokens=bench.input_tokens,
        output_tokens=bench.output_tokens,
        r3_scale=NUM_LAYERS * TOPK * 4,
        incremental_r3=True,
        tool_appends=True,
    )
    ctx = multiprocessing.get_context("spawn")
    processes = []
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="miles-rollout-bench-") as temp:
            prompt_path = Path(temp) / "prompts.jsonl"
            prompt_path.write_text(
                "".join(
                    json.dumps({"prompt": [{"role": "user", "content": f"task {i}"}], "label": ""}) + "\n"
                    for i in range(bench.sessions * (bench.repetitions + 1))
                )
            )
            backend_port = find_available_port(28000)
            backend = ctx.Process(target=agent.serve_backend, args=([s.response_body for s in specs], backend_port))
            processes.append(backend)
            backend.start()
            wait_for_server_ready("127.0.0.1", backend_port, backend, timeout=90)
            agent_port = find_available_port(29000)
            agent_process = ctx.Process(target=agent.serve_agent, args=([s.request_body for s in specs], agent_port))
            processes.append(agent_process)
            agent_process.start()
            wait_for_server_ready("127.0.0.1", agent_port, agent_process, timeout=90)
            os.environ["MILES_BENCH_AGENT_URL"] = f"http://127.0.0.1:{agent_port}"
            args = build_args(bench, prompt_path, backend_port)
            http_utils.init_http_client(args)
            server_port = find_available_port(33000)
            config = compute_session_server_config(
                args,
                host="127.0.0.1",
                port=server_port,
                instance_id="bench-0",
                backend_url=f"http://127.0.0.1:{backend_port}",
            )
            server = ctx.Process(target=run_session_server, args=(config,))
            processes.append(server)
            server.start()
            wait_for_server_ready("127.0.0.1", server_port, server, timeout=90)
            args.session_server_instances = [
                SessionServerInstance(addr=f"127.0.0.1:{server_port}", instance_id="bench-0")
            ]
            rollout = load_rollout_function(
                RolloutFnConstructorInput(args=args, data_source=RolloutDataSourceWithBuffer(args)),
                args.rollout_function_path,
            )
            manager = psutil.Process()
            server_process = psutil.Process(server.pid)
            for step in range(bench.repetitions + 1):
                gc.collect()
                peak_rss = [manager.memory_info().rss]
                stop = threading.Event()

                def sample_rss():
                    while not stop.wait(0.02):
                        peak_rss[0] = max(peak_rss[0], manager.memory_info().rss)

                sampler = threading.Thread(target=sample_rss, daemon=True)
                sampler.start()
                cpu0 = time.process_time()
                server_cpu0 = server_process.cpu_times()
                start = time.perf_counter()
                try:
                    output = call_rollout_function(rollout, RolloutFnTrainInput(rollout_id=step))
                    wall = time.perf_counter() - start
                    cpu = time.process_time() - cpu0
                    server_cpu1 = server_process.cpu_times()
                    peak_rss[0] = max(peak_rss[0], manager.memory_info().rss)
                finally:
                    stop.set()
                    sampler.join()
                assert len(output.samples) == bench.sessions
                assert all(len(group) == 1 and len(group[0]) == 1 for group in output.samples)
                samples = [sample for group in output.samples for trajectory in group for sample in trajectory]
                r3_bytes = validate_samples(samples, specs, bench.sessions)
                measurement = {
                    "warmup": step == 0,
                    "wall_s": wall,
                    "trajectories_per_s": len(samples) / wall,
                    "manager_cpu_s": cpu,
                    "manager_peak_rss_bytes": peak_rss[0],
                    "server_cpu_s": server_cpu1.user + server_cpu1.system - server_cpu0.user - server_cpu0.system,
                    "samples": len(samples),
                    "turns": len(samples) * len(specs),
                    "r3_bytes": r3_bytes,
                }
                print(json.dumps(measurement), flush=True)
                results.append(measurement)
                del output, samples
    finally:
        try:
            run(agent.close_client())
            if http_utils._http_client is not None:
                run(http_utils._http_client.aclose())
                http_utils._http_client = None
        finally:
            for process in reversed(processes):
                _terminate_proc(process)
    measured = results[1:]
    return {
        "config": vars(bench),
        "rollout_function": args.rollout_function_path,
        "steps": results,
        "median_trajectories_per_s": statistics.median(r["trajectories_per_s"] for r in measured),
        "median_manager_cpu_s": statistics.median(r["manager_cpu_s"] for r in measured),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sessions", type=_positive_int, default=8)
    parser.add_argument("--turns", type=_positive_int, default=8)
    parser.add_argument("--input-tokens", type=_positive_int, default=256)
    parser.add_argument("--output-tokens", type=_positive_int, default=256)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--json-out", required=True)
    bench = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    result = run_benchmark(bench)
    Path(bench.json_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
