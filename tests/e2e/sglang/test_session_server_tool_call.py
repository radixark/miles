"""E2E test: session-server pretokenized TITO with real model inference.

Starts the full miles pipeline (sglang + miles-router with session support)
via ``execute_train --debug-rollout-only``, then runs the agentic_tool_call
generate function with a custom agent that performs multi-turn tool calls and
asserts the pretokenized prefix invariant on every turn.

Requires 1 GPU.
"""

import json
import os
from pathlib import Path

import pytest

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
PROMPT_DATA_PATH = "/root/datasets/session_tool_call.jsonl"
TITO_STATS_PATH = Path("/tmp/tito_stats.json")
TITO_PASS_RATE_THRESHOLD = 0.95


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} " f"--local-dir /root/models/{MODEL_NAME}")

    prompts = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to weather tools. "
                        "Use the get_weather tool to look up weather information. "
                        "When you have gathered all the information, "
                        "wrap your final summary in <final_answer>...</final_answer> tags."
                    ),
                },
                {
                    "role": "user",
                    "content": ("What's the weather like in Beijing, Shanghai, Tokyo, and New York?"),
                },
            ],
        },
    ]
    with open(PROMPT_DATA_PATH, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        f"--prompt-data {PROMPT_DATA_PATH} "
        "--input-key messages "
        "--num-rollout 1 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.7 "
        "--global-batch-size 64 "
    )

    generate_args = (
        "--custom-generate-function-path "
        "miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path "
        "tests.e2e.sglang.session_tool_agent.run_agent "
    )

    router_args = "--use-miles-router " "--chat-template-path autofix "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen "
        "--rm-type random "
    )

    infra_args = (
        "--debug-rollout-only "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--colocate "
        "--train-backend fsdp "
    )

    train_args = f"{ckpt_args}" f"{rollout_args}" f"{generate_args}" f"{router_args}" f"{sglang_args}" f"{infra_args}"

    if TITO_STATS_PATH.exists():
        TITO_STATS_PATH.unlink()

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=1,
        megatron_model_type=None,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


def check_tito_pass_rate():
    assert TITO_STATS_PATH.exists(), f"TITO stats file not found at {TITO_STATS_PATH}"
    stats = json.loads(TITO_STATS_PATH.read_text())
    rate = stats["pass_rate"]
    total = stats["total"]
    matched = stats["matched"]
    mismatch = stats["mismatch"]
    print(f"TITO stats: {matched}/{total} matched, " f"{mismatch} mismatch, pass_rate={rate:.1%}")
    assert rate >= TITO_PASS_RATE_THRESHOLD, (
        f"TITO pass rate {rate:.1%} ({matched}/{total}) " f"is below threshold {TITO_PASS_RATE_THRESHOLD:.0%}"
    )


@pytest.mark.system
def test_session_server_tool_call():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
    check_tito_pass_rate()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
    check_tito_pass_rate()
