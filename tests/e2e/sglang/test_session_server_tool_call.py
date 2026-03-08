"""E2E test: session-server pretokenized TITO with real model inference.

Starts the full miles pipeline (sglang + miles-router with session support)
via ``execute_train --debug-rollout-only``, then runs the agentic_tool_call
generate function with a custom agent that performs multi-turn tool calls and
asserts the pretokenized prefix invariant on every turn.

Requires 1 GPU.
"""

import json
import os

import pytest

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
PROMPT_DATA_PATH = "/root/datasets/session_tool_call.jsonl"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} " f"--local-dir /root/models/{MODEL_NAME}")

    user_content = (
        "I need the weather for four cities. You MUST call get_weather exactly "
        "4 times, one city per turn, in this order:\n"
        "  Turn 1: Beijing\n"
        "  Turn 2: Shanghai\n"
        "  Turn 3: Tokyo\n"
        "  Turn 4: New York\n"
        "After all 4 tool results come back, summarize everything in one message."
    )

    prompts = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to tools. "
                        "You MUST use the provided tools. Each turn you call "
                        "exactly one tool. Do NOT answer without using a tool "
                        "until you have called it 4 times."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to tools. "
                        "Always call the provided tool once per turn. Do NOT "
                        "give a final answer until you have made all 4 tool "
                        "calls."
                    ),
                },
                {"role": "user", "content": user_content},
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
        "--num-rollout 2 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0 "
        "--global-batch-size 2 "
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

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=1,
        megatron_model_type=None,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


@pytest.mark.system
def test_session_server_tool_call():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
