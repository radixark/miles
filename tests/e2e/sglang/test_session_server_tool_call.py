"""E2E test: session-server pretokenized TITO with real model inference.

Starts the full miles pipeline (sglang + miles-router with session support)
via ``execute_train --debug-rollout-only``, then runs the agentic_tool_call
generate function with a custom agent that performs multi-turn tool calls and
asserts the pretokenized prefix invariant on every turn.

Requires 1 GPU.
"""

import json
import os
from dataclasses import dataclass

import pytest

import miles.utils.external_utils.command_utils as U

# ---------------------------------------------------------------------------
# Model family registry
# ---------------------------------------------------------------------------

MODEL_FAMILY = os.environ.get("SESSION_TEST_MODEL_FAMILY", "qwen3")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None = None
    tito_model: str = "default"
    num_gpus: int = 1


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3": ModelConfig(
        model_name="Qwen/Qwen3-4B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    ),
    "glm47": ModelConfig(
        model_name="zai-org/GLM-4.7-Flash",
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        tito_model="glm47",
        num_gpus=1,
    ),
}

PROMPT_DATA_PATH = "/root/datasets/session_tool_call.jsonl"


def _get_config() -> ModelConfig:
    if MODEL_FAMILY not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family {MODEL_FAMILY!r}. " f"Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[MODEL_FAMILY]


def prepare():
    cfg = _get_config()
    U.exec_command("mkdir -p /root/models /root/datasets")
    if MODEL_FAMILY == "glm47":
        U.exec_command(
            "pip install git+https://github.com/huggingface/transformers.git@"
            "76732b4e7120808ff989edbd16401f61fa6a0afa --break-system-packages"
        )
    U.exec_command(f"hf download {cfg.model_name} --local-dir /root/models/{cfg.model_name.split('/')[-1]}")

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
                    "content": "What's the weather like in Beijing, Shanghai, Tokyo, and New York?",
                },
            ],
        },
    ]
    with open(PROMPT_DATA_PATH, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def execute():
    cfg = _get_config()
    local_model_dir = f"/root/models/{cfg.model_name.split('/')[-1]}"

    ckpt_args = f"--hf-checkpoint {local_model_dir} "

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
        "tests.e2e.sglang.utils.session_tool_agent.run_agent "
    )

    router_args = (
        "--use-miles-router " "--use-session-server " "--chat-template-path autofix " f"--tito-model {cfg.tito_model} "
    )

    sglang_args = f"--rollout-num-gpus-per-engine {cfg.num_gpus} " f"--sglang-reasoning-parser {cfg.reasoning_parser} "
    if cfg.tool_call_parser:
        sglang_args += f"--sglang-tool-call-parser {cfg.tool_call_parser} "
    sglang_args += "--rm-type random "

    infra_args = (
        "--debug-rollout-only "
        "--ci-test "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {cfg.num_gpus} "
        "--colocate "
        "--train-backend fsdp "
    )

    train_args = f"{ckpt_args}" f"{rollout_args}" f"{generate_args}" f"{router_args}" f"{sglang_args}" f"{infra_args}"

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=cfg.num_gpus,
        megatron_model_type=None,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "SGLANG_E2E_MODEL_PATH": local_model_dir,
            "MILES_TITO_MODEL": cfg.tito_model,
        },
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
    if MODEL_FAMILY == "glm47":
        U.exec_command("pip install transformers==4.57.1 --break-system-packages")
