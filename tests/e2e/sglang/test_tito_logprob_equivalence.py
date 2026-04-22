"""E2E test: verify TITO session logprobs match a fresh full re-prefill.

After a multi-turn agent conversation through the session server, the
full ``accumulated_token_ids`` (the token sequence built incrementally
by TITO) is sent to ``/generate`` with ``max_new_tokens=0`` for a single
prefill pass.  The resulting ``input_token_logprobs`` are compared
per-turn against the ``output_token_logprobs`` from the session's decode
phase.  Token IDs must match exactly; logprob values are compared with a
tight tolerance for prefill-vs-decode numerical differences.

When an MoE model is used with ``ENABLE_R3=1``, routed_experts arrays
are also compared.

Uses the same infrastructure as ``test_session_server_tool_call``:
``execute_train --debug-rollout-only`` with a custom generate function
that wraps the normal agentic flow with re-prefill verification.

Requires 1 GPU.
"""

import json
import os
from dataclasses import dataclass
import miles.utils.external_utils.command_utils as U

# ---------------------------------------------------------------------------
# Model family registry
# ---------------------------------------------------------------------------

MODEL_FAMILY = os.environ.get("SESSION_TEST_MODEL_FAMILY", "qwen3")
ENABLE_R3 = os.environ.get("ENABLE_R3", "0") == "1"


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

PROMPT_DATA_PATH = "/root/datasets/session_logprob_verify.jsonl"


def _get_config() -> ModelConfig:
    if MODEL_FAMILY not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family {MODEL_FAMILY!r}. " f"Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[MODEL_FAMILY]


def prepare():
    cfg = _get_config()
    U.exec_command("mkdir -p /root/models /root/datasets")
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
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.0 "
        "--global-batch-size 16 "
    )

    generate_args = (
        "--custom-generate-function-path "
        "tests.e2e.sglang.utils.logprob_verify_generate.generate "
        "--custom-agent-function-path "
        "tests.e2e.sglang.utils.session_tool_agent.run_agent "
    )

    router_args = (
        "--use-miles-router " "--use-session-server " "--chat-template-path autofix " f"--tito-model {cfg.tito_model} "
    )
    if ENABLE_R3:
        router_args += "--use-rollout-routing-replay "

    sglang_args = f"--rollout-num-gpus-per-engine {cfg.num_gpus} " f"--sglang-reasoning-parser {cfg.reasoning_parser} "
    if cfg.tool_call_parser:
        sglang_args += f"--sglang-tool-call-parser {cfg.tool_call_parser} "
    sglang_args += "--rm-type random " "--sglang-enable-deterministic-inference "

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


def test_tito_logprob_equivalence():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
