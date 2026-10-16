"""True-on-policy rollout top-k candidates match the trainer, on the session-server chat path.

A single-turn chat agent sends each prompt through the session server, which requests the candidates as
SGLang's flat base64 arrays (``return_flat_raw_output_top_logprobs``); the check is in
``_true_on_policy_rollout_topk``.
"""

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.fsdp._true_on_policy_rollout_topk import main

register_cuda_ci(
    est_time=900, suite="stage-c-2-gpu-h200", labels=["fsdp", "sglang", "replay"], hardware=["hopper"], num_gpus=2
)
# --ci-test already asserts both are exactly 0 on every rollout.
register_ci_gate(
    metric_key="rollout/topk_logprob_abs_diff", steps="all", constraint={"abs_floor_up": 0.0, "abs_floor_down": 0.0}
)
register_ci_gate(
    metric_key="rollout/topk_set_mismatch", steps="all", constraint={"abs_floor_up": 0.0, "abs_floor_down": 0.0}
)
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")

SESSION_ARGS = (
    "--use-session-server "
    "--session-server-workers 1 "
    "--tito-model qwen3 "
    "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
    "--custom-agent-function-path miles.utils.test_utils.single_turn_chat_agent.run_agent "
)

if __name__ == "__main__":
    main(__file__, rollout_path_args=SESSION_ARGS)
