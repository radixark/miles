"""Through the session server, the R3 the trainer consumes is the same through the output store as inline.

Two two-rollout runs differ only in --use-output-store: Qwen3-30B-A3B on four colocated GPUs (TP 2 x CP 2, EP 4)
with session_verify_agent driving each multi-turn session, whose schedule appends user, system, tool and assistant
turns and rolls back. Deterministic inference with fixed per-sample seeds and one running request at a time makes
both runs hold the same sessions, and --lr 0 keeps the weights, so every rollout does. The rolled-back picker
leaves one sample per session. --save-debug-train-data keeps rollout_routed_experts as routing replay consumed it on
each rank; every sample must come back with the same tokens and a byte-identical R3 on each CP rank.
"""

import os
import tempfile
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.output_store_r3 import assert_same_trainer_r3, save_debug_train_data_arg

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.session_verify_runner import PROMPT_DATA_PATH, _ensure_prompt_data

register_cuda_ci(
    est_time=1200, suite="stage-c-4-gpu-h200", labels=["mooncake", "replay"], hardware=["hopper"], num_gpus=4
)

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 4
CP_SIZE = 2
NUM_ROLLOUT = 2
SESSIONS_PER_ROLLOUT = 2
DUMP_DIR = Path(tempfile.mkdtemp(prefix="session_output_store_r3_"))


def prepare() -> None:
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    # The agent synthesizes its own conversation; the rollout still needs a prompt file.
    _ensure_prompt_data()


def execute(mode: str) -> None:
    U = command_utils.default_config().create_backend()
    train_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "
        "--megatron-to-hf-mode bridge "
        f"--prompt-data {PROMPT_DATA_PATH} --input-key messages --rm-type random "
        f"--num-rollout {NUM_ROLLOUT} --rollout-batch-size 1 --n-samples-per-prompt {SESSIONS_PER_ROLLOUT} "
        f"--global-batch-size {SESSIONS_PER_ROLLOUT} "
        "--rollout-max-response-len 1024 --rollout-temperature 0.7 "
        # Short turns, as the session server's other Qwen3-30B-A3B e2e runs them.
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        # Same sessions in both runs: batch-invariant kernels, a seed per sample, one request at a time.
        "--sglang-enable-deterministic-inference --rollout-seed 42 --sglang-max-running-requests 1 "
        "--use-session-server v2 "
        "--session-sample-picker-path miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves "
        "--custom-generate-function-path miles.utils.test_utils.session_verify_agent.generate "
        "--custom-agent-function-path miles.utils.test_utils.session_verify_agent.run_agent "
        "--tito-model qwen3 --sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen25 "
        "--session-verify-cycles 2 --tool-call-failure-mode append_tool "
        "--use-rollout-routing-replay "
        f"{save_debug_train_data_arg(DUMP_DIR, mode)}"
        f"{command_utils.get_mooncake_object_store_args()} "
        f"{'--use-output-store ' if mode == 'store' else ''}"
        "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        # Unchanged weights: every rollout of both runs generates from the same model.
        "--optimizer adam --lr 0 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
        f"--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size {CP_SIZE} --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
        "--rematerialize-param-from-master-weight "
        f"--rollout-num-gpus-per-engine {NUM_GPUS} --sglang-mem-fraction-static 0.7 "
        f"{command_utils.get_default_wandb_args(__file__)} "
    )
    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute("inline")
    execute("store")
    assert_same_trainer_r3(DUMP_DIR, cp_size=CP_SIZE, num_samples=NUM_ROLLOUT * SESSIONS_PER_ROLLOUT)
