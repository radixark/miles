from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=600, suite="stage-c-2-gpu-h200", labels=["torchtitan"])

# The baseline: one model, no parallelism beyond data, colocated engines. What
# it covers is the path itself -- torchtitan's Trainer built from miles' args,
# its checkpointer loading the HF weights, and its parameters streamed back to
# sglang under HF names through the model's own state-dict adapter.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-0.6B",
    titan_model_name="qwen3",
    titan_model_flavor="0.6B",
    num_gpus=2,
    seq_len=2048,
    max_response_len=1024,
    num_rollout=8,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
