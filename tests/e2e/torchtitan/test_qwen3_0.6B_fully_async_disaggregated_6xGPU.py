from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h200", labels=["torchtitan", "fully-async"])

# Rollout on its own GPUs, generating while training runs. That rules out the
# colocated IPC weight transfer -- the engines are not on these devices -- so
# this is the case that covers rank 0 broadcasting each bucket over NCCL while
# every rank still walks the whole weight stream (producing a tensor takes
# collectives, so a rank that skipped ahead would hang the others). Three
# rollouts to reach a drain across a weight update.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-0.6B",
    titan_model_name="qwen3",
    titan_model_flavor="0.6B",
    num_gpus=4,
    seq_len=4096,
    max_response_len=2048,
    colocate=False,
    rollout_num_gpus=2,
    fully_async=True,
    num_rollout=3,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
