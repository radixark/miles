from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["torchtitan", "fully-async", "replay"])

# Routing replay while generation and training run at once. The rollout that
# recorded the routing is a weight version or more behind the trainer replaying
# it, which is the one thing no synchronous run can check: everything else about
# R3 was validated with the rollout and the trainer in lockstep.
#
# It also puts the disaggregated weight path under load rather than merely
# reaching it -- expert parallelism makes each rank's export partial, so
# completing the stream takes an owner broadcast, and fully_async repeats the
# whole exchange continuously instead of once per rollout.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-30B-A3B",
    titan_model_name="qwen3",
    titan_model_flavor="30B-A3B",
    num_gpus=4,
    ep_size=4,
    seq_len=8192,
    max_response_len=4096,
    use_r3=True,
    colocate=False,
    rollout_num_gpus=4,
    fully_async=True,
    num_rollout=3,
    mem_fraction_static=0.8,
    extra_args="--sglang-moe-runner-backend triton ",
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
