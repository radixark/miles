from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=2700, suite="stage-c-8-gpu-h200", labels=["torchtitan", "replay"], hardware=["hopper"])

# Every axis at once on one MoE model: tensor, pipeline, context and expert
# parallelism with routing replay, and the FSDP shard degree left at one. Each
# axis is covered on its own elsewhere; this case exists because their
# interactions are where the bookkeeping goes wrong -- the CP logits gather
# under a pipeline schedule, expert weights that are rank-partial on two axes at
# export time, replay queues consumed per microbatch across stages. The
# train/rollout log-prob gap and the reward curve are what to read.
#
# The sequence is 8k rather than the 16k the single-axis cases use: with four
# axes on eight GPUs the FSDP shard degree is one, so the optimizer state is
# whole on every rank, and the context-parallel loss gathers the full
# sequence's logits before the cross entropy. At 16k that gather has no
# headroom left on an H200.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-30B-A3B",
    titan_model_name="qwen3",
    titan_model_flavor="30B-A3B",
    num_gpus=8,
    tp_size=2,
    pp_size=2,
    cp_size=2,
    ep_size=2,
    seq_len=8192,
    max_response_len=4096,
    use_r3=True,
    mem_fraction_static=0.55,
    num_rollout=4,
    extra_args="--sglang-moe-runner-backend triton ",
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
