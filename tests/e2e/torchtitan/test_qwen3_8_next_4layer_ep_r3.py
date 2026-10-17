from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["torchtitan"], hardware=["hopper"], num_gpus=8)

# The 4-layer slice keeps one of every Qwen3.8-Next layer kind -- three GatedDeltaNet
# layers, one QSA layer, the PLE layer -- with all 512 experts, so one node exercises
# the hyper-connection, sparse-attention and n-gram paths the full model runs.
# Experts are split over all eight trainer ranks with routing replay on, and the
# weight check compares every streamed tensor against the engine's except the
# frozen PLE table and the vision towers, which are never trained.
CASE = CaseConfig(
    model_repo="CharyZeng/Qwen3.8-Flash-Next-4layer",
    titan_model_name="qwen3_8_next",
    titan_model_flavor="Flash-Next-4layer",
    num_gpus=8,
    seq_len=4096,
    max_response_len=2048,
    ep_size=8,
    use_r3=True,
    rollout_num_gpus_per_engine=4,
    extra_args=(
        "--model-name qwen4_exp "
        "--sglang-tp-size 4 --sglang-ep-size 4 "
        "--sglang-linear-attn-prefill-backend flashinfer "
        "--sglang-moe-runner-backend triton "
        "--sglang-disable-radix-cache "
        "--check-weight-update-equal --check-weight-update-skip-list visual. ple_embedding. "
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
