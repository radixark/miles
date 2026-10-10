from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["torchtitan"],
    hardware=["hopper"],
    disabled="needs a CI image with GLM-5.3-Flash sglang support (branch sglang-miles-glm53); "
    "re-enable once it lands (#3989).",
)

# four layers cover both attention kinds and both MLP kinds; glm5_next supports FSDP + EP only
CASE = CaseConfig(
    model_repo="CharyZeng/GLM-5.3-Flash-4layer",
    titan_model_name="glm5_next",
    titan_model_flavor="4layer",
    num_gpus=8,
    ep_size=8,
    rollout_num_gpus_per_engine=4,
    seq_len=4096,
    max_response_len=512,
    mem_fraction_static=0.5,
    extra_args=(
        "--sglang-ep-size 4 "
        "--sglang-moe-runner-backend triton "
        "--sglang-disable-radix-cache "
        "--sglang-dsa-prefill-backend tilelang "
        "--sglang-dsa-decode-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
        "--model-name glm5_next "
        "--ci-disable-logprobs-checker "
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
