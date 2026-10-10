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

# two layers per stage keep a KDA and a DSA layer on each side of the pipeline cut
CASE = CaseConfig(
    model_repo="CharyZeng/GLM-5.3-Flash-4layer",
    titan_model_name="glm5_next",
    titan_model_flavor="4layer",
    num_gpus=8,
    pp_size=2,
    cp_size=2,
    ep_size=4,
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
