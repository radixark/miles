from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=2400, suite="stage-c-4-gpu-h200", labels=["torchtitan", "replay"], hardware=["hopper"])

# The first model torchtitan itself does not ship: the flavor lives under
# miles/backends/torchtitan_utils/models/ and reuses torchtitan's DeepSeek-V3
# blocks (MLA attention, sigmoid router with the correction bias, shared
# expert), so this case covers the miles-side registry end to end -- HF load
# through the DeepSeek adapter, expert parallelism, routing replay, and the
# weight stream back into SGLang's own glm4_moe_lite implementation. SGLang runs
# attention on triton: flashinfer's SM100 prefill kernel has no head_dim 256
# instantiation, which is what MLA's 192 + 64 query head pads to.
CASE = CaseConfig(
    model_repo="zai-org/GLM-4.7-Flash",
    titan_model_name="glm4_moe_lite",
    titan_model_flavor="30B-A3B",
    num_gpus=4,
    ep_size=4,
    seq_len=16384,
    max_response_len=8192,
    use_r3=True,
    mem_fraction_static=0.5,
    extra_args="--sglang-attention-backend triton --sglang-moe-runner-backend triton ",
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
