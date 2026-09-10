from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1500, suite="stage-c-8-gpu-h200", labels=["torchtitan", "weight-update"])

# disk-delta is the only transfer that reconciles the weight stream against the
# checkpoint's own bytes, which makes it the one case that can catch a stream
# that is subtly wrong rather than merely slow. It has already caught three:
# fp32 master weights where the checkpoint holds bf16, a multimodal config whose
# dtype lives in text_config so the cast never applied, and a checkpoint that
# mixes dtypes per tensor -- qwen3.5 keeps its linear-attention log scales in
# fp32 beside bf16 weights. It also exercises a protocol that declines its first
# round, since that one only captures the baseline the next delta diffs against.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3.5-4B",
    titan_model_name="qwen3_5",
    titan_model_flavor="4B",
    num_gpus=4,
    seq_len=8192,
    max_response_len=4096,
    colocate=False,
    rollout_num_gpus=4,
    transfer_mode="disk-delta",
    num_rollout=3,
    mem_fraction_static=0.6,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
