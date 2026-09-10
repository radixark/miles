from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=700, suite="stage-c-2-gpu-h200", labels=["torchtitan"])

# Context parallelism stays inside the trainer: attention sees torchtitan's
# shorter per-rank sequences while the loss is handed the full one, gathered
# back through the inverse of torchtitan's own load-balancing permutation. The
# gather has to carry gradients and the sequence has to be a multiple of
# cp * 128 for flex attention -- neither shows up as a wrong number, only as a
# crash or a silently mispermuted loss.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-0.6B",
    titan_model_name="qwen3",
    titan_model_flavor="0.6B",
    num_gpus=2,
    cp_size=2,
    seq_len=4096,
    max_response_len=2048,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
