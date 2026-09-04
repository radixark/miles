from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h200", labels=["torchtitan", "weight-update"])

# Pushing weights out of a pipelined trainer, which is where the stream stops
# being something every rank can complete on its own. Each stage holds only its
# own layers, so the export is rank-partial twice over -- expert parallelism
# splits a stage's experts as well -- and the transfer protocol chooses how much
# of that gets reassembled before it ships. Completing the whole model on every
# rank is what a pipelined 30B could not afford, and skipping the completion
# inside a stage is what silently corrupted expert weights, so this case exists
# to keep both halves honest.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3.5-4B",
    titan_model_name="qwen3_5",
    titan_model_flavor="4B",
    num_gpus=4,
    pp_size=2,
    seq_len=8192,
    max_response_len=4096,
    colocate=False,
    rollout_num_gpus=4,
    transfer_mode="broadcast",
    num_rollout=2,
    mem_fraction_static=0.6,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
