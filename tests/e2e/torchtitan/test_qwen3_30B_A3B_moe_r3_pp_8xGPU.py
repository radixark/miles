from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["torchtitan", "replay"])

# Expert parallelism, pipeline parallelism and rollout routing replay at once,
# which is the combination that hides mistakes. The weight stream is
# rank-partial under both axes, so completing it takes an owner broadcast rather
# than each rank exporting what it holds. The replay queues are read one entry
# per microbatch, and the pipeline schedule's shape-inference forward would
# otherwise take the first entry for itself and leave every microbatch replaying
# its predecessor's routing -- visible only as a slow divergence between
# training and rollout, since neighbouring samples share a prompt.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-30B-A3B",
    titan_model_name="qwen3",
    titan_model_flavor="30B-A3B",
    num_gpus=8,
    pp_size=2,
    ep_size=4,
    seq_len=16384,
    max_response_len=8192,
    use_r3=True,
    mem_fraction_static=0.55,
    extra_args="--sglang-moe-runner-backend triton ",
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
