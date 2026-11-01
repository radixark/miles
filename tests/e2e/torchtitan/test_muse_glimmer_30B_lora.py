from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500, suite="stage-c-4-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=4
)

# Muse Glimmer 30B (dense, 52 layers: 39 sliding-window and 13 global NoPE) with rank-32 adapters
# on every decoder projection. Only the adapters train; each sync streams them to sglang as a PEFT
# adapter. Two trainer GPUs (TP=2) and two one-GPU engines: the fp32 base is 120 GB, which a
# colocated layout cannot fit next to the engines once the CI weight checker makes every sync
# re-send the base. The disaggregated sync streams the adapters only and has no single-rank
# adapter gather, so the CI weight checker (which randomizes the engine's weights and expects
# the sync to rewrite them all) and --check-lora-weight-equal do not apply; the log-prob checker
# still compares trainer and engine on every rollout. The trainer builds the text decoder only,
# so the engine skips the vision tower too: its layers carry the same projection names and would
# otherwise be adapted (sglang indexes LoRA modules by the text layer count).
CASE = CaseConfig(
    model_repo="meta-models/Muse-Glimmer-30B",
    titan_model_name="muse_glimmer",
    titan_model_flavor="30B",
    num_gpus=2,
    tp_size=2,
    seq_len=4096,
    max_response_len=2048,
    colocate=False,
    rollout_num_gpus=2,
    rollout_num_gpus_per_engine=1,
    num_rollout=4,
    extra_args=(
        "--lora-rank 32 --lora-alpha 32 --target-modules all-linear --lr 1e-5 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "
        "--sglang-language-model-only --ci-disable-weight-update-checker --rm-type math "
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
