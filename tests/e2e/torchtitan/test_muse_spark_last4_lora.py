from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["torchtitan", "lora", "replay"],
    hardware=["hopper"],
    num_gpus=6,
    disabled="needs a CI image whose sglang serves Muse Spark Open (#4601)",
)

# RadixArk/Muse-Spark-Open-last4-partial: layers 58-61 of Muse Spark Open (three sliding-window
# layers and one global NoPE layer, 256 compressed experts top-8), 108 GiB. Rank-32 adapters on
# attention q/k/v/o and on the routed experts in the shared-outer layout. sglang serves no LoRA for
# the compressed experts, so every sync sends the base merged with the adapter and the engines run
# LoRA-free (--lora-merge-for-rollout). Four trainer GPUs (TP=2 with EP=2, since torchtitan needs
# EP >= TP for MoE; the fp32 base is 232 GB) and two one-GPU engines serving the omni checkpoint as
# released (the towers cost 3.8 GiB; sglang's language-model-only mode does not cover this
# architecture), at a memory fraction that fits the 108 GiB of weights plus the receive buffer of
# the largest merged tensor, a layer's 16 GiB packed gate_up_proj; the CI weight checker's snapshot
# of those weights does not fit, and the activation headroom CUDA-graph capture reserves would
# starve the sliding-window KV pool. A four-layer slice never solves a math problem, so the
# tie-free deterministic_random reward keeps the GRPO advantages nonzero; routing replay keeps the
# trainer on the engine's expert choices.
CASE = CaseConfig(
    model_repo="RadixArk/Muse-Spark-Open-last4-partial",
    titan_model_name="muse_spark",
    titan_model_flavor="816B-A42B",
    num_gpus=4,
    tp_size=2,
    ep_size=2,
    seq_len=4096,
    max_response_len=1024,
    colocate=False,
    rollout_num_gpus=2,
    rollout_num_gpus_per_engine=1,
    use_r3=True,
    num_rollout=2,
    mem_fraction_static=0.8,
    extra_args=(
        "--lora-rank 32 --lora-alpha 32 --target-modules all-linear --experts-shared-outer-loras "
        "--lora-merge-for-rollout --lr 1e-5 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        "--sglang-kv-cache-dtype bfloat16 --sglang-chunked-prefill-size 1024 --sglang-disable-cuda-graph "
        "--ci-disable-weight-update-checker --ci-disable-logprobs-checker --ci-disable-kl-checker "
        "--rm-type deterministic_random "
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
