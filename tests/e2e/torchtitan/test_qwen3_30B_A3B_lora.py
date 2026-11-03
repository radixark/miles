from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500,
    suite="stage-c-8-gpu-h200",
    labels=["torchtitan", "lora", "replay"],
    hardware=["hopper"],
    num_gpus=6,
    disabled="needs the image's torchtitan to carry the shared-outer expert adapters (https://github.com/pytorch/torchtitan/pull/5738)",
)

# Qwen3-30B-A3B with rank-32 adapters on attention and on the routed experts in the shared-outer layout
# (--experts-shared-outer-loras): w13 shares lora_A across the 128 experts, w2 shares lora_B. Four trainer
# GPUs (TP=2, EP=2) and two one-GPU engines serving the expert adapters through sglang's fused-MoE LoRA path;
# the disaggregated sync streams adapters only, so the CI weight checker does not apply. Routing replay keeps
# the trainer on the engine's expert choices.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-30B-A3B",
    titan_model_name="qwen3",
    titan_model_flavor="30B-A3B",
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
    mem_fraction_static=0.7,
    extra_args=(
        "--lora-rank 32 --lora-alpha 32 --target-modules all-linear --experts-shared-outer-loras --lr 1e-5 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        "--sglang-lora-backend triton --sglang-max-lora-rank 32 --sglang-moe-runner-backend triton "
        "--ci-disable-weight-update-checker "
        """--apply-chat-template-kwargs '{"enable_thinking": false}' --rm-type math """
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
