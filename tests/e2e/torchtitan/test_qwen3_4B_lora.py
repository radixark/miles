from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=700, suite="stage-c-4-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=4
)

# The megatron Qwen3-4B LoRA recipe (examples/lora/run-qwen3-4B-megatron-lora.sh) on
# torchtitan: TP=2 with one engine per GPU, rank-32 all-linear adapters and dynamic
# batching. Only the adapters train; each sync streams them to sglang as a PEFT
# adapter, and --check-lora-weight-equal checksums every streamed tensor.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-4B",
    titan_model_name="qwen3",
    titan_model_flavor="4B",
    num_gpus=4,
    tp_size=2,
    seq_len=4096,
    max_response_len=2048,
    rollout_num_gpus_per_engine=1,
    num_rollout=4,
    # thinking fills the response cap and scores 0, leaving the adapter no gradient
    extra_args=(
        "--lora-rank 32 --lora-alpha 32 --target-modules all-linear --lr 1e-5 "
        "--check-lora-weight-equal --use-dynamic-batch-size --max-tokens-per-gpu 9216 "
        """--apply-chat-template-kwargs '{"enable_thinking": false}' --rm-type math """
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
