import os

import miles.utils.external_utils.command_utils as U

NUM_GPUS = 8
MODEL_TYPE = "qwen3.5"


def execute():
    ckpt_args = (\n        "--hf-checkpoint ${model_dir}/${model_name} "
"--ref-load ${model_dir}/${model_name}_torch_dist "
"--load ${ckpt_dir}/${variant} "
"--save ${ckpt_dir}/${variant} "
"--save-interval ${interval} "\n    )\n\n    rollout_args = (\n        "--prompt-data ${data_path} "
"--input-key prompt "
"--label-key label "
"--apply-chat-template "
"--apply-chat-template-kwargs {"enable_thinking":false} "
"--rollout-shuffle "
"--rm-type dapo "
"--reward-key score "
"--num-rollout 3000 "
"--rollout-batch-size 32 "
"--n-samples-per-prompt 8 "
"--rollout-max-response-len 8192 "
"--rollout-temperature 1 "
"--global-batch-size 256 "
"--balance-data "\n    )\n\n    eval_args = (\n        "--eval-interval ${interval} "
"--n-samples-per-eval-prompt 4 "
"--eval-max-response-len 8192 "
"--eval-top-p 1 "\n    )\n\n    perf_args = (\n        "--tensor-model-parallel-size 2 "
"--sequence-parallel "
"--pipeline-model-parallel-size 4 "
"--context-parallel-size 2 "
"--expert-model-parallel-size 8 "
"--expert-tensor-parallel-size 1 "
"--recompute-granularity full "
"--recompute-method uniform "
"--recompute-num-layers 1 "
"--use-dynamic-batch-size "
"--max-tokens-per-gpu 8192 "\n    )\n\n    grpo_args = (\n        "--advantage-estimator gspo "
"--kl-loss-coef 0.00 "
"--kl-loss-type low_var_kl "
"--kl-coef 0.00 "
"--entropy-coef 0.00 "
"--eps-clip 4e-4 "\n    )\n\n    optimizer_args = (\n        "--optimizer adam "
"--lr 1e-6 "
"--lr-decay-style constant "
"--weight-decay 0.1 "
"--adam-beta1 0.9 "
"--adam-beta2 0.98 "
"--optimizer-cpu-offload "
"--overlap-cpu-optimizer-d2h-h2d "
"--use-precision-aware-optimizer "\n    )\n\n    wandb_args = (\n        "--use-wandb "
"--wandb-project ${project_name} "
"--wandb-group ${variant} "
"--wandb-key ${WANDB_KEY} "\n    )\n\n    tb_args = (\n        "--use-tensorboard "
"--tb-dir /home/yangchengyi/data/tb "
"--tb-project-name ${project_name} "
"--tb-experiment-name ${variant} "\n    )\n\n    sglang_args = (\n        "--rollout-num-gpus-per-engine 8 "
"--sglang-mem-fraction-static 0.62 "
"--sglang-ep-size 8 "
"--sglang-cuda-graph-bs 1 2 4 8 16 32 48 64 "
"--sglang-max-running-requests 256 "
"--sglang-chunked-prefill-size 8192 "
"--sglang-server-concurrency 256 "\n    )\n\n    misc_args = (\n        "--attention-dropout 0.0 "
"--hidden-dropout 0.0 "
"--accumulate-allreduce-grads-in-fp32 "
"--attention-softmax-in-fp32 "
"--attention-backend flash "
"--moe-token-dispatcher-type flex "
"--moe-enable-deepep "\n    )

    train_args = (
        f"{ckpt_args} "\n        f"{rollout_args} "\n        f"{optimizer_args} "\n        f"{grpo_args} "\n        f"{wandb_args} "\n        f"{tb_args} "\n        f"{perf_args} "\n        f"{eval_args} "\n        f"{sglang_args} "\n        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
