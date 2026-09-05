import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2
SGLANG_ENGINE_IP = "127.0.0.1"
SGLANG_ENGINE_PORT = 32000


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 250} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-mem-fraction-static 0.6 "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.55 "  # loose threshold at 250 step
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node 2 "
        # External engines run on their own hosts, so the trainer stays disaggregated
        # (no --colocate) and miles starts and owns its own router.
        "--rollout-external "
        # TODO test multi-engine
        f"--rollout-external-engine-addrs {SGLANG_ENGINE_IP}:{SGLANG_ENGINE_PORT} "
        # Weights cross the trainer/rollout boundary over shared storage, the only
        # path that does not assume a shared NCCL fabric with the external engine.
        "--update-weight-transfer-mode disk-delta "
        "--update-weight-disk-dir /root/miles_weight_updates "
        "--update-weight-local-checkpoint-dir /root/miles_rollout_checkpoint "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        before_ray_job_submit=_launch_background,
    )


def _launch_background():
    # Stand up the engine miles will attach to. Real users launch this on their
    # own infra (here, a private Trainium SGLang build); miles starts and owns the
    # router itself and registers this engine with it, so the harness does not.
    _launch_sglang_engine()


def _launch_sglang_engine():
    from sglang.srt.server_args import ServerArgs

    from miles.backends.sglang_utils.sglang_engine import launch_server_process

    print("launch_sglang_engine", flush=True)
    launch_server_process(
        ServerArgs(
            model_path=f"/root/models/{MODEL_NAME}/",
            trust_remote_code=True,
            enable_memory_saver=True,
            host=SGLANG_ENGINE_IP,
            port=SGLANG_ENGINE_PORT,
            tp_size=2,
        )
    )


if __name__ == "__main__":
    prepare()
    execute()
