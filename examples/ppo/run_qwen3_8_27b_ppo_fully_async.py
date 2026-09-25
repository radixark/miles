from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

# Fully-async Qwen3.8-27B PPO on 2 training + 2 rollout nodes; see README.md.


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3.8-27B"
    megatron_model_type: str = "qwen3.8-27B"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    rollout_num_gpus: int = 16
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}/ "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 300 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    async_args = (
        "--fully-async "
        "--async-max-concurrent-samples 128 "
        "--rollout-submission-granularity sample "
        "--max-weight-staleness 1 "
        "--pause-generation-mode in_place "
        # Async samples need the generating policy's log-probs.
        "--use-rollout-logprobs "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )

    ppo_args = (
        "--advantage-estimator ppo "
        "--critic-lr 1e-5 "
        "--num-critic-only-steps 1 "
        "--normalize-advantages "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.8 " "--sglang-max-running-requests 512 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        # Expandable segments limit fragmentation while actor and critic share the GPUs.
        "--no-offload-train "
        '--train-env-vars \'{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}\' '
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{async_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
        train_script="train_async.py",
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    assert U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), (
        "Start a four-node Ray cluster, then run with MILES_SCRIPT_EXTERNAL_RAY=1."
    )
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
