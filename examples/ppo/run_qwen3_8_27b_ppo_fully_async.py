from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

# Fully-async PPO (actor + critic) with the Megatron backend on 2 training nodes + 2 rollout
# nodes.
#
# This is the disaggregated counterpart of run_qwen3_4b_ppo.py. There the actor, critic and
# inference engines time-share one node under --colocate; here the 16 rollout GPUs run SGLang
# continuously (--fully-async keeps --async-max-concurrent-samples trajectories in flight and
# trains as soon as a batch of finished groups is available), and the 16 training GPUs hold the
# actor and the critic. Three things follow from that split:
#
#   * --use-rollout-logprobs is mandatory. The rollout engines are up to --max-weight-staleness
#     weight versions behind the trainer, so the behaviour policy's log probs have to come from
#     the engine that generated the sample, not from a fresh actor forward pass.
#   * The critic is placed on the actor's GPUs (see the README), so the two models share every
#     card. They stay resident together (--no-offload-train): at PP2 the pair peaks at about
#     66 GB (actor phase) + 29 GB (idle critic) of 140 GB, provided the idle model has released
#     its allocator cache and the allocator does not fragment -- hence expandable_segments below.
#   * PP > 1 is needed so that each model's parameters and gradients are spread over 8 GPUs per
#     replica; TP is capped at 4 by the model's 4 KV groups.
#
# python examples/ppo/run_qwen3_8_27b_ppo_fully_async.py


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3.8-27B"
    megatron_model_type: str = "qwen3.8-27B"
    num_gpus_per_node: int = 8
    # actor world size, and therefore the critic's too: must equal TP * PP * CP * DP below.
    actor_num_nodes: int = 2
    # one single-GPU SGLang engine per rollout GPU.
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

    # --critic-load / --critic-lr / --critic-save fall back exactly as in run_qwen3_4b_ppo.py.
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
        # Trajectories kept in flight on the 16 engines; a finished group is submitted for training
        # as soon as it completes, and a new prompt takes its slot.
        "--async-max-concurrent-samples 128 "
        "--rollout-submission-granularity sample "
        # A group generated under weight version v is still trained on after the update to v+1;
        # older groups are dropped (rollout/fully_async/stale_groups_filtered).
        "--max-weight-staleness 1 "
        # in_place freezes in-flight generation for the weight update and resumes it; abort is
        # rejected by --fully-async and retract would regenerate everything in flight.
        "--pause-generation-mode in_place "
        # The behaviour policy's log probs come from the engine that generated the sample.
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
        # Master weights and Adam moments live on the host; on the GPU each model keeps only its
        # bf16 parameters and gradients.
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
        # Both models resident; see the header. The linear-attention Triton kernels benchmark new
        # sequence-length buckets at runtime with memory the caching allocator never hands back, so
        # the allocator must not hoard fragmented blocks (measured: 110 GB reserved for 66 GB live
        # without expandable segments, 47 GB for 43 GB with them).
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
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
