# TODO(jensen): may need to merge this file into the main test file in the future.
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U
from miles.utils.timer import log_experiment_start

MODEL_NAME = "Moonlight-16B-A3B-Instruct"
MODEL_TYPE = "moonlight"

GPUS_PER_NODE = 8
# For h100 80g * 8:
# training gpu cannot be only 1 because of oom


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["nccl", "rdma"] = "nccl"
    # Right now tp=ep=pp=1
    train_tp: int = 8
    train_ep: int = 1
    train_pp: int = 1
    train_etp: int = 8
    sglang_tp: int = 8
    sglang_dp: int = 1
    sglang_ep: int = 1
    sglang_pp: int = 1
    # Total Ressources
    num_train_gpus: int = 8
    num_rollout_gpus: int = 8
    # Optimizations
    pipelined_transfer: bool = False
    # Profiling
    use_pytorch_profiler_update_weight: bool = False
    # multi-node settings
    multinode: bool = False
    head_node_ip: str | None = None
    node_rank: int = 0
    nnodes: int = 1

    def validate(self):
        if self.multinode:
            assert self.num_train_gpus % GPUS_PER_NODE == 0, "num_train_gpus must be multiple of GPUS_PER_NODE"
            assert self.num_rollout_gpus % GPUS_PER_NODE == 0, "num_rollout_gpus must be multiple of GPUS_PER_NODE"
            assert (
                self.num_train_gpus + self.num_rollout_gpus == self.nnodes * GPUS_PER_NODE
            ), "num_train_gpus + num_rollout_gpus must equal to nnodes * GPUS_PER_NODE"


def prepare(args: ScriptArgs):
    if args.node_rank == 0:
        U.exec_command("mkdir -p /root/models /root/datasets")
        U.exec_command(
            "hf download moonshotai/Moonlight-16B-A3B-Instruct --local-dir /root/models/Moonlight-16B-A3B-Instruct"
        )
        U.hf_download_dataset("zhuzilin/dapo-math-17k")
    num_gpus = args.num_train_gpus + args.num_rollout_gpus
    if not args.multinode:
        U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=num_gpus)
    else:
        # NOTE: currently when it comes to multinode case, all gpus of training/rollout should be multiple of GPUS_PER_NODE
        # Convert training/rollout nodes separately
        # assert args.num_train_gpus % args.num_rollout_gpus == 0 or args.num_rollout_gpus % args.num_train_gpus == 0
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=GPUS_PER_NODE,
            multinode=True,
            master_addr=args.head_node_ip,
            nnodes=args.nnodes,
            dir_dst="/root/multinode",
            node_rank=args.node_rank,
        )


def execute(args: ScriptArgs):
    # Log experiment configuration at the start
    log_experiment_start(
        {
            "mode": args.mode,
            "num_train_gpus": args.num_train_gpus,
            "num_rollout_gpus": args.num_rollout_gpus,
            "train_tp": args.train_tp,
            "train_ep": args.train_ep,
            "train_pp": args.train_pp,
            "sglang_tp": args.sglang_tp,
            "sglang_dp": args.sglang_dp,
            "sglang_ep": args.sglang_ep,
            "sglang_pp": args.sglang_pp,
            "pipelined_transfer": args.pipelined_transfer,
            "multinode": args.multinode,
            "nnodes": args.nnodes,
            "node_rank": args.node_rank,
            "model": MODEL_NAME,
        }
    )

    if args.multinode:
        num_gpus_per_node = 8
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
            f"--ref-load /root/multinode/{MODEL_NAME}_torch_dist_nodes_{args.nnodes} "
        )
    else:
        num_gpus_per_node = args.num_train_gpus + args.num_rollout_gpus
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "
    num_gpus = args.num_train_gpus + args.num_rollout_gpus

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )
    # Training parallellism settings
    perf_args = (
        f"--tensor-model-parallel-size {args.train_tp} "
        "--sequence-parallel "  # NOTE: necessary: ```ValueError: During training, performance may degrade if MoE and tensor parallelismare enabled without also enabling sequence parallelism.```
        # f"--context-parallel-size {args.train_cp} "
        f"--pipeline-model-parallel-size {args.train_pp} "
        f"--expert-model-parallel-size {args.train_ep} "
        f"--expert-tensor-parallel-size {args.train_etp} "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
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
        f"--rollout-num-gpus-per-engine {args.sglang_tp} "
        f"--rollout-num-gpus {args.num_rollout_gpus} "
        f"--sglang-data-parallel-size {args.sglang_dp} "
        f"--sglang-expert-parallel-size {args.sglang_ep} "
        f"--sglang-pipeline-parallel-size {args.sglang_pp} "
        "--sglang-mem-fraction-static 0.8 "
    )
    if args.mode == "rdma":
        sglang_args += "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
    if args.pipelined_transfer and args.mode == "rdma":
        sglang_args += "--rdma-pipelined-transfer "

    # ci_args = "--ci-test "

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
        f"--actor-num-gpus-per-node {args.num_train_gpus} "
        # 1GB buffer for weight update
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        # enable correctness check
        f"--check-weight-update-equal "
    )
    if args.mode == "rdma":
        misc_args += "--update-weight-transfer-mode rdma "

    profile_args = ""
    if bool(args.use_pytorch_profiler_update_weight):
        profile_args += (
            "--use-pytorch-profiler-update-weight "
            "--profile-update-weight-start 2 "
            "--profile-update-weight-end 3 "
            "--tensorboard-dir /root/profiler_logs/ "
        )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        # f"{ci_args} "
        f"{misc_args} "
        f"{profile_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars={"RAY_DEBUG": "1"},
        multinode=args.multinode,
        is_head_node=args.node_rank == 0,
        num_gpus=num_gpus,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    args.validate()
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
