# TODO(jensen): may need to merge this file into the main test file in the future.
from dataclasses import dataclass
from typing import Literal
import typer
import miles.utils.external_utils.command_utils as U
from miles.utils.timer import log_experiment_start

MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507"
MODEL_TYPE = "qwen3-235B-A22B"
import time

GPUS_PER_NODE = 8
# For h100 80g * 8:
# training gpu cannot be only 1 because of oom
import os


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["nccl", "rdma"] = "nccl"
    # Right now tp=ep=pp=1
    train_tp: int = 4
    train_ep: int = 16
    train_pp: int = 4
    train_cp: int = 2
    train_etp: int = 1
    sglang_tp: int = 32  # NOTE: for sglang, moe_tp_size = tp_size // ep_size
    sglang_dp: int = 1  # baseline NCCL hangs when dp = 4 and nccl 2.27
    sglang_ep: int = 32
    sglang_pp: int = 1
    # Total Ressources
    num_train_gpus: int = 8 * GPUS_PER_NODE  # 8 nodes * 8 GPUs
    num_rollout_gpus: int = 64  # 8 nodes * 8 GPUs for rollout
    # Optimizations
    pipelined_transfer: bool = False
    # Profiling
    use_pytorch_profiler_update_weight: bool = False
    # multi-node settings
    multinode: bool = True
    head_node_ip: str | None = None
    node_rank: int = 0
    nnodes: int = 16
    # inter_node_transfer_engine_info_port: int = 15500  # TODO: initialize this port from ray.
    decoder_last_pipeline_num_layers: int = 22
    wait_after: bool = False
    enable_nccl_nvls: bool = False
    bucket_size: float = 1.0

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
            "hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /root/models/Qwen3-235B-A22B-Instruct-2507"
        )
        U.hf_download_dataset("zhuzilin/dapo-math-17k")
        U.hf_download_dataset("zhuzilin/aime-2024")
        # hf download --repo-type dataset zhuzilin/aime-2024
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
            decoder_last_pipeline_num_layers=args.decoder_last_pipeline_num_layers,
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
            "train_cp": args.train_cp,
            "train_etp": args.train_etp,
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
            f"--ref-load /root/multinode/{MODEL_NAME}_torch_dist/ "
            f"--load /root/multinode/{MODEL_NAME}_slime_nodes/ "
            f"--save /root/multinode/{MODEL_NAME}_slime_nodes/ "
            "--save-interval 20 "
        )
    else:
        num_gpus_per_node = args.num_train_gpus + args.num_rollout_gpus
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
            f"--ref-load /root/{MODEL_NAME}_torch_dist "
            f"--load /root/{MODEL_NAME}_slime "
            f"--save /root/{MODEL_NAME}_slime "
        )
    num_gpus = args.num_train_gpus + args.num_rollout_gpus

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    # Training parallellism settings
    perf_args = (
        f"--tensor-model-parallel-size {args.train_tp} "
        "--sequence-parallel "  # NOTE: necessary: ```ValueError: During training, performance may degrade if MoE and tensor parallelismare enabled without also enabling sequence parallelism.```
        f"--pipeline-model-parallel-size {args.train_pp} "
        f"--context-parallel-size {args.train_cp} "
        f"--expert-model-parallel-size {args.train_ep} "
        f"--expert-tensor-parallel-size {args.train_etp} "
        f"--decoder-last-pipeline-num-layers {args.decoder_last_pipeline_num_layers} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    # Evaluation settings
    eval_args = (
        # "--eval-interval 20 "  # Commented out in original script
        "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 "
        "--eval-top-p 0.7 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        # "--use-kl-loss "  # Commented out in original script
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.sglang_tp} "
        f"--rollout-num-gpus {args.num_rollout_gpus} "
        "--sglang-mem-fraction-static 0.75 "
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {args.sglang_dp} "
        f"--sglang-ep-size {args.sglang_ep} "
        "--sglang-enable-dp-lm-head "
        "--sglang-cuda-graph-bs 1 2 4 8 16 "
        # "--sglang-moe-a2a-backend deepep "
    )
    if args.mode == "rdma":
        sglang_args += "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
    if args.sglang_dp > 1:
        sglang_args += "--sglang-enable-dp-attention "
    mem = (
        int(args.bucket_size * 1024 * 1024 * 1024)
        if args.pipelined_transfer and args.mode == "rdma"
        else (4 * 1024 * 1024 * 1024)
    )
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
        f"--actor-num-nodes {args.num_train_gpus // GPUS_PER_NODE} "
        f"--actor-num-gpus-per-node {GPUS_PER_NODE} "
        # 4GB buffer for weight update
        f"--update-weight-buffer-size {mem} "
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
    profile_args += (
        "--use-pytorch-profiler-update-weight "
        "--profile-update-weight-start 2 "
        "--profile-update-weight-end 3 "
        "--tensorboard-dir /root/newnew_profiler_logs/ "
    )
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        # f"{ci_args} "
        f"{misc_args} "
        f"{profile_args} "
    )
    if args.node_rank > 0:
        time.sleep(20)
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
        train_script="train.py",
        extra_env_vars={
            "RAY_DEBUG": "1",
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": (
                "1" if args.enable_nccl_nvls else "0"
            ),  # Assuming NVLINK is available for multi-node setup
        },
        multinode=args.multinode,
        is_head_node=args.node_rank == 0,
        num_gpus=num_gpus,
    )
    if args.node_rank > 0 and args.wait_after:
        if args.mode == "nccl":
            time.sleep(800)
        else:
            time.sleep(3600)


@U.dataclass_cli
def main(args: ScriptArgs):
    args.validate()
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
