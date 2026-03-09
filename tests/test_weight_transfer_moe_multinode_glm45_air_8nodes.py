# TODO(jensen): may need to merge this file into the main test file in the future.
from dataclasses import dataclass
from typing import Literal
import typer
import miles.utils.external_utils.command_utils as U
from miles.utils.timer import log_experiment_start

MODEL_NAME = "GLM-4.5-Air"
MODEL_TYPE = "glm4.5-106B-A12B"
import time

GPUS_PER_NODE = 8
import os


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["nccl", "rdma", "rdma-shared", "all"] = "all"
    # Recommended 3D parallelism for 32 GPUs (4× H100 nodes):
    # TP=1 (active params only 12B), PP=4 (handle 106B total), EP=8 (128 experts across nodes)
    # → DP = 32 / (TP=1 × PP=4) = 8, EP=8 ≤ DP=8 ✓
    train_tp: int = 1
    train_ep: int = 8
    train_pp: int = 4
    train_cp: int = 1
    train_etp: int = 1
    # Rollout parallelism: 4 engines × 8 GPUs each (EP=8, DP_attn)
    sglang_tp: int = 8
    sglang_ep: int = 8
    # Total Resources: 8 nodes = 64 GPUs, split 50/50
    num_train_gpus: int = 4 * GPUS_PER_NODE  # 4 nodes * 8 GPUs = 32
    num_rollout_gpus: int = 4 * GPUS_PER_NODE  # 4 nodes * 8 GPUs = 32
    # multi-node settings
    multinode: bool = True
    head_node_ip: str | None = None
    node_rank: int = 0
    nnodes: int = 8
    # 46 layers, PP=4: ceil(46/4)=12 per stage, last stage = 46 - 12*3 = 10
    decoder_last_pipeline_num_layers: int = 10
    wait_after: bool = False
    enable_nccl_nvls: bool = False
    bucket_size: float = 1.0
    released_mc_transfer_timeout: bool = False
    no_save_optim: bool = False
    skip_validation: bool = False

    def validate(self):
        if self.multinode:
            assert self.num_train_gpus % GPUS_PER_NODE == 0, "num_train_gpus must be multiple of GPUS_PER_NODE"
            assert self.num_rollout_gpus % GPUS_PER_NODE == 0, "num_rollout_gpus must be multiple of GPUS_PER_NODE"
            assert (
                self.num_train_gpus + self.num_rollout_gpus == self.nnodes * GPUS_PER_NODE
            ), "num_train_gpus + num_rollout_gpus must equal to nnodes * GPUS_PER_NODE"

    def selected_modes(self) -> list[str]:
        if self.mode == "all":
            return ["nccl", "rdma", "rdma-shared"]
        else:
            return [self.mode]


def prepare(args: ScriptArgs):
    if args.node_rank == 0:
        U.exec_command("mkdir -p /root/models /root/datasets")
        U.exec_command(f"hf download zai-org/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
        U.hf_download_dataset("zhuzilin/dapo-math-17k")
        U.hf_download_dataset("zhuzilin/aime-2024")

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


def execute(args: ScriptArgs, mode: str, base_log_dir: str):
    is_rdma = mode in ("rdma", "rdma-shared")

    run_log_dir = f"{base_log_dir}/glm45-air-profile/{mode}"
    os.makedirs(run_log_dir, exist_ok=True)
    os.environ["MILES_LOG_DIR"] = run_log_dir

    log_experiment_start(
        {
            "mode": mode,
            "model": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "num_train_gpus": args.num_train_gpus,
            "num_rollout_gpus": args.num_rollout_gpus,
            "train_tp": args.train_tp,
            "train_ep": args.train_ep,
            "train_pp": args.train_pp,
            "train_cp": args.train_cp,
            "train_etp": args.train_etp,
            "sglang_tp": args.sglang_tp,
            "sglang_ep": args.sglang_ep,
            "multinode": args.multinode,
            "nnodes": args.nnodes,
            "node_rank": args.node_rank,
        }
    )

    # --- Checkpoint ---
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/multinode/{MODEL_NAME}_torch_dist/ "
    if args.no_save_optim:
        ckpt_args += "--no-save-optim "

    # --- Rollout ---
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 13 --rollout-batch-size 4 --n-samples-per-prompt 4 "
        "--rollout-max-response-len 100 --rollout-temperature 0.8 "
        "--global-batch-size 16 --balance-data "
    )

    # --- Training parallelism ---
    perf_args = (
        f"--tensor-model-parallel-size {args.train_tp} "
        f"--pipeline-model-parallel-size {args.train_pp} "
        f"--context-parallel-size {args.train_cp} "
        f"--expert-model-parallel-size {args.train_ep} "
        f"--expert-tensor-parallel-size {args.train_etp} "
        f"--decoder-last-pipeline-num-layers {args.decoder_last_pipeline_num_layers} "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    )
    if args.train_tp > 1:
        perf_args += "--sequence-parallel "

    # --- Eval ---
    eval_args = (
        "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 --eval-max-response-len 16384 --eval-top-p 0.7 "
    )

    # --- GRPO ---
    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 4e-4 "
    )

    # --- Optimizer ---
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    # --- SGLang: 4 engines × 8 GPUs each, with router ---
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.sglang_tp} "
        f"--rollout-num-gpus {args.num_rollout_gpus} "
        "--sglang-mem-fraction-static 0.8 "
        f"--sglang-ep-size {args.sglang_ep} "
        "--sglang-cuda-graph-bs 1 2 4 8 16 "
        "--use-miles-router "
        "--sglang-enable-dp-attention --sglang-enable-dp-lm-head "
    )
    if is_rdma:
        sglang_args += "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
    if args.skip_validation:
        sglang_args += "--sglang-load-format dummy "
    else:
        sglang_args += """--sglang-model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' """

    # --- Misc ---
    mem = int(args.bucket_size * 1024 * 1024 * 1024) if is_rdma else (4 * 1024 * 1024 * 1024)
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_train_gpus // GPUS_PER_NODE} "
        f"--actor-num-gpus-per-node {GPUS_PER_NODE} "
        f"--update-weight-buffer-size {mem} "
    )
    if not args.skip_validation:
        misc_args += "--check-weight-update-equal "
    if is_rdma:
        misc_args += "--update-weight-transfer-mode rdma "
    if mode == "rdma-shared":
        misc_args += "--rdma-shared-buffer "

    # --- Assemble ---
    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} {sglang_args} {misc_args}"
    )

    # Worker nodes start late to give head node time
    if args.node_rank > 0:
        time.sleep(20)

    os.environ["MODEL_ARGS_ROTARY_BASE"] = "1000000"
    mc_transfer_timeout = "300" if args.released_mc_transfer_timeout else "30"
    num_gpus = args.num_train_gpus + args.num_rollout_gpus

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=GPUS_PER_NODE,
        megatron_model_type=MODEL_TYPE,
        train_script="train.py",
        extra_env_vars={
            "MC_TRANSFER_TIMEOUT": mc_transfer_timeout,
            "RAY_DEBUG": "1",
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1" if args.enable_nccl_nvls else "0",
            "MILES_LOG_DIR": run_log_dir,
        },
        multinode=args.multinode,
        is_head_node=args.node_rank == 0,
        num_gpus=num_gpus,
    )

    if args.node_rank > 0 and args.wait_after:
        time.sleep(3600)


@U.dataclass_cli
def main(args: ScriptArgs):
    args.validate()
    prepare(args)
    base_log_dir = os.environ.get("MILES_LOG_DIR", "/root")
    for mode in args.selected_modes():
        print(f"\n{'='*60}")
        print(f"  Running: {MODEL_NAME} / {mode}")
        print(f"{'='*60}\n")
        execute(args, mode, base_log_dir)


if __name__ == "__main__":
    typer.run(main)
