"""Inkling 975B fully-asynchronous RL (reasoning / dapo-math-17k).

Disaggregated split on 12 x 4 GPU (GB300): 8 training nodes + 4 dedicated
rollout nodes (one 16-GPU sglang engine). Generation runs continuously in a
background worker (examples/fully_async) and training consumes finished
groups; weights are broadcast to the paused engine between versions.

Example:

    MILES_SCRIPT_NUM_NODES=12 python scripts/run_inkling_975b_fully_async.py train \
        --num-gpus-per-node 4 --num-rollout 100
"""

from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

NUM_LAYERS = 66


@app.callback()
def _callback():
    """Keep `train` as an explicit subcommand (typer collapses single-command apps)."""



@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Inkling"

    num_rollout: int = 100

    hf_checkpoint: str | None = None
    torch_dist: str | None = None
    # Defaults to torch_dist. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    torch_dist_local: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 4
    rollout_num_nodes: int = 4
    lr: float = 1e-6
    rollout_max_response_len: int = 4096
    sglang_context_length: int = 8192
    optimizer_nvme_dir: str = "/tmp/opt_offload"
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)

    enable_r3: bool = True

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.torch_dist is None:
            self.torch_dist = f"{self.model_dir}/{self.model_name}_torch_dist"
        if self.torch_dist_local is None:
            self.torch_dist_local = self.torch_dist
        assert 0 < self.rollout_num_nodes < self.num_nodes
        self.colocate = False
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node


def _get_parallel_config(args: ScriptArgs) -> str:
    """Validated layout for the training half. Other TP / PP / EP combinations
    that fit your compute can be supplied via --extra-args."""
    total_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if total_gpus == 32:  # 8 nodes x 4 GPUs: 66 = 2x33; EP16 <= TP4 x DP4
        return (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 2 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} training GPUs. "
        f"Please specify your parallel config in `run_inkling_975b_fully_async._get_parallel_config`."
    )


def _train(args: ScriptArgs):
    print(
        f"running {args.model_name} fully-async/dapo_math on "
        f"{args.actor_num_nodes} train + {args.rollout_num_nodes} rollout nodes "
        f"x {args.num_gpus_per_node} GPUs"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.torch_dist_local} "
        "--model-name inkling "
        "--megatron-to-hf-mode raw "
        "--no-load-optim --no-load-rng --finetune "
    )
    if args.save_dir is not None:
        ckpt_args += f"--save {args.save_dir}/{args.run_id}/checkpoints --save-interval 10 "

    rollout_args = (
        # continuous background generation; training pulls finished groups
        "--rollout-function-path fully_async_rollout.generate_rollout_fully_async "
        "--pause-generation-mode in_place "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--apply-chat-template "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--eps-clip-c 3.0 "
        "--use-tis "
    )
    if args.enable_r3:
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-distributed-optimizer "
        # bf16 grad accumulation: the 8-node training half holds ~30B expert
        # params per GPU; an fp32 grad buffer (+66GB) does not fit alongside
        # bf16 weights and the fp32 master shards. Megatron force-enables fp32
        # accumulation under bf16 unless --grad-reduce-in-bf16 is set. Watch
        # train_rollout_logprob_abs_diff for numeric drift.
        "--grad-reduce-in-bf16 "
        "--no-check-for-nan-in-loss-and-grad "
        # NVMe-streamed optimizer state (GPU-stepped, one bucket resident).
        # No train-actor disk backup: in the disaggregated split the training
        # half owns its GPUs and never pauses for the rollout engine.
        f"--optimizer-state-nvme-dir {args.optimizer_nvme_dir} "
        "--optimizer-state-nvme-chunk-mb 256 "
    )

    perf_args = _get_parallel_config(args)
    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # fixed micro-batches: dynamic token packing exposes a PP-p2p x EP-a2a
        # NCCL launch-order race on varlen shapes.
        "--micro-batch-size 1 "
    )

    inkling_args = "--inkling-attn-backend flex " "--inkling-freeze-global-scale all "

    sglang_args = (
        "--rollout-num-gpus-per-engine 16 "
        # Dedicated rollout nodes (no training co-tenant). Weights are 111GB
        # per GPU (975B bf16 / TP16); 0.80 x 276GB = 221GB static leaves a
        # ~110GB KV pool (2x the colocated recipe) and ~55GB dynamic headroom
        # for cuda-graph capture and prefill activations. Concurrency and the
        # token cap scale with the doubled KV pool.
        "--sglang-mem-fraction-static 0.80 "
        "--sglang-max-running-requests 128 "
        "--sglang-max-total-tokens 655360 "
        "--sglang-cuda-graph-max-bs 128 "
        "--sglang-attention-backend fa4 "
        "--sglang-moe-runner-backend triton "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--sglang-enable-multimodal "
        f"--sglang-context-length {args.sglang_context_length} "
        "--sglang-disable-custom-all-reduce "
    )

    misc_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--no-bias-dropout-fusion "
        "--distributed-timeout-minutes 30 "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_nodes * args.num_gpus_per_node} "
        "--update-weight-transfer-mode broadcast "
    )

    extra_env_vars = {
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        "SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM": "false",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "MILES_SGLANG_DUMMY_LOAD": "0",
        "SGLANG_SERVER_ENGINE_ROLLOUT_RETURN_LOGPROB": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "NCCL_MNNVL_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_RAS_ENABLE": "0",
        # make examples/fully_async importable for --rollout-function-path
        "PYTHONPATH": f"{U.repo_base_dir}/examples/fully_async",
    }
    extra_env_vars["PYTHONPATH"] = f"{args.megatron_path}:{extra_env_vars['PYTHONPATH']}"

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{perf_args} "
        f"{inkling_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="inkling-975b",
        train_script="train_async.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    app()
