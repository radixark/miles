from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MEGATRON_MODEL_TYPE = {"DeepSeek-V4.1": "deepseek-v4.1"}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_name: Literal["DeepSeek-V4.1"] = "DeepSeek-V4.1"
    task: Literal["dapo_aime", "gsm8k"] = "gsm8k"
    enable_eval: bool = False

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    save_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H200", "B200", "B300", "GB200", "GB300"] = "auto"
    tp_size: int = 4
    pp_size: int = 1
    cp_size: int = 1
    sequence_parallel: bool = True
    ep_size: int | None = None
    optimizer_offload: bool = False
    disk_offload: bool = False
    offload_disk_dir: str | None = None
    colocate_memory_peak_device: Literal["cpu", "gpu"] | None = None
    rollout_gpus_per_engine: int | None = None
    sglang_mem_fraction_static: float = 0.6
    recompute: Literal["none", "full", "selective"] = "none"
    grad_reduce_bf16: bool = False
    check_weight_update: bool = True
    sglang_cuda_graph: bool = False
    sglang_cuda_graph_max_bs: int = 128
    sglang_engram_host_table: bool = False
    sglang_max_running_requests: int = 128
    sglang_radix_cache: bool = False
    rollout_num_nodes: int = 0
    fully_async: bool = False
    load_from_hf: bool = False
    use_fault_tolerance: bool = False

    num_rollout: int = 5
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int | None = None
    max_tokens_per_gpu: int = 2048

    dump_details: bool = False
    debug_data_root: str = "/root/shared_data"
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    skip_saving: bool = True

    enable_r3: bool = True
    enable_indexer_replay: bool = False
    train_deterministic: bool = True
    extra_args: str = ""

    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)
    rollout_num_gpus: int = field(init=False)

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        assert 0 <= self.rollout_num_nodes < self.num_nodes
        self.colocate = self.rollout_num_nodes == 0
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node
        if self.ep_size is None:
            self.ep_size = self.actor_num_nodes * self.num_gpus_per_node // self.pp_size
        if self.colocate:
            self.rollout_num_gpus = self.num_nodes * self.num_gpus_per_node
        else:
            self.rollout_num_gpus = self.rollout_num_nodes * self.num_gpus_per_node
        if self.rollout_max_response_len is None:
            self.rollout_max_response_len = 256 if self.task == "gsm8k" else 4096

    @property
    def megatron_model_type(self):
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _parallel_args(args: ScriptArgs) -> str:
    return (
        f"--tensor-model-parallel-size {args.tp_size} "
        f"--pipeline-model-parallel-size {args.pp_size} "
        f"--context-parallel-size {args.cp_size} "
        f"{'--allgather-cp ' if args.cp_size > 1 else ''}"
        f"{'--sequence-parallel ' if args.sequence_parallel else ''}"
        f"--expert-model-parallel-size {args.ep_size} "
        "--expert-tensor-parallel-size 1 "
    )


def _prepare_spmd(args: ScriptArgs):
    assert args.hf_checkpoint is not None
    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=args.hf_checkpoint,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=False,
        num_nodes=1,
        extra_args=(
            "--dsv4-impl miles "
            f"--tensor-model-parallel-size {args.tp_size} "
            "--pipeline-model-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
            "--context-parallel-size 1 "
        ),
        dir_dst=args.model_dir,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


@app.command()
@U.dataclass_cli
def prepare_data(args: ScriptArgs):
    _download_dataset(args)


def _train(args: ScriptArgs):
    assert args.hf_checkpoint is not None
    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ref_load = args.hf_checkpoint if args.load_from_hf else f"{args.model_dir}/{args.torch_dist_name}"
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {ref_load} "
    if not args.skip_saving:
        ckpt_args += f"--load {load_save_path} --save {load_save_path} --save-interval 20 --save-retain-interval 20 "

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )
    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )
    eval_args = ""
    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                """--apply-chat-template-kwargs '{"thinking_mode":"thinking"}' """
            )
            if args.enable_eval:
                eval_args += (
                    "--eval-interval 20 --eval-top-p 0.7 "
                    f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                    "--n-samples-per-eval-prompt 8 --eval-max-response-len 4096 "
                )
        case "gsm8k":
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet " "--input-key messages "
            if args.enable_eval:
                eval_args += (
                    "--eval-interval 20 --eval-top-p 0.7 "
                    f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                    "--n-samples-per-eval-prompt 1 --eval-max-response-len 256 "
                )

    perf_args = _parallel_args(args) + f"--micro-batch-size 1 --max-tokens-per-gpu {args.max_tokens_per_gpu} "

    grpo_args = (
        "--advantage-estimator grpo "
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
    if args.optimizer_offload:
        optimizer_args += "--optimizer-cpu-offload --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d "
    if args.disk_offload:
        optimizer_args += "--stream-optimizer-state-to-disk "
        if args.colocate:
            optimizer_args += "--offload-train-target disk "
        if args.offload_disk_dir is not None:
            optimizer_args += f"--offload-train-disk-dir {args.offload_disk_dir} "

    engine_gpus = args.rollout_gpus_per_engine or args.num_gpus_per_node
    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--sglang-tp-size {engine_gpus} "
        "--sglang-dp-size 1 "
        f"--sglang-ep-size {engine_gpus} "
        "--sglang-attention-backend dsv4 "
        "--sglang-moe-runner-backend auto "
        f"{f'--sglang-cuda-graph-max-bs-decode {args.sglang_cuda_graph_max_bs} ' if args.sglang_cuda_graph else '--sglang-disable-cuda-graph '}"
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"{'' if args.sglang_radix_cache else '--sglang-disable-radix-cache '}"
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
        "SGLANG_OPT_FP8_WO_A_GEMM": "0",
        "SGLANG_OPT_FUSE_WQA_WKV": "0",
        "SGLANG_DISABLE_MULTIMEM_AG": "1",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1" if args.sglang_engram_host_table else "0",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTHONFAULTHANDLER": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }

    peak_device = args.colocate_memory_peak_device or ("gpu" if args.hardware == "GB300" else "cpu")
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        f"{'--grad-reduce-in-bf16' if args.grad_reduce_bf16 else '--accumulate-allreduce-grads-in-fp32'} "
        f"{f'--colocate --colocate-memory-peak-device {peak_device} ' if args.colocate else f'--rollout-num-gpus {args.rollout_num_gpus} '}"
        f"{'--fully-async ' if args.fully_async else ''}"
        "--dsv4-impl miles "
        "--model-name deepseekv41 "
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        f"{'--check-weight-update-equal --check-weight-update-skip-list engram_hasher. engram.embed. ' if args.check_weight_update else ''}"
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--transformer-impl transformer_engine "
        "--bf16 "
    )
    if args.recompute == "full":
        misc_args += "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    elif args.recompute == "selective":
        misc_args += "--recompute-granularity selective --recompute-modules moe mlp "
    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "
    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "
    if args.debug_train_run_id is not None:
        rollout_id = args.debug_train_rollout_id or 1
        misc_args += (
            f"--load-debug-rollout-data {args.debug_data_root}/{args.debug_train_run_id}/dump_details/rollout_data/{rollout_id}.pt "
            "--debug-train-only "
        )
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
    if 129280 % (64 * args.tp_size):
        misc_args += f"--make-vocab-size-divisible-by {256 // args.tp_size} "
    if args.enable_indexer_replay:
        misc_args += "--use-rollout-indexer-replay --sglang-max-total-tokens 2097152 "
    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    app()
