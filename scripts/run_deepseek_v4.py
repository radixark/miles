"""
DeepSeek V4 training script.

Supports:
  - DeepSeek-V4-Flash-FP8         Public FP8 repackage of deepseek-ai/DeepSeek-V4-Flash
                                  (sgl-project/DeepSeek-V4-Flash-FP8, 291B, 43 layers).
                                  Verified full-model profiles: 8 nodes x 8 GPUs on H200
                                  or 8 nodes x 4 GPUs on GB300.
  - DeepSeek-V4-Flash-FP8-4layer  4-layer prune of the above for single-node
                                  smoke testing. **Cannot generate meaningful output -
                                  pipeline-only sanity check.**
  - DeepSeek-V4-Pro-FP8           Verified profile: 32 nodes x 8 GPUs on H200.

Usage patterns:

  1. One-shot full pipeline (download + convert + train):
       python scripts/run_deepseek_v4.py full-train \
           --model-name DeepSeek-V4-Flash-FP8-4layer \
           --num-nodes 1 --num-gpus-per-node 8

  2. Individual steps (download -> FP8->BF16 -> BF16->torch_dist -> rsync -> train):
       python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-single   --model-name DeepSeek-V4-Flash-FP8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-spmd     --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 8 --num-gpus-per-node 8
       python scripts/run_deepseek_v4.py prepare-cp       --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py train            --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 8 --num-gpus-per-node 8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "DeepSeek-V4-Flash-FP8": "sgl-project",
    # 4-layer prune of sgl-project/DeepSeek-V4-Flash-FP8.
    "DeepSeek-V4-Flash-FP8-4layer": "Pinaster",
    "DeepSeek-V4-Pro-FP8": "sgl-project",
}

_MEGATRON_MODEL_TYPE = {
    "DeepSeek-V4-Flash-FP8": "deepseek-v4-flash",
    "DeepSeek-V4-Flash-FP8-4layer": "deepseek-v4-flash-4layer",
    "DeepSeek-V4-Pro-FP8": "deepseek-v4-pro",
}

_PRO_MODEL_NAMES = ("DeepSeek-V4-Pro-FP8",)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_org: str = ""
    model_name: Literal[
        "DeepSeek-V4-Flash-FP8",
        "DeepSeek-V4-Flash-FP8-4layer",
        "DeepSeek-V4-Pro-FP8",
    ] = "DeepSeek-V4-Flash-FP8"

    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    enable_eval: bool = True

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    # Defaults to model_dir. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    model_local_dir: str | None = None
    save_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 8
    enable_mtp: bool = False
    optimizer_offload: bool = True
    use_fault_tolerance: bool = True
    cp_size: int = 1

    # debug configs
    dump_details: bool = False
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    debug_data_root: str = "/root/shared_data"
    skip_saving: bool = False

    # precision configs
    enable_r3: bool = True
    train_deterministic: bool = True
    fp8_training: bool = True
    enable_mis: bool = False

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]
        if self.model_local_dir is None:
            self.model_local_dir = self.model_dir
        if self.model_name in _PRO_MODEL_NAMES:
            self.enable_r3 = False

    @property
    def megatron_model_type(self):
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        if self.model_name == "DeepSeek-V4-Pro-FP8":
            return "DeepSeek-V4-Pro-BF16"
        return f"{self.model_name}-bf16"


def _download_dataset(args: ScriptArgs):
    """Download the task-specific dataset(s)."""
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _hf_checkpoint_path(args: ScriptArgs) -> str:
    """Resolve hf_checkpoint path: explicit override wins, else {model_dir}/{model_name}."""
    return args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"


def _ensure_4layer_model_type(args: ScriptArgs):
    """Undo the old deepseek_ref workaround for local 4-layer prunes."""
    if args.model_name != "DeepSeek-V4-Flash-FP8-4layer":
        return
    cfg = Path(_hf_checkpoint_path(args)) / "config.json"
    if not cfg.exists():
        return
    text = cfg.read_text()
    if '"model_type": "deepseek_ref"' in text:
        cfg.write_text(text.replace('"model_type": "deepseek_ref"', '"model_type": "deepseek_v4"'))
        print(f"[patch] {cfg}: model_type deepseek_ref -> deepseek_v4")


def _prepare_download(args: ScriptArgs):
    """Download HF checkpoint + task dataset. Idempotent: hf skips existing blobs."""
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Only download if the user has NOT supplied a pre-existing checkpoint dir.
    # (prepare_single / train with --hf-checkpoint bypass this.)
    if args.hf_checkpoint is None:
        dest = f"{args.model_dir}/{args.model_name}"
        U.exec_command(f"hf download {args.model_org}/{args.model_name} " f"--local-dir {dest}")
    _ensure_4layer_model_type(args)
    _download_dataset(args)


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs):
    """Download HF checkpoint + dataset from HuggingFace. Run on one node (shared NFS)."""
    _prepare_download(args)


def _prepare_single(args: ScriptArgs):
    _download_dataset(args)

    src = _hf_checkpoint_path(args)
    U.fp8_cast_bf16(
        path_src=src,
        path_dst=f"{args.model_dir}/{args.bf16_name}/",
    )


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    """FP8 -> BF16 cast for Megatron. Needs --hf-checkpoint (or pre-downloaded). One node."""
    _prepare_single(args)


def _prepare_spmd(args: ScriptArgs):
    is_4layer = args.model_name == "DeepSeek-V4-Flash-FP8-4layer"
    extra_args = "--expert-tensor-parallel-size 1 --context-parallel-size 1 "
    if args.num_nodes == 1 and is_4layer:
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        )
    elif args.num_nodes == 8 and args.model_name == "DeepSeek-V4-Flash-FP8":
        extra_args += (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        )
    elif args.num_nodes == 32 and args.num_gpus_per_node == 8 and args.model_name == "DeepSeek-V4-Pro-FP8":
        extra_args += (
            "--tensor-model-parallel-size 8 "
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 32 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
            "--make-vocab-size-divisible-by 32 "
        )
    else:
        raise NotImplementedError(
            f"No verified SPMD conversion config for {args.model_name} "
            f"({args.num_nodes} nodes x {args.num_gpus_per_node} GPUs/node). "
            f"Please specify your conversion parallel config in `run_deepseek_v4.py`."
        )

    num_gpus_for_convert = args.num_gpus_per_node
    if is_4layer:
        num_gpus_for_convert = min(num_gpus_for_convert, 4)

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=True if args.num_nodes > 1 else False,
        num_nodes=args.num_nodes,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _prepare_cp(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.torch_dist_name}",
        path_dst=f"{args.model_local_dir}/{args.torch_dist_name}",
        num_nodes=args.num_nodes,
    )
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
        num_nodes=args.num_nodes,
    )


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    total_gpus = args.num_nodes * args.num_gpus_per_node

    # Single-node smoke-test configs
    if args.num_nodes == 1:
        return (
            f"--tensor-model-parallel-size {args.num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )

    # GB300: 4 GPUs/node
    if args.num_gpus_per_node == 4:
        if total_gpus == 32:  # 8 nodes x 4 GPUs
            return (
                "--tensor-model-parallel-size 2 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 4 "
                "--decoder-last-pipeline-num-layers 3 "
                "--context-parallel-size 2 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
            )

    # H200: 8 GPUs/node
    if args.num_gpus_per_node == 8:
        if total_gpus == 64:  # 8 nodes x 8 GPUs
            return (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 4 "
                "--decoder-last-pipeline-num-layers 3 "
                "--context-parallel-size 2 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
        elif total_gpus == 256:  # 32 nodes x 8 GPUs (Pro)
            return (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 7 "
                "--decoder-last-pipeline-num-layers 6 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 32 "
                "--expert-tensor-parallel-size 1 "
            )

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `run_deepseek_v4._get_parallel_config`."
    )


def _train(args: ScriptArgs):
    print(f"running on {args.num_nodes} nodes")
    _ensure_4layer_model_type(args)

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} " f"--save {load_save_path} " "--save-interval 20 " "--save-retain-interval 20 "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len 4096 "
                """--apply-chat-template-kwargs '{"thinking":true}' """
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 4096 "
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    perf_args = _get_parallel_config(args)

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

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
        optimizer_args += (
            "--optimizer-cpu-offload " "--use-precision-aware-optimizer " "--overlap-cpu-optimizer-d2h-h2d "
        )

    if args.model_name == "DeepSeek-V4-Pro-FP8":
        sglang_world_size = 32
        sglang_tp_size = 32
        sglang_dp_size = 32
        sglang_ep_size = 32
        sglang_a2a_backend = "deepep"
    else:
        sglang_world_size = args.num_gpus_per_node
        sglang_tp_size = sglang_world_size
        sglang_dp_size = sglang_world_size
        sglang_ep_size = sglang_world_size
        sglang_a2a_backend = None
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-dp-size {sglang_dp_size} "
        "--sglang-enable-dp-attention "
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        "--sglang-max-running-requests 64 "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-server-concurrency 1024 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
    )
    if sglang_a2a_backend:
        sglang_args += f"--sglang-moe-a2a-backend {sglang_a2a_backend} " "--sglang-cuda-graph-max-bs 8 "
    if sglang_a2a_backend and args.enable_mtp:
        raise AssertionError("MTP rollout is not supported yet")
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
    sglang_args += f"--sglang-ep-size {sglang_ep_size} "
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
    }
    if args.model_name == "DeepSeek-V4-Pro-FP8":
        extra_env_vars["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "256"
        extra_env_vars["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--sglang-mem-fraction-static 0.7 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--colocate "
    )

    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "

    if args.enable_mis:
        misc_args += (
            "--use-tis "
            "--custom-config-path examples/train_infer_mismatch_helper/mis.yaml "
            "--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "

    if args.debug_train_run_id is not None:
        if args.debug_train_rollout_id is None:
            args.debug_train_rollout_id = 1
        misc_args += (
            f"--load-debug-rollout-data "
            f"{args.debug_data_root}/{args.debug_train_run_id}/dump_details/rollout_data/{args.debug_train_rollout_id}.pt "
        )
        misc_args += "--debug-train-only "

    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
        misc_args += "--use-miles-router "

    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    if args.fp8_training:
        misc_args += "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " "--fp8-recipe blockwise "
        extra_env_vars |= {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
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
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training. Assumes data/model/torch_dist are already prepared on {model_local_dir}."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    _prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")

    if args.model_local_dir != args.model_dir:
        _prepare_cp(args)
    else:
        print(f"[full_train] Skipping rsync: model_local_dir == model_dir ({args.model_dir})")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"

    _train(args)


if __name__ == "__main__":
    app()
