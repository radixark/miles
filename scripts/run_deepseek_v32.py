from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_HF = (".kv_b_proj.",)
DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_MEGATRON = (
    ".linear_kv_up_proj",
    ".linear_k_up_proj",
    ".linear_v_up_proj",
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "deepseek-ai"
    model_name: str = "DeepSeek-V3.2"
    megatron_model_type: str = "deepseek-v32"
    from_bf16_ckpt: bool = False
    use_single_node: bool = False
    num_gpus_per_node: int = 8
    actor_num_nodes: int | None = None
    actor_num_gpus_per_node: int | None = 8
    rollout_num_gpus: int | None = None
    hardware: Literal["B200", "B300", "GB200", "GB300", "H100", "H200"] = "B200"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 3000
    rollout_mxfp8: bool = False
    rollout_fp8: bool = False
    train_mxfp8: bool = False
    fp8_param_gather: bool = False
    enable_mis: bool = False
    tis_use_rs: bool = True

    def __post_init__(self):
        if self.use_single_node:
            self.actor_num_nodes = 1
            self.actor_num_gpus_per_node = 4
            self.rollout_num_gpus = 4
        assert not (self.rollout_fp8 and self.rollout_mxfp8), "rollout_fp8 and rollout_mxfp8 are mutually exclusive"
        if self.hardware in ("H100", "H200"):
            assert not self.rollout_mxfp8, "MXFP8 rollout is not supported on H100/H200 (no native MXFP8)"
            assert not self.train_mxfp8, "MXFP8 training is not supported on H100/H200 (no native MXFP8)"


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    if args.from_bf16_ckpt:
        U.exec_command(
            f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}-bf16"
        )
    else:
        U.exec_command(
            f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
        )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _prepare_bf16_ckpt(args: ScriptArgs):
    if not args.from_bf16_ckpt:
        U.fp8_cast_bf16(
            path_src=f"{args.model_dir}/{args.model_name}",
            path_dst=f"{args.model_dir}/{args.model_name}-bf16/",
        )


def _prepare_mxfp8_ckpt(args: ScriptArgs):
    if args.rollout_mxfp8:
        extra_args = args.extra_args
        if "--extra-high-precision-layers-hf" not in extra_args:
            extra_args += (
                f" --extra-high-precision-layers-hf {' '.join(DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_HF)} "
            )
        U.exec_command(
            f"python tools/convert_hf_to_mxfp8.py --model-dir {args.model_dir}/{args.model_name}-bf16 "
            f"--save-dir {args.model_dir}/{args.model_name}-MXFP8 "
            f"{extra_args} "
        )


def _prepare_fp8_ckpt(args: ScriptArgs):
    """Convert BF16 checkpoint to block-quant FP8 (for sglang rollout, no MXFP8)."""
    if args.rollout_fp8:
        U.exec_command(
            f"python tools/convert_hf_to_fp8.py "
            f"--model-dir {args.model_dir}/{args.model_name}-bf16 "
            f"--save-dir {args.model_dir}/{args.model_name}-FP8 "
            f"--strategy block --block-size 128 128"
        )


def _prepare_megatron_ckpt(args: ScriptArgs):

    if args.use_single_node:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
            megatron_path=args.megatron_path,
        )

    else:
        extra_args = (
            "--tensor-model-parallel-size 4 "
            "--expert-model-parallel-size 16 "
            "--pipeline-model-parallel-size 6 "
            "--decoder-last-pipeline-num-layers 13 "
            "--expert-tensor-parallel-size 1 "
        )
        multinode = True
        num_nodes = args.actor_num_nodes

        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            multinode=multinode,
            num_nodes=num_nodes,
            extra_args=extra_args,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
            megatron_path=args.megatron_path,
        )


def _prepare_cp(args: ScriptArgs, skip_existing: bool = False):
    if args.use_single_node:
        return
    torch_dist_dst = f"{args.model_local_dir}/{args.model_name}_torch_dist"
    if not (skip_existing and Path(torch_dist_dst).exists()):
        U.rsync_simple(
            path_src=f"{args.model_dir}/{args.model_name}_torch_dist",
            path_dst=torch_dist_dst,
        )

    if args.rollout_mxfp8:
        hf_suffix = "-MXFP8"
    elif args.rollout_fp8:
        hf_suffix = "-FP8"
    else:
        hf_suffix = ""
    hf_name = f"{args.model_name}{hf_suffix}"
    hf_dst = f"{args.model_local_dir}/{hf_name}"
    if not (skip_existing and Path(hf_dst).exists()):
        U.rsync_simple(
            path_src=f"{args.model_dir}/{hf_name}",
            path_dst=hf_dst,
        )


def _execute_train(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.rollout_mxfp8 or args.train_mxfp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-MXFP8"
    elif args.rollout_fp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-FP8"
    else:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint}/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
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

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        "--use-fault-tolerance "
        # f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {
        "SGLANG_NSA_FORCE_MLA": "1",
        # "INDEXER_ROPE_NEOX_STYLE": "1",  # v3.2 uses NeoX (non-interleaved) style; GLM-5 uses interleaved.
        "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
        "NVSHMEM_DISABLE_NCCL": "1",
    }

    if args.train_mxfp8:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300":
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe mxfp8 "
                    # --moe-router-padding-for-quantization
                )
                if args.fp8_param_gather:
                    raise NotImplementedError("FP8 param gather is not supported yet.")
                    misc_args += (
                        "--fp8-param-gather "
                        "--reuse-grad-buf-for-mxfp8-param-ag "
                        "--overlap-param-gather "
                        "--overlap-grad-reduce "
                    )
                    optimizer_args += (
                        "--use-precision-aware-optimizer "
                        # "--offload-optimizer-states "
                    )
                else:
                    optimizer_args += (
                        "--optimizer-cpu-offload "
                        "--overlap-cpu-optimizer-d2h-h2d "
                        "--use-precision-aware-optimizer "
                    )
    else:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    match args.hardware:
        case "B200" | "B300" | "GB200" | "GB300" | "H100" | "H200":
            if args.use_single_node:
                perf_args += (
                    f"--tensor-model-parallel-size {args.actor_num_gpus_per_node} "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 1 "
                    "--context-parallel-size 1 "
                    f"--expert-model-parallel-size {args.actor_num_gpus_per_node} "
                    "--expert-tensor-parallel-size 1 "
                )
            else:
                perf_args += (
                    "--tensor-model-parallel-size 2 "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 4 "
                    "--decoder-last-pipeline-num-layers 13 "
                    "--context-parallel-size 1 "
                    "--expert-model-parallel-size 16 "
                    "--expert-tensor-parallel-size 1 "
                )

            sglang_args = (
                "--sglang-mem-fraction-static 0.8 "
                "--sglang-attention-backend nsa "
                "--sglang-nsa-decode-backend flashmla_sparse "
                "--sglang-nsa-prefill-backend flashmla_sparse "
                "--sglang-kv-cache-dtype bf16 "
                # NSA KV cache requires page_size=64 on CUDA.
                "--sglang-page-size 64 "
            )

            if args.rollout_mxfp8:
                if args.use_single_node:
                    sglang_world_size = 2
                else:
                    sglang_world_size = 8
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {sglang_world_size} "
                    "--sglang-fp8-gemm-backend flashinfer_cutlass "
                    "--sglang-moe-runner-backend flashinfer_trtllm_routed "
                    f"--sglang-tp-size {sglang_world_size} "
                    f"--sglang-dp-size {sglang_world_size} "
                    "--sglang-enable-dp-attention "
                    "--sglang-enable-dp-lm-head "
                    # f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    # f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
                    # "--sglang-moe-dense-tp-size 1 "
                )

                if "--extra-high-precision-layers-hf" not in args.extra_args:
                    misc_args += (
                        f"--extra-high-precision-layers-hf {' '.join(DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_HF)} "
                        f"--extra-high-precision-layers-megatron {' '.join(DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_MEGATRON)} "
                    )

                te_precision_config_text = """
configs:
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
matchers:
  mla_kv_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_kv_up_proj"
    config: "bf16"
  absorbed_k_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_k_up_proj"
    config: "bf16"
  absorbed_v_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_v_up_proj"
    config: "bf16"
""".strip()
                if "--te-precision-config-file" not in args.extra_args:
                    misc_args += f"--te-precision-config-file {U.save_to_temp_file(te_precision_config_text, 'yaml')} "
            else:
                if args.use_single_node:
                    sglang_world_size = 2
                else:
                    sglang_world_size = 8
                sglang_decode_max_bs = 256
                if args.hardware in ("H100", "H200"):
                    sglang_moe_runner_backend = "triton"
                else:
                    sglang_moe_runner_backend = "flashinfer_trtllm_routed"
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {sglang_world_size} "
                    f"--sglang-moe-runner-backend {sglang_moe_runner_backend} "
                    f"--sglang-tp-size {sglang_world_size} "
                    f"--sglang-dp-size {sglang_world_size} "
                    "--sglang-enable-dp-attention "
                    "--sglang-enable-dp-lm-head "
                    # f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    # f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
                    # "--sglang-moe-dense-tp-size 1 "
                )
        case _:
            raise NotImplementedError

    if args.enable_mis:
        config_text = f"""
use_tis: true
use_rs: {"true" if args.tis_use_rs else "false"}
tis_level: "token"
rs_level: "token"
tis_mode: "truncate"
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null
rs_veto_threshold: 1.0e-4
tis_batch_normalize: true
""".strip()
        misc_args += (
            f"--custom-config-path {U.save_to_temp_file(config_text, 'yaml')} "
            "--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

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
        extra_env_vars={**misc_env_vars},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Full pipeline: download, cast, convert, copy, train."""
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_mxfp8_ckpt(args)
    _prepare_fp8_ckpt(args)
    _prepare_megatron_ckpt(args)
    # _prepare_cp(args, skip_existing=True)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download model/data and convert to Megatron checkpoints (run on head node)."""
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_mxfp8_ckpt(args)
    _prepare_fp8_ckpt(args)
    _prepare_megatron_ckpt(args)


@app.command()
@U.dataclass_cli
def prepare_megatron_ckpt(args: ScriptArgs):
    _prepare_megatron_ckpt(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    """Copy model/checkpoint to local storage (run on each node)."""
    _prepare_cp(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training only (assumes prepare and optional prepare-cp are done)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
