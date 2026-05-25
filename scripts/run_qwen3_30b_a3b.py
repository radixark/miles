from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-30B-A3B"
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: int | None = None
    hardware: Literal["H100", "B200", "B300", "GB200", "GB300"] = "H100"
    enable_eval: bool = True
    tensor_model_parallel_size: int | None = None
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int | None = None
    cp_comm_type: Literal["p2p", "a2a", "all_gather", "allgather", "a2a+p2p"] | None = None
    expert_model_parallel_size: int | None = None
    expert_tensor_parallel_size: int = 1
    rollout_num_gpus: int | None = None
    rollout_num_gpus_per_engine: int | None = None
    sglang_expert_parallel_size: int | None = None
    use_sequence_parallel: bool = True
    max_tokens_per_gpu: int = 32768
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_fp8: bool = False
    rollout_mxfp8: bool = False
    rollout_int4: bool = False
    rollout_attn_fp8: bool = False
    train_fp8: bool = False
    train_mxfp8: bool = False
    enable_megatron_bridge: bool = False
    enable_mis: bool = False
    tis_use_rs: bool = True

    def __post_init__(self):
        if self.cp_comm_type == "allgather":
            self.cp_comm_type = "all_gather"
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.tensor_model_parallel_size is None:
            self.tensor_model_parallel_size = 4
        if self.context_parallel_size is None:
            self.context_parallel_size = 1
        if self.expert_model_parallel_size is None:
            if self.hardware == "H100":
                self.expert_model_parallel_size = 8
            else:
                self.expert_model_parallel_size = self.num_gpus_per_node if self.train_mxfp8 else 4
        if self.rollout_num_gpus_per_engine is None:
            if self.rollout_fp8:
                self.rollout_num_gpus_per_engine = 2
            elif self.rollout_int4:
                self.rollout_num_gpus_per_engine = 1
            elif self.hardware == "H100":
                self.rollout_num_gpus_per_engine = 8
            else:
                self.rollout_num_gpus_per_engine = 4
        if self.sglang_expert_parallel_size is None:
            self.sglang_expert_parallel_size = 1
        if self.sglang_expert_parallel_size > self.rollout_num_gpus_per_engine:
            raise ValueError(
                "sglang_expert_parallel_size cannot exceed rollout_num_gpus_per_engine "
                f"({self.sglang_expert_parallel_size} > {self.rollout_num_gpus_per_engine})"
            )
        if self.rollout_int4:
            assert not self.rollout_fp8, "rollout_int4 and rollout_fp8 cannot be enabled at the same time"
            assert not self.rollout_mxfp8, "rollout_int4 and rollout_mxfp8 cannot be enabled at the same time"
        if self.rollout_mxfp8:
            assert not self.rollout_fp8, "rollout_mxfp8 and rollout_fp8 cannot be enabled at the same time"
            assert self.hardware in ("B200", "B300", "GB200", "GB300"), "rollout_mxfp8 only supports Blackwell GPUs"
        if self.train_mxfp8:
            assert not self.train_fp8, "train_mxfp8 and train_fp8 cannot be enabled at the same time"
            assert self.hardware in ("B200", "B300", "GB200", "GB300"), "train_mxfp8 only supports Blackwell GPUs"
            assert self.rollout_mxfp8, "train_mxfp8 requires rollout_mxfp8 to be enabled"


def prepare(args: ScriptArgs, *, convert_checkpoint_kwargs: dict | None = None):
    """Download HF assets and (optionally) convert to torch-dist.

    `convert_checkpoint_kwargs` lets callers (e.g. the deterministic variant)
    override or extend the convert_checkpoint call without baking
    true-on-policy concerns into this base script.
    """
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    if args.rollout_fp8:
        U.exec_command(f"hf download Qwen/{args.model_name}-FP8 --local-dir {args.model_dir}/{args.model_name}-FP8")

    if args.rollout_mxfp8:
        U.exec_command(
            f"python tools/convert_hf_to_mxfp8.py --model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-MXFP8 "
            f"{args.extra_args} "
        )

    if args.rollout_int4:
        U.exec_command(
            f"python tools/convert_hf_to_int4_direct.py --model-dir {args.model_dir}/{args.model_name} --save-dir {args.model_dir}/{args.model_name}-INT4"
        )

    if not args.enable_megatron_bridge:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}",
            megatron_path=args.megatron_path,
            **(convert_checkpoint_kwargs or {}),
        )


# TODO improve layering: split algorithm vs infra
def execute(args: ScriptArgs):
    is_debug_mode = args.mode == "debug_minimal"
    megatron_load_path = (
        f"{args.model_dir}/{args.model_name}/"
        if args.enable_megatron_bridge
        else f"{args.model_dir}/{args.model_name}_torch_dist"
    )
    ref_load_path = megatron_load_path
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.rollout_fp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-FP8"
    elif args.train_mxfp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-MXFP8"
    elif args.rollout_int4:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-INT4"
    else:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint}/ "
        f"--ref-load {ref_load_path} "
        f"--load {megatron_load_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if is_debug_mode else 20} "
        f"--save-retain-interval {2 if is_debug_mode else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if (not is_debug_mode) and args.enable_eval:
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
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
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
        # need to comment this when using model with MLA
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.rollout_int4:
        misc_env_vars |= {
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
        }

    if args.train_fp8 or args.train_mxfp8:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300":
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe mxfp8 "
                    # "--fp8-param-gather "
                    # "--reuse-grad-buf-for-mxfp8-param-ag "
                    # --moe-router-padding-for-quantization
                )
            case "H100" | "H200":
                # ref: fp8 blog
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe blockwise "
                    # "--fp8-param-gather "
                )
                misc_env_vars |= {
                    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
                }

    if args.enable_megatron_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    match (args.hardware, args.num_nodes):
        case ("H100", 1):
            perf_args += (
                f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
                f"{'--sequence-parallel ' if args.use_sequence_parallel and args.tensor_model_parallel_size > 1 else ''}"
                f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
                f"--context-parallel-size {args.context_parallel_size} "
                f"{f'--cp-comm-type {args.cp_comm_type} ' if args.cp_comm_type is not None else ''}"
                f"--expert-model-parallel-size {args.expert_model_parallel_size} "
                f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
            )
            sglang_args = (
                f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
                f"{f'--rollout-num-gpus {args.rollout_num_gpus} ' if args.rollout_num_gpus is not None else ''}"
                f"{f'--sglang-ep-size {args.sglang_expert_parallel_size} ' if args.sglang_expert_parallel_size > 1 else ''}"
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-max-bs 512 "
            )
            optimizer_args += (
                "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )
        case ("B200" | "B300" | "GB200" | "GB300", 1 | 2 | 4):
            perf_args += (
                f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
                f"{'--sequence-parallel ' if args.use_sequence_parallel and args.tensor_model_parallel_size > 1 else ''}"
                f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
                f"--context-parallel-size {args.context_parallel_size} "
                f"{f'--cp-comm-type {args.cp_comm_type} ' if args.cp_comm_type is not None else ''}"
                f"--expert-model-parallel-size {args.expert_model_parallel_size} "
                f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
            )
            sglang_args = (
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-attention-backend trtllm_mha "
                f"{f'--rollout-num-gpus {args.rollout_num_gpus} ' if args.rollout_num_gpus is not None else ''}"
                f"{f'--sglang-ep-size {args.sglang_expert_parallel_size} ' if args.sglang_expert_parallel_size > 1 else ''}"
            )
            if args.rollout_fp8:
                sglang_world_size = 2
                sglang_attn_tp_size = 2
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
                    f"--sglang-ep-size {sglang_world_size} "
                    "--sglang-moe-runner-backend deep_gemm "
                    "--sglang-moe-a2a-backend deepep "
                    f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
                )
            elif args.rollout_mxfp8:
                sglang_world_size = 1
                sglang_attn_tp_size = 1
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
                    "--sglang-fp8-gemm-backend triton "
                    # Currently, only cutlass moe runner is supported in sglang for mxfp8, which does not support ep
                    # f"--sglang-ep-size {sglang_world_size} "
                    "--sglang-moe-runner-backend cutlass "
                    # TODO: mxfp8 deepep and deepgemm is not supported in sglang yet
                    # "--sglang-moe-a2a-backend deepep "
                    f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
                )
            else:
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
                    "--sglang-cuda-graph-max-bs 512 "
                )
        case _:
            raise NotImplementedError

    if args.rollout_attn_fp8:
        sglang_args += "--sglang-kv-cache-dtype fp8_e4m3 "

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
        extra_env_vars=misc_env_vars,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
