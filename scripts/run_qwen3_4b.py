from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U
from miles.true_on_policy import (
    apply_true_on_policy_script_defaults,
    build_true_on_policy_launch_plan,
    get_megatron_model_type,
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal", "debug_one_sample"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str | None = None
    num_gpus_per_node: int | None = None
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    multi_eval: bool = False
    true_on_policy: bool = False
    sglang_rl_on_policy_target: str | None = None
    true_on_policy_contract: str | None = None
    dynamic_sampling: bool = False
    enable_eval: bool = True
    train_backend: Literal["fsdp", "megatron"] = "megatron"
    rollout_fp8: bool = False
    train_fp8: bool = False
    enable_megatron_bridge: bool = False
    enable_mis: bool = False
    use_kl_loss: bool = True
    tis_use_rs: bool = True

    def __post_init__(self):
        if self.train_backend == "megatron":
            self.megatron_model_type = get_megatron_model_type(self.model_name)

        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]

        # Derived parallelism defaults for Qwen3 dense models
        self.tensor_model_parallel_size = 1 if self.model_name == "Qwen3-0.6B" else 2
        self.pipeline_model_parallel_size = 1
        self.context_parallel_size = 1 if self.model_name == "Qwen3-0.6B" else 4
        self.cp_comm_type = "a2a" if self.context_parallel_size > 1 else None
        self.use_sequence_parallel = self.tensor_model_parallel_size > 1
        self.max_tokens_per_gpu = 32768 if self.train_backend == "fsdp" else 9216
        self.rollout_num_gpus_per_engine = 1
        self.train_memory_margin_bytes = 3221225472

        apply_true_on_policy_script_defaults(self)
        if self.train_backend == "megatron" and self.enable_megatron_bridge and self.use_kl_loss:
            raise ValueError(
                "Megatron bridge mode does not provide a Megatron-format ref checkpoint for KL. "
                "Disable KL loss or disable bridge mode."
            )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    if args.multi_eval:
        U.hf_download_dataset("zyzshishui0627/gpqa_diamond", data_dir=args.data_dir)
        U.hf_download_dataset("zyzshishui0627/IFBench", data_dir=args.data_dir)

    if args.rollout_fp8:
        U.exec_command(f"hf download Qwen/{args.model_name}-FP8 --local-dir {args.model_dir}/{args.model_name}-FP8")

    if (args.train_backend == "megatron") and not args.enable_megatron_bridge:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}",
            megatron_path=args.megatron_path,
        )


def execute(args: ScriptArgs):
    is_debug_mode = args.mode != "normal"
    is_debug_one_sample = args.mode == "debug_one_sample"
    model_parallel_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    actor_num_gpus_per_node = model_parallel_size
    train_world_size = args.num_nodes * actor_num_gpus_per_node
    data_parallel_size = max(1, train_world_size // model_parallel_size)
    debug_num_rollout = max(2, data_parallel_size)
    debug_global_batch_size = data_parallel_size
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    megatron_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}{'-FP8' if args.rollout_fp8 else ''} "
        f"--save {load_save_path} "
        f"--save-interval {2 if is_debug_mode else 20} "
    )
    if not args.enable_megatron_bridge:
        ckpt_args += f"--load {megatron_load_path} "
    if args.use_kl_loss:
        ref_load_path = f"{args.model_dir}/{args.model_name}"
        if args.train_backend == "megatron":
            ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
        ckpt_args += f"--ref-load {ref_load_path} "

    if args.train_backend == "megatron":
        ckpt_args += f"--save-retain-interval {2 if is_debug_mode else 20} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # By default it is thinking mode
        # """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {debug_num_rollout if is_debug_one_sample else 3000} "
        f"--rollout-batch-size {1 if is_debug_one_sample else 32} "
        f"--n-samples-per-prompt {1 if is_debug_one_sample else 8} "
        f"--rollout-max-response-len {2 if is_debug_one_sample else (100 if args.mode == 'debug_minimal' else 8192)} "
        "--rollout-temperature 1 "
        f"--global-batch-size {debug_global_batch_size if is_debug_one_sample else 256} "
        "--balance-data "
    )

    if args.dynamic_sampling and not is_debug_mode:
        rollout_args += (
            "--over-sampling-batch-size 64 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    # sometimes disable eval to speed up debugging
    eval_args = ""
    if (not is_debug_mode) and args.enable_eval:
        eval_max_response_len = 16384
        eval_args += "--eval-interval 20 "
        if args.multi_eval:
            eval_config_text = f"""
eval:
  defaults:
    max_response_len: {eval_max_response_len}
    top_p: 0.7
  datasets:
    - name: aime
      path: {args.data_dir}/aime-2024/aime-2024.jsonl
      rm_type: math
      n_samples_per_eval_prompt: 16
    - name: gpqa
      path: {args.data_dir}/gpqa_diamond/gpqa_eval.jsonl
      rm_type: gpqa
      n_samples_per_eval_prompt: 2
    - name: ifbench
      path: {args.data_dir}/IFBench/IFBench_eval.jsonl
      rm_type: ifbench
      n_samples_per_eval_prompt: 1
""".strip()
            eval_args += f"--eval-config {U.save_to_temp_file(eval_config_text, 'yaml')} "
        else:
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 16 "
                f"--eval-max-response-len {eval_max_response_len} "
                "--eval-top-p 1 "
            )

    grpo_args = "--advantage-estimator grpo " "--entropy-coef 0.00 " "--eps-clip 0.2 " "--eps-clip-high 0.28 "
    if args.use_kl_loss:
        grpo_args += "--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "

    optimizer_args = (
        "--optimizer adam "
        # "--fsdp-cpu-offload "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-chunked-prefill-size 4096 "
        f"{'--sglang-disable-cuda-graph ' if is_debug_one_sample else ''}"
    )
    ci_args = "--ci-test --ci-disable-kl-checker " if is_debug_one_sample else ""

    match args.train_backend:
        case "fsdp":
            train_backend_args = (
                "--train-backend fsdp "
                "--attn-implementation flash_attention_2 "
                "--gradient-checkpointing "
                f"--update-weight-buffer-size {512 * 1024 * 1024} "  # 512MB
                """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
            )
            sglang_args += "--sglang-mem-fraction-static 0.75 "
            perf_args = f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "

        case "megatron":
            train_backend_args = (
                f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
                f"{'--sequence-parallel ' if args.use_sequence_parallel and args.tensor_model_parallel_size > 1 else ''}"
                f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
                f"--context-parallel-size {args.context_parallel_size} "
                f"{f'--cp-comm-type {args.cp_comm_type} ' if args.cp_comm_type is not None else ''}"
                "--expert-model-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
                "--recompute-granularity full "
                "--recompute-method uniform "
                "--recompute-num-layers 1 "
                # default dropout in megatron is 0.1
                "--attention-dropout 0.0 "
                "--hidden-dropout 0.0 "
                # should be good for model performance
                "--accumulate-allreduce-grads-in-fp32 "
                "--attention-softmax-in-fp32 "
                # need to comment this when using model with MLA
                "--attention-backend flash "
                f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
            )
            # TODO improve
            sglang_args += "--sglang-mem-fraction-static 0.7 "
            perf_args = f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "

        case _:
            raise NotImplementedError

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"{'--use-fault-tolerance ' if not is_debug_mode else ''}"
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.model_name == "Qwen3-4B-Base":
        misc_args += "--sglang-context-length 36000 "
        misc_env_vars |= {
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }

    if args.train_fp8:
        misc_args += (
            "--transformer-impl transformer_engine "
            "--bf16 "
            "--fp8-format e4m3 "
            "--fp8-recipe blockwise "
            "--fp8-param-gather "
        )
        misc_env_vars |= {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        }

    if args.enable_megatron_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

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

    true_on_policy_plan = build_true_on_policy_launch_plan(args)
    true_on_policy_args = true_on_policy_plan.train_args
    true_on_policy_envs = true_on_policy_plan.env_vars

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{ci_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        # TODO may get it from `config`
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            **misc_env_vars,
            **true_on_policy_envs,
        },
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
