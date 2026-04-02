import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zianglih"
    model_name: str = "DeepSeek-V3.2-6layer"
    megatron_model_type: str = "deepseek-v32-6layer"
    num_gpus_per_node: int | None = 8
    actor_num_gpus_per_node: int | None = 4
    rollout_num_gpus: int | None = 4
    hardware: Literal["B200", "B300", "GB200", "GB300"] = "B200"
    enable_eval: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_mxfp8: bool = False
    train_mxfp8: bool = False
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0
    enable_mis: bool = False
    tis_use_rs: bool = True

    def __post_init__(self):
        pass


def _process_deepseek_v32_checkpoint(args: ScriptArgs):
    """Patch checkpoint config so HF auto classes resolve to DeepSeek-v3.2."""
    config_path = Path(args.model_dir) / args.model_name / "config.json"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, skipping checkpoint processing")
        return

    with open(config_path) as f:
        config = json.load(f)

    if (
        config.get("model_type") == "deepseek_v32"
        and config.get("architectures") == ["DeepseekV32ForCausalLM"]
        and isinstance(config.get("auto_map"), dict)
        and config["auto_map"].get("AutoConfig") == "configuration_deepseek_v32.DeepseekV32Config"
        and config["auto_map"].get("AutoModelForCausalLM") == "modeling_deepseek_v32.DeepseekV32ForCausalLM"
    ):
        print("Checkpoint already patched, skipping")
        return

    config["architectures"] = ["DeepseekV32ForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_deepseek_v32.DeepseekV32Config",
        "AutoModelForCausalLM": "modeling_deepseek_v32.DeepseekV32ForCausalLM",
    }
    config["model_type"] = "deepseek_v32"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Patched {config_path}")


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(
        f"huggingface-cli download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    _process_deepseek_v32_checkpoint(args)

    if args.rollout_mxfp8:
        U.exec_command(
            f"python tools/convert_hf_to_mxfp8.py --model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-MXFP8 "
            f"--num-layers-at-start-in-bf16 {args.num_layers_at_start_in_bf16} "
            f"--num-layers-at-end-in-bf16 {args.num_layers_at_end_in_bf16} "
            f"{args.extra_args} "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=4,
        # To support multi-node training, for simplicity, we put model into shared folder
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.rollout_mxfp8 or args.train_mxfp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-MXFP8"
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
        "--num-rollout 3000 "
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
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {
        "SGLANG_NSA_FORCE_MLA": "1",
        "INDEXER_ROPE_NEOX_STYLE": "0",
    }

    if args.train_mxfp8:
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

    if args.train_mxfp8 and (args.num_layers_at_start_in_bf16 > 0 or args.num_layers_at_end_in_bf16 > 0):
        misc_args += (
            "--first-last-layers-bf16 "
            f"--num-layers-at-start-in-bf16 {args.num_layers_at_start_in_bf16} "
            f"--num-layers-at-end-in-bf16 {args.num_layers_at_end_in_bf16} "
        )

    match args.hardware:
        case "B200" | "B300" | "GB200" | "GB300":
            perf_args += (
                f"--tensor-model-parallel-size {args.actor_num_gpus_per_node} "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                f"--expert-model-parallel-size {args.actor_num_gpus_per_node} "
                "--expert-tensor-parallel-size 1 "
            )

            sglang_args = (
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-attention-backend nsa "
                "--sglang-nsa-decode-backend flashmla_sparse "
                "--sglang-nsa-prefill-backend flashmla_sparse "
            )

            if args.rollout_mxfp8:
                sglang_world_size = 1
                sglang_attn_tp_size = 1
                sglang_decode_max_bs = 256
                sglang_args += (
                    "--sglang-enable-dp-attention "
                    f"--rollout-num-gpus-per-engine 1 "
                    "--sglang-fp8-gemm-backend flashinfer_trtllm "
                    "--sglang-moe-runner-backend flashinfer_trtllm_routed "
                    f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
                    "--sglang-moe-dense-tp-size 1 "
                )
                misc_args += (
                    "--extra-high-precision-layers .kv_b_proj. "
                    "--extra-high-precision-layers-megatron .linear_kv_up_proj .linear_k_up_proj .linear_v_up_proj "
                )
                optimizer_args += (
                    "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
                )
                misc_env_vars["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "256"
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
                misc_args += f"--te-precision-config-file {U.save_to_temp_file(te_precision_config_text, 'yaml')} "
            else:
                sglang_args += "--rollout-num-gpus-per-engine 1 " "--sglang-cuda-graph-max-bs 256 "
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


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
