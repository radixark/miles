"""
GLM-5 744B-A40B Training Script

=====================

Args:
  --model-name: Model variant to use.
      GLM-5         Full 744B model (requires >=16 nodes)
      GLM-5_4layer  4-layer pruned model (single-node testing)
      GLM-5_20layer 20-layer pruned model (multi-node testing)
  --num-nodes: Number of nodes for training. Determines parallelism config:
      1  -> for GLM-5_4layer minimal test
      6  -> for GLM-5_20layer multi-node test
      16+-> for full GLM-5 model
  --num-gpus-per-node: GPUs per node (default: 8)
  --mode: "normal" or "debug_minimal" (shorter response length for quick testing)
  --fp8-rollout: Enable FP8 rollout (converts HF checkpoint to FP8 block quant for sglang; megatron still uses bf16)
  --enable-eval: Enable evaluation every 20 steps
  --enable-mtp: Enable multi-token prediction (EAGLE speculative decoding)
  --enable-optimizer-offload: Offload optimizer to CPU
  --data-dir: Directory for datasets (default: /root/datasets, shared NFS)
  --model-dir: Directory for model weights and converted checkpoints (default: /root/models, shared NFS)
  --model-local-dir: Node-local directory for model copies (default: /root/local_data, local disk)

=====================

I. Usage for single node minimal test:
  `ray stop --force && pkill -9 -f sglang || true && sleep 3 && ray start --head --port=6378 --dashboard-port=8266`
  `python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5_4layer --num-nodes 1`

=====================

II. Usage for multi node (20 layers, 6 nodes as an example):

  1. Setup containers on all nodes

  2. Start Ray cluster on all nodes

  3. Download model/data + patch checkpoint + convert to megatron.
     Run on **head node**; megatron conversion uses Ray to coordinate multi-node work.
       `python scripts/run_glm5_744b_a40b.py prepare --model-name GLM-5_20layer --num-nodes 6`

  4. (Optional) Copy model from shared NFS to local disk on each node.
     Run independently on every node.
       python scripts/run_glm5_744b_a40b.py prepare-cp --model-name GLM-5_20layer --num-nodes 6

  5. Run training. Execute on head node; uses Ray internally for distributed training.
       python scripts/run_glm5_744b_a40b.py train --model-name GLM-5_20layer --num-nodes 6
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-5"
    megatron_model_type: str = "glm5-744B-A40B"
    num_gpus_per_node: int = 8
    fp8_rollout: bool = False
    enable_eval: bool = False
    enable_mtp: bool = False
    enable_pd: bool = True
    enable_optimizer_offload: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/local_data"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        if self.num_nodes == 1:
            self.enable_pd = False
            self.mode = "debug_minimal"

        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"glm5-744B-A40B_{m.group(1)}layer"

        if self.model_name == "GLM-5":
            self.model_org = "zai-org"
        elif self.model_name in ["GLM-5_4layer", "GLM-5_20layer"]:
            self.model_org = "Pinaster"
        else:
            raise NotImplementedError(f"{self.model_name} is not supported")


def _is_pruned(args: ScriptArgs):
    return re.search(r"(\d+)layer", args.model_name) is not None


def _process_glm_checkpoint(args: ScriptArgs):
    """Patch config.json to use DeepseekV32 architecture if not already patched."""
    config_path = Path(args.model_dir) / args.model_name / "config.json"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, skipping checkpoint processing")
        return

    with open(config_path) as f:
        config = json.load(f)

    if config.get("model_type") == "deepseek_v32":
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


def _convert_to_fp8(args: ScriptArgs):
    """Convert HF checkpoint to FP8 (block quantization). Megatron still uses bf16."""
    src = f"{args.model_dir}/{args.model_name}"
    dst = f"{args.model_dir}/{args.model_name}_fp8"
    U.exec_command(
        f"python tools/convert_hf_to_fp8.py "
        f"--model-dir {src} --save-dir {dst} "
        f"--strategy block --block-size 128 128"
    )


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Skip model download for pruned variants (assumed to already exist in model_dir)
    U.exec_command(
        f"huggingface-cli download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_megatron_ckpt(args: ScriptArgs):
    extra_args = "--tensor-model-parallel-size 1 " "--expert-tensor-parallel-size 1 "
    num_gpus_per_node = args.num_gpus_per_node
    multinode = True
    num_nodes = None

    num_layers_match = re.search(r"(\d+)layer", args.model_name)
    if num_layers_match and int(num_layers_match.group(1)) <= 4:
        extra_args += "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        num_gpus_per_node = min(4, num_gpus_per_node)
        multinode = False
    elif num_layers_match:
        extra_args += "--expert-model-parallel-size 4 "
        num_nodes = 2
    else:
        extra_args += (
            "--pipeline-model-parallel-size 4 "
            "--expert-model-parallel-size 32 "
            "--decoder-last-pipeline-num-layers 18 "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        megatron_path=args.megatron_path,
    )


def _prepare_cp(args: ScriptArgs, skip_existing: bool = False):
    torch_dist_dst = f"{args.model_local_dir}/{args.model_name}_torch_dist"
    if not (skip_existing and Path(torch_dist_dst).exists()):
        U.rsync_simple(
            path_src=f"{args.model_dir}/{args.model_name}_torch_dist",
            path_dst=torch_dist_dst,
        )
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    hf_dst = f"{args.model_local_dir}/{hf_name}"
    if not (skip_existing and Path(hf_dst).exists()):
        U.rsync_simple(
            path_src=f"{args.model_dir}/{hf_name}",
            path_dst=hf_dst,
        )


def _execute_train(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{hf_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 1 "

    if args.num_nodes == 1:  # minimal test for 4 layers model
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes == 6:  # for 20 layers model, to test multi-node
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 3 "
            "--decoder-last-pipeline-num-layers 6 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes >= 16:  # slime's setting for full model
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 4 "
            "--decoder-last-pipeline-num-layers 18 "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 32 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        raise NotImplementedError

    perf_args += (
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if _is_pruned(args) else 16384} "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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
    if args.enable_optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    if args.enable_pd:
        sglang_decode_max_bs = 8
        if args.num_nodes < 16:
            sglang_world_size = 16
        else:
            sglang_world_size = 64

    else:
        sglang_decode_max_bs = 256
        sglang_world_size = 8

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.70 "
        "--sglang-enable-dp-attention "
        f"--sglang-ep-size {sglang_world_size} "
        f"--sglang-dp-size {sglang_world_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
    )
    if args.fp8_rollout:
        sglang_args += "--sglang-moe-a2a-backend deepep " "--sglang-deepep-mode auto "
    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
    if args.enable_pd:
        sglang_args += "--prefill-num-servers 1 "
    sglang_args += (
        # dsa
        "--sglang-page-size 64 "
        "--sglang-nsa-decode-backend flashmla_sparse "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
        # concurrency
        f"--sglang-max-running-requests 512 "
        f"--sglang-chunked-prefill-size {2048 * sglang_world_size} "
        "--sglang-watchdog-timeout 3600 "
    )
    sglang_extra_env_vars = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": f"{32 if args.enable_pd else 256}",
    }

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        # use deepep for megatron
        "--moe-enable-deepep "
        "--moe-token-dispatcher-type flex "
        "--allgather-cp "
        # ------------
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
        "--disable-weights-backuper "
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
        extra_env_vars={
            **sglang_extra_env_vars,
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Full pipeline: download, convert, copy, train."""
    _prepare_download(args)
    _process_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)
    _prepare_cp(args, skip_existing=True)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download model/data and convert to megatron checkpoint (run on head node)."""
    _prepare_download(args)
    _process_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    """Copy model to local storage (run on all nodes via run_spmd)."""
    _prepare_cp(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
