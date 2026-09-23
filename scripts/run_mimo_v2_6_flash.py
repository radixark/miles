"""MiMo-V2.6-Flash-RL training (RL with GRPO, or SFT), adapted from run_gpt_oss_20b.py.

Megatron bridge mode on a BF16 checkpoint converted with tools/convert_mimo_v2_to_bf16.py; the
official checkpoint (fused kv-interleaved FP8 qkv, MXFP4 experts) cannot be read directly. The
bridge (miles_plugins/megatron_bridge/mimo_v2.py) builds the per-layer SWA/global attention, the SWA
sink and the 192/128 Q-K/V head dims, so there is no `torch_dist` conversion and no `--ref-load`
(bridge mode has no Megatron reference checkpoint, hence no KL loss, as for gpt-oss).

`prepare` writes `<model-dir>/<model-name>` unless it already exists: it downloads the official
XiaomiMiMo/MiMo-V2.6-Flash-RL (173 GB) to `<model-dir>/MiMo-V2.6-Flash-RL` and converts it on the GPU:
  MiMo-V2.6-Flash-RL-bf16  the full model (622 GB)
  mimo26-p4-bf16           `--layers 0,1,5,6` (46 GB), a 4-layer partial with one instance of every
                           decoder variant: global+dense, SWA+MoE, global+MoE, SWA+MoE

Args:
  --mode: `rl` runs GRPO with a colocated BF16 SGLang engine on the same checkpoint (dapo-math-17k
      from `--data-dir` unless `--prompt-data` is given); `sft` runs `--debug-train-only` SFT on
      chat data with a `messages` column (no SGLang), following run_qwen3_sft.py.
  --model-name: checkpoint directory under `--model-dir`; selects the matching model args.
  --tensor/pipeline/expert-model-parallel-size: TP (with sequence parallel when > 1), PP and EP;
      default 2/2/2 for the partial and 2/2/8 for the full model (16 GPUs). TP must not exceed 4,
      the global-attention KV head count. Context parallel is not supported.
  --recompute / --no-recompute: full uniform recompute, one layer per checkpoint.
  --async-train / --no-async-train: SFT only; `train_async.py` prefetches the next batch, so a run
      resumed from its checkpoint skips one batch, while `train.py` resumes at the next batch.
  --qkv-format: `thd` packs samples with dynamic batching (`--max-tokens-per-gpu`); `bshd` runs one
      unpacked sample per micro-batch.
  --num-rollout, --rollout-batch-size: steps and samples (prompts in RL) per step.
  --save / --save-interval / --load: Megatron checkpoints under `<output-dir>/checkpoints`.
  --train-offload-disk-dir: NVMe directory for the full model's streamed Adam state (bf16 moments,
      bf16 gradient reduction) and, in RL, the actor offloaded while the engines generate.
  --extra-args: appended verbatim to the train argv.

The full model needs two nodes: start the ray head on the first, join the second, and run with
`MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip> --num-nodes 2`.

Examples:
  python scripts/run_mimo_v2_6_flash.py --mode sft --model-name mimo26-p4-bf16 --prompt-data <jsonl>
  python scripts/run_mimo_v2_6_flash.py --mode rl --model-name mimo26-p4-bf16
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_HF_REPO = "XiaomiMiMo/MiMo-V2.6-Flash-RL"


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    # Source decoder layers kept by tools/convert_mimo_v2_to_bf16.py; None keeps all.
    layers: str | None = None
    # Default TP / PP / EP.
    parallel: tuple[int, int, int] = (2, 2, 2)
    # The BF16 engine holds the whole model: 620 GB needs TP8 on 141 GB GPUs.
    rollout_num_gpus_per_engine: int = 4
    sglang_mem_fraction_static: float = 0.6
    # Full-parameter Adam state of the full model (3.7 TB) fits neither 16 GPUs nor two hosts'
    # memory, so it streams through node-local NVMe; that needs bf16 gradient reduction.
    stream_optimizer_state: bool = False


_RECIPES = {
    "MiMo-V2.6-Flash-RL-bf16": _Recipe(
        megatron_model_type="mimo-v2.6-flash",
        parallel=(2, 2, 8),
        rollout_num_gpus_per_engine=8,
        sglang_mem_fraction_static=0.8,
        stream_optimizer_state=True,
    ),
    "mimo26-p4-bf16": _Recipe(megatron_model_type="mimo-v2.6-flash-4layer", layers="0,1,5,6"),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    mode: Literal["rl", "sft"] = "rl"
    model_name: Literal["MiMo-V2.6-Flash-RL-bf16", "mimo26-p4-bf16"] = "mimo26-p4-bf16"
    num_gpus_per_node: int = 8
    # None takes the recipe default of --model-name.
    tensor_model_parallel_size: int | None = None
    pipeline_model_parallel_size: int | None = None
    expert_model_parallel_size: int | None = None
    recompute: bool = True
    # SFT uses train_async.py like run_qwen3_sft.py; --no-async-train runs train.py, whose
    # checkpoints resume exactly at the next unconsumed batch.
    async_train: bool = True
    # thd packs samples with dynamic batching; bshd runs one unpacked sample per micro-batch.
    qkv_format: Literal["thd", "bshd"] = "thd"
    max_tokens_per_gpu: int = 9216
    num_rollout: int = 4
    rollout_batch_size: int = 16
    prompt_data: str = ""
    save: bool = False
    save_interval: int = 2
    load: str = ""
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    # NVMe directory for the streamed optimizer state and the offloaded actor (full model).
    train_offload_disk_dir: str = "/root/shared_data/train_offload"

    def __post_init__(self):
        recipe = _RECIPES[self.model_name]
        tp, pp, ep = recipe.parallel
        self.tensor_model_parallel_size = self.tensor_model_parallel_size or tp
        self.pipeline_model_parallel_size = self.pipeline_model_parallel_size or pp
        self.expert_model_parallel_size = self.expert_model_parallel_size or ep


def prepare(args: ScriptArgs):
    target = Path(args.model_dir) / args.model_name
    # The converter writes the index last, so it marks a finished conversion.
    if not (target / "model.safetensors.index.json").exists():
        source = f"{args.model_dir}/{_HF_REPO.split('/')[1]}"
        layers = _RECIPES[args.model_name].layers
        layer_args = f"--layers {layers}" if layers else ""
        U.exec_command_cpu(f"mkdir -p {args.model_dir}")
        U.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {source}")
        U.exec_command_gpu(
            f"python {U.repo_base_dir}/tools/convert_mimo_v2_to_bf16.py "
            f"--model-dir {source} --save-dir {target} --device cuda {layer_args}"
        )
    if args.mode == "rl" and not args.prompt_data:
        U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    ckpt_args = f"--hf-checkpoint {args.model_dir}/{args.model_name} " "--megatron-to-hf-mode bridge "
    if args.save:
        ckpt_args += f"--save {args.output_dir}/checkpoints " f"--save-interval {args.save_interval} "
    if args.load:
        ckpt_args += f"--load {args.load} "

    if args.mode == "sft":
        data_args = (
            "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
            f"--prompt-data {args.prompt_data} "
            "--input-key messages "
            "--rollout-shuffle "
            f"--num-rollout {args.num_rollout} "
            f"--rollout-batch-size {args.rollout_batch_size} "
            f"--global-batch-size {args.rollout_batch_size} "
            "--loss-type sft_loss "
            "--calculate-per-token-loss "
            "--disable-compute-advantages-and-returns "
            "--debug-train-only "
        )
    else:
        data_args = (
            f"--prompt-data {args.prompt_data or f'{args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl'} "
            "--input-key prompt "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            f"--num-rollout {args.num_rollout} "
            f"--rollout-batch-size {args.rollout_batch_size} "
            "--n-samples-per-prompt 8 "
            "--rollout-max-response-len 8192 "
            "--rollout-temperature 1.0 "
            "--rollout-top-p 0.95 "
            "--num-steps-per-rollout 1 "
            "--advantage-estimator grpo "
            "--entropy-coef 0.00 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )

    tp = args.tensor_model_parallel_size
    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        f"{'--sequence-parallel ' if tp > 1 else ''}"
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        "--expert-tensor-parallel-size 1 "
    )
    if args.qkv_format == "thd":
        perf_args += f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    else:
        perf_args += "--qkv-format bshd --micro-batch-size 1 "
    if args.recompute:
        perf_args += "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "

    recipe = _RECIPES[args.model_name]
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if recipe.stream_optimizer_state:
        optimizer_args += (
            "--stream-optimizer-state-to-disk "
            "--stream-optimizer-state-moment-dtype bf16 "
            "--grad-reduce-in-bf16 "
            f"--offload-train-disk-dir {args.train_offload_disk_dir} "
        )
        if args.mode == "rl":
            optimizer_args += "--offload-train-target disk "
    else:
        optimizer_args += "--accumulate-allreduce-grads-in-fp32 "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {recipe.rollout_num_gpus_per_engine} "
        "--sglang-dtype bfloat16 "
        f"--sglang-mem-fraction-static {recipe.sglang_mem_fraction_static} "
        "--sglang-decode-log-interval 1000 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend fused "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.mode == "rl":
        misc_args += "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{data_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args if args.mode == 'rl' else ''} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py" if args.mode == "sft" and args.async_train else "train.py",
        # Colocated RL offloads through torch_memory_saver, which rejects expandable segments.
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} if args.mode == "sft" else {},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
