"""MiniMax-M3.1-Flash DAPO training (bf16 train, bf16 rollout).

The official checkpoint (MXFP8 linears, NVFP4 routed experts) cannot be trained directly, so `prepare`
downloads MiniMaxAI/MiniMax-M3.1-Flash to `<model-dir>/MiniMax-M3.1-Flash`, converts it with
tools/convert_minimax_m3_to_bf16.py into `<model-dir>/<model-name>` and that into the Megatron
`<ckpt-dir>/<megatron-model-type>_torch_dist` used as the reference model, unless they already exist:
  MiniMax-M3.1-Flash-bf16         the full model (852 GB), 8 nodes x 4 GPUs
  MiniMax-M3.1-Flash-4layer-bf16  `--layers 0,1,3,4` (36 GB), two dense and two MoE layers, 4 GPUs

Args:
    --model-name: checkpoint directory under `--model-dir`; selects the matching recipe.
    --rollout-max-response-len: per-reply generation budget.
    --check-weight-update-equal: bitwise weight-sync check; defaults on for the 4-layer model only.
    --enable-r3: rollout routing replay for the MoE router.
    --train-offload-dir: NVMe directory for the actor offloaded while the engines generate and the
        streamed Adam state.

The full model expects MILES_SCRIPT_EXTERNAL_RAY=1 with all 8 nodes already joined to the ray cluster.

python scripts/run_minimax_m3_1_flash.py --model-name MiniMax-M3.1-Flash-bf16 --num-nodes 8
python scripts/run_minimax_m3_1_flash.py --model-name MiniMax-M3.1-Flash-4layer-bf16 --num-nodes 1
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_HF_REPO = "MiniMaxAI/MiniMax-M3.1-Flash"


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    num_gpus: int
    # TP / PP / EP of the trainer.
    parallel: tuple[int, int, int]
    rollout_num_gpus_per_engine: int
    # Source decoder layers kept by tools/convert_minimax_m3_to_bf16.py; None keeps all.
    layers: str | None = None
    # ~13B routed-expert params per GPU at EP8/PP4: fp32 master + Adam fit neither the GPU nor host RAM.
    stream_optimizer_state: bool = False
    # The weight check snapshots every weight on the host; for the full model that exceeds host RAM.
    check_weight_update_equal: bool = True
    # A single node runs out of host memory saving the full model, so its conversion is sharded.
    convert_args: str = ""


_RECIPES = {
    "MiniMax-M3.1-Flash-bf16": _Recipe(
        megatron_model_type="minimax-m3.1-flash",
        num_gpus=32,
        parallel=(4, 4, 8),
        rollout_num_gpus_per_engine=8,
        stream_optimizer_state=True,
        check_weight_update_equal=False,
        convert_args="--pipeline-model-parallel-size 4 --expert-model-parallel-size 8",
    ),
    "MiniMax-M3.1-Flash-4layer-bf16": _Recipe(
        megatron_model_type="minimax-m3.1-flash-4layer",
        num_gpus=4,
        parallel=(2, 1, 2),
        rollout_num_gpus_per_engine=4,
        layers="0,1,3,4",
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: Literal["MiniMax-M3.1-Flash-bf16", "MiniMax-M3.1-Flash-4layer-bf16"] = "MiniMax-M3.1-Flash-4layer-bf16"
    num_gpus_per_node: int = 4
    run_id: str = U.create_run_id()
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    train_offload_dir: str = "/root/train_offload"
    num_rollout: int = 5
    rollout_max_response_len: int = 4096
    check_weight_update_equal: bool | None = None
    enable_r3: bool = True
    save: bool = False
    extra_args: str = ""

    def __post_init__(self):
        if self.check_weight_update_equal is None:
            self.check_weight_update_equal = _RECIPES[self.model_name].check_weight_update_equal


def prepare(args: ScriptArgs):
    backend = args.create_backend()
    recipe = _RECIPES[args.model_name]
    target = Path(args.model_dir) / args.model_name
    with U.exclusive_path_lock(str(target)):
        # The converter writes the index last, so it marks a finished conversion.
        if not (target / "model.safetensors.index.json").exists():
            source = f"{args.model_dir}/{_HF_REPO.split('/')[1]}"
            layer_args = f"--layers {recipe.layers}" if recipe.layers else ""
            backend.exec_command_cpu(f"mkdir -p {args.model_dir}")
            backend.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {source}")
            backend.exec_command_gpu(
                f"python {U.repo_base_dir}/tools/convert_minimax_m3_to_bf16.py "
                f"--model-dir {source} --save-dir {target} --device cuda {layer_args}"
            )
    backend.convert_checkpoint(
        model_name=recipe.megatron_model_type,
        megatron_model_type=recipe.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=args.num_nodes > 1,
        num_nodes=args.num_nodes,
        extra_args=recipe.convert_args,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=str(target),
        megatron_path=args.megatron_path,
    )
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    recipe = _RECIPES[args.model_name]
    num_gpus = args.num_nodes * args.num_gpus_per_node
    assert num_gpus == recipe.num_gpus, f"{args.model_name} is laid out for {recipe.num_gpus} GPUs, got {num_gpus}"
    tp, pp, ep = recipe.parallel

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.ckpt_dir}/{recipe.megatron_model_type}_torch_dist "
    )
    if args.save:
        save_dir = f"{args.output_dir}/{args.run_id}/checkpoints"
        ckpt_args += (
            f"--load {save_dir} --save {save_dir} --save-interval 10 "
            "--no-save-optim --no-save-rng --no-load-optim --no-load-rng "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 16 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pp} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 8192 "
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
        "--lr 3e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if recipe.stream_optimizer_state:
        optimizer_args += "--stream-optimizer-state-to-disk --stream-optimizer-state-moment-dtype bf16 "

    engine_gpus = recipe.rollout_num_gpus_per_engine
    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} --sglang-tp-size {engine_gpus} --sglang-ep-size {engine_gpus} "
        # routing replay needs materialized topk ids
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-disable-radix-cache "
        "--sglang-disable-prefill-cuda-graph "
        "--sglang-kv-cache-dtype bfloat16 "
        "--sglang-mem-fraction-static 0.7 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--offload-train-target disk "
        f"--offload-train-disk-dir {args.train_offload_dir} "
        "--colocate "
        "--model-name minimax_m3 "
        "--qkv-format thd "
        "--rollout-health-check-interval 300 "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal --check-weight-update-skip-list vision_tower. multi_modal_projector. patch_merge_mlp. "
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # inductor's subprocess compile pool deadlocks under the torch_memory_saver LD_PRELOAD
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    # ray's runtime_env replaces the actor env; without the JIT cache dirs the kernels recompile every run
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        cache_dir = os.environ.get(cache_var)
        if cache_dir:
            extra_env_vars[cache_var] = cache_dir

    args.create_backend().execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
