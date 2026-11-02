"""GLM-5.3-Flash GRPO LoRA training through native LoRA (raw mode).

The model is miles' own glm5_next plugin, the one `scripts/run_glm5_3_flash.py` trains in full;
`miles.backends.megatron_utils.lora.native` attaches the adapters. They cover the KDA projections, the
DSA layers' MLA projections except `kv_b_proj`, and the dense and shared-expert MLPs. Routed experts and
`kv_b_proj` stay frozen: native LoRA has no per-expert adapter, and the DSA attention absorbs
`kv_b_proj`'s weight without calling the module.

python scripts/run_glm5_3_flash_lora.py prepare --model-name GLM-5.3-Flash-4layer
python scripts/run_glm5_3_flash_lora.py train --model-name GLM-5.3-Flash-4layer --num-nodes 1 --num-gpus-per-node 8
python scripts/run_glm5_3_flash_lora.py train --model-name GLM-5.3-Flash --num-nodes 3 --num-gpus-per-node 8
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer
from scripts.run_glm5_3_flash import _MODEL_REGISTRY

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = {
    "GLM-5.3-Flash": "zai-org/GLM-5.3-Flash",
    "GLM-5.3-Flash-4layer": "CharyZeng/GLM-5.3-Flash-4layer",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: Literal["GLM-5.3-Flash", "GLM-5.3-Flash-4layer"] = "GLM-5.3-Flash-4layer"
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    run_id: str = U.create_run_id()
    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    save_dir: str = "/root/shared_data"
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 5
    rollout_max_response_len: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: str = "attn,mlp"
    exclude_modules: str = "*.mlp.experts.*,*.self_attn.kv_b_proj"
    enable_r3: bool = False
    skip_saving: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"

    @property
    def megatron_model_type(self) -> str:
        return _MODEL_REGISTRY[self.model_name]


def _prepare(args: ScriptArgs):
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.ckpt_dir} {args.data_dir}")
    backend.exec_command_cpu(f"hf download {_HF_REPO[args.model_name]} --local-dir {args.hf_checkpoint}")
    if args.model_name == "GLM-5.3-Flash-4layer":
        # the slice omits the processor config SGLang loads for this multimodal model
        backend.exec_command_cpu(
            f"hf download {_HF_REPO['GLM-5.3-Flash']} processor_config.json --local-dir {args.hf_checkpoint}"
        )
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    os.environ["CONVERT_KEEP_PP1"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    backend.convert_checkpoint(
        model_name=args.megatron_model_type,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        # unsplit experts leave each rank of the full model with half of its 300B weights
        extra_args=(
            "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} --expert-tensor-parallel-size 1"
        ),
    )


def _train(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ckpt_dir}/{args.megatron_model_type}_torch_dist "
        "--megatron-to-hf-mode raw "
    )
    if not args.skip_saving:
        load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
        ckpt_args += f"--load {load_save_path} --save {load_save_path} --save-interval 10 "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--target-modules "{args.target_modules}" --exclude-modules "{args.exclude_modules}" '
        "--no-gradient-accumulation-fusion --lora-base-cpu-backup --check-lora-weight-equal "
    )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )

    num_gpus = args.num_nodes * args.num_gpus_per_node
    # the frozen base needs no optimizer state, so experts spread over every GPU without PP
    assert 288 % num_gpus == 0, f"expert parallelism over all {num_gpus} GPUs must divide the 288 routed experts"
    if args.model_name == "GLM-5.3-Flash":
        tensor_parallel_size, engine_gpus = 8, 8
    else:
        tensor_parallel_size, engine_gpus = 2, 4
    parallel_args = (
        f"--tensor-model-parallel-size {tensor_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {num_gpus} "
        "--expert-tensor-parallel-size 1 "
    )
    engine_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} --sglang-tp-size {engine_gpus} --sglang-ep-size {engine_gpus} "
    )

    perf_args = (
        f"{parallel_args}"
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
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"{engine_args}"
        # routing replay needs materialized topk ids; the SM100 default resolves to a fused runner
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-disable-radix-cache "
        "--sglang-dsa-prefill-backend tilelang "
        "--sglang-dsa-decode-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        "--sglang-lora-backend triton "
        # the shared-expert adapter needs the shared expert outside the fused routed MoE
        "--sglang-disable-shared-experts-fusion "
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
        "--offload-train-target cpu "
        "--sglang-mem-fraction-static 0.5 "
        "--colocate "
        "--model-name glm5_next "
        "--qkv-format thd "
        "--rollout-health-check-interval 300 "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} "
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
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the checkpoint and dataset, and build the raw-mode torch_dist base."""
    _prepare(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    app()
