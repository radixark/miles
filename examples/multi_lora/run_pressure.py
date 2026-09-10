"""Serve Qwen3-30B-A3B or GLM-5.2 for the multi-LoRA DAPO pressure test.

Requires a shared BF16 checkpoint, matching Miles/SGLang sources and an existing
four-node Ray cluster. The default split is 16 training and 16 rollout GPUs.

Args:
    model_name: Model recipe; Qwen matches the multi-LoRA gateway example.
    hf_checkpoint: Full BF16 Hugging Face checkpoint directory.
    n_adapters: ``auto`` runs the real GPU probe; an integer skips it.
    output_dir: Shared directory for probe results, checkpoints and sampler exports.
    sglang_pythonpath: Optional checkout's ``python`` directory.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/multi_lora/run_pressure.py \
        --model-name qwen3-30B-A3B --model-dir /models --output-dir /shared/pressure --n-adapters auto
"""

import shlex
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass(frozen=True)
class _Recipe:
    checkpoint_name: str
    model_type: str
    training_tp: int
    training_ep: int
    rollout_tp: int
    dsa: bool = False


_RECIPES = {
    "qwen3-30B-A3B": _Recipe("Qwen3-30B-A3B", "qwen3-30B-A3B", 2, 8, 2),
    "glm5.2": _Recipe("GLM-5.2", "glm5.2-744B-A40B_lora", 16, 16, 16, dsa=True),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_name: Literal["qwen3-30B-A3B", "glm5.2"] = "qwen3-30B-A3B"
    model_dir: str = "/root/models"
    hf_checkpoint: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    sglang_pythonpath: str = ""
    n_adapters: str = "auto"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    training_tp: int | None = None
    training_ep: int | None = None
    rollout_num_gpus: int = 16
    rollout_tp: int | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    context_length: int = 8192
    dsa_backend: str = "megatron"
    train_memory_margin_bytes: int = 2 * 1024**3
    engine_host_lora_budget_bytes: int = 0
    tinker_port: int = 10639
    sglang_mem_fraction_static: float = 0.8
    max_running_requests: int = 64
    extra_args: str = ""

    def __post_init__(self):
        recipe = _RECIPES[self.model_name]
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{recipe.checkpoint_name}"
        for name in ("training_tp", "training_ep", "rollout_tp"):
            if getattr(self, name) is None:
                setattr(self, name, getattr(recipe, name))
        if self.n_adapters != "auto" and int(self.n_adapters) < 1:
            raise ValueError("n_adapters must be auto or a positive count")
        training_gpus = self.actor_num_nodes * self.num_gpus_per_node
        if training_gpus % self.training_tp or training_gpus % self.training_ep:
            raise ValueError("training GPU count must be divisible by TP and EP")


def _model_extras(args):
    if not _RECIPES[args.model_name].dsa:
        return "linear_qkv,linear_proj,linear_fc1,linear_fc2", "", "", {}
    training = f"--dsa-attention-backend {args.dsa_backend} "
    if args.dsa_backend == "megatron":
        training += "--custom-megatron-init-path examples.multi_lora.glm52_native_dsa.install "
    return (
        "linear_qkv,linear_proj,linear_fc1,linear_fc2,linear_q_down_proj,linear_kv_down_proj,linear_q_up_proj,linear_kv_up_proj",
        training,
        "--sglang-attention-backend nsa --sglang-dsa-decode-backend flashmla_sparse "
        "--sglang-dsa-prefill-backend flashmla_sparse --sglang-page-size 64 --sglang-disable-shared-experts-fusion ",
        {"INDEXER_ROPE_NEOX_STYLE": "0", "SGLANG_NSA_FORCE_MLA": "1"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    targets, model_training_args, model_sglang_args, env = _model_extras(args)
    checkpoint_args = f"--hf-checkpoint {shlex.quote(args.hf_checkpoint)} --megatron-to-hf-mode bridge --save {shlex.quote(args.output_dir + '/' + args.run_id + '/training')} --save-interval 100000 --tinker-checkpoint-root {shlex.quote(args.output_dir + '/' + args.run_id)} "
    lora_args = (
        f"--multi-lora-n-adapters {args.n_adapters} --lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} --lora-dropout 0 --no-gradient-accumulation-fusion "
        f"--target-modules {targets} "
        f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
    )
    if args.engine_host_lora_budget_bytes:
        lora_args += f"--engine-host-lora-budget-bytes {args.engine_host_lora_budget_bytes} "
    perf_args = (
        f"--train-backend megatron {model_training_args} --qkv-format thd "
        f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--tensor-model-parallel-size {args.training_tp} --expert-model-parallel-size {args.training_ep} "
        "--expert-tensor-parallel-size 1 --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--seq-length {args.context_length} --rollout-max-context-len {args.context_length} "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.context_length} "
        f"--micro-batch-size 1 --global-batch-size {args.actor_num_nodes * args.num_gpus_per_node // args.training_tp} "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
    )
    optimizer_args = "--optimizer adam --lr 1e-5 "
    sglang_args = (
        f"--rollout-num-gpus {args.rollout_num_gpus} --rollout-num-gpus-per-engine {args.rollout_tp} "
        f"--sglang-ep-size {args.rollout_tp} --sglang-lora-backend triton --sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-context-length {args.context_length} --sglang-max-running-requests {args.max_running_requests} "
        f"--sglang-chunked-prefill-size {args.context_length} --sglang-cuda-graph-max-bs-decode 16 "
        f"--sglang-watchdog-timeout 3600 --sglang-moe-runner-backend triton {model_sglang_args} "
    )
    if args.n_adapters != "auto":
        sglang_args += f"--sglang-max-loaded-loras {2 * int(args.n_adapters)} "
    env["RAY_DEDUP_LOGS"] = "0"
    if args.sglang_pythonpath:
        env["PYTHONPATH"] = args.sglang_pythonpath
    # Client processes own per-LoRA W&B runs. Do not put their credentials in
    # the Ray command line or the server's argument dumps.
    U.execute_train(
        train_args=(
            f"{checkpoint_args} {lora_args} {perf_args} {optimizer_args} {sglang_args} --tinker-server-port {args.tinker_port} {args.extra_args}"
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_RECIPES[args.model_name].model_type,
        train_script="serve_tinker.py",
        config=args,
        megatron_path=args.megatron_path,
        extra_env_vars=env,
        cleanup_processes=False,  # the supervisor reaps only the preceding trial's processes
    )


if __name__ == "__main__":
    typer.run(main)
