"""Qwen3.8-Flash-Next DAPO on the torchtitan backend.

=====================

The rollout, algorithm and SGLang settings are `run_qwen3_8_next.py`'s, so the two
backends' curves read against each other; only the trainer differs. torchtitan loads
the HF directory as-is (no torch_dist conversion) through the miles model package
`miles.backends.torchtitan_utils.models.qwen3_8_next`.

The trainer is FSDP with expert parallelism over all trainer GPUs; tensor, context
and pipeline parallelism are not supported for this model. torchtitan has no CPU
optimizer, so the fp32 master weights and Adam state (16 bytes per trained
parameter) stay on the GPUs between rollouts: about 2 TB for the full model's 126 B
trained parameters, which needs 32 or more 141 GB GPUs. The frozen ~102 GB PLE table
is held once per node in /dev/shm, so the container needs a /dev/shm that large.

Multi-node runs expect an already-joined ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1).

=====================

Args:
  --model-name: Qwen3.8-Flash-Next-4layer (smoke slice, default) or Qwen3.8-Flash-Next (48 layers).
  --num-nodes / --num-gpus-per-node: trainer and colocated rollout GPUs.
  --num-rollout: Rollouts to train (default: 5).
  --enable-r3: Rollout routing replay.
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

Single-node smoke slice:

  python scripts/run_qwen3_8_next_torchtitan.py

Full model on four 8-GPU nodes:

  python scripts/run_qwen3_8_next_torchtitan.py --model-name Qwen3.8-Flash-Next --num-nodes 4
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass(frozen=True)
class _Recipe:
    hf_repo: str
    titan_model_flavor: str
    min_gpus: int
    rollout_num_gpus_per_engine: int


_RECIPES = {
    "Qwen3.8-Flash-Next": _Recipe("Qwen/Qwen3.8-Flash-Next", "Flash-Next", 32, 8),
    "Qwen3.8-Flash-Next-4layer": _Recipe("CharyZeng/Qwen3.8-Flash-Next-4layer", "Flash-Next-4layer", 8, 4),
}

NUM_EXPERTS = 512
RESPONSE_LEN = 4096
# Rotary tables are sized from this, so it has to cover the longest prompt plus the response.
SEQ_LEN = RESPONSE_LEN + 2048


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: Literal["Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-4layer"] = "Qwen3.8-Flash-Next-4layer"
    num_gpus_per_node: int = 8
    run_id: str = U.create_run_id()
    num_rollout: int = 5
    enable_r3: bool = False
    check_weight_update_equal: bool = True
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    extra_args: str = ""

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node

    def __post_init__(self):
        if self.num_gpus < self.recipe.min_gpus:
            raise ValueError(f"{self.model_name} needs at least {self.recipe.min_gpus} GPUs, got {self.num_gpus}")
        # expert parallelism spans every trainer GPU, so each must hold a whole number of experts
        if NUM_EXPERTS % self.num_gpus:
            raise ValueError(f"{NUM_EXPERTS} experts do not split evenly over {self.num_gpus} GPUs")


def prepare(args: ScriptArgs):
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    backend.exec_command_cpu(f"hf download {args.recipe.hf_repo} --local-dir {args.model_dir}/{args.model_name}")
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    ckpt_args = f"--hf-checkpoint {args.model_dir}/{args.model_name} "

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
        f"--rollout-max-response-len {RESPONSE_LEN} "
        '--apply-chat-template-kwargs \'{"thinking_mode":"thinking"}\' '
    )

    # FSDP over every trainer GPU, routed experts additionally split over all of them
    perf_args = (
        "--train-backend torchtitan "
        "--titan-model-name qwen3_8_next "
        f"--titan-model-flavor {args.recipe.titan_model_flavor} "
        f"--seq-length {SEQ_LEN} "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_gpus} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
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

    engine_gpus = args.recipe.rollout_num_gpus_per_engine
    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} --sglang-tp-size {engine_gpus} --sglang-ep-size {engine_gpus} "
        "--sglang-linear-attn-prefill-backend flashinfer "
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-disable-radix-cache "
        "--sglang-mem-fraction-static 0.7 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        f"--update-weight-buffer-size {1 * 1024**3} "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--model-name qwen4_exp "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )
    if args.check_weight_update_equal:
        # the PLE table is frozen and never shipped; the vision towers are re-sent from the checkpoint
        misc_args += "--check-weight-update-equal --check-weight-update-skip-list visual. ple_embedding. "
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    extra_env_vars = {"SGLANG_HEALTH_CHECK_TIMEOUT": "120", "SGLANG_DISABLE_MULTIMEM_AG": "1"}
    # ray's runtime_env replaces the actor env; without the JIT cache dirs the kernels recompile every run
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if os.environ.get(cache_var):
            extra_env_vars[cache_var] = os.environ[cache_var]

    args.create_backend().execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{perf_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
            f"{sglang_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
