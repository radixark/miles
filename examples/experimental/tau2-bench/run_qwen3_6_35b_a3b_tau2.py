"""tau2-bench GRPO full-parameter launcher for Qwen3.6-35B-A3B (native Megatron, 8xH200).

Trains on the train splits of airline / retail / telecom and evaluates every
domain's test split (``eval/<domain>`` in the logs is that domain's pass^1,
averaged over ``--n-samples-per-eval-prompt`` samples per task). The model / parallel / optimizer layout follows
``scripts/run_qwen3_6_35b_a3b_mtp.py`` without MTP training or speculative
decoding: TP1 + SP, EP8, CPU-offloaded precision-aware Adam, one 8-GPU sglang
engine, colocated, with rollout routing replay (R3).

Each episode is one tau2 simulation driven by ``tau2_agent_function.run``; the
policy speaks Qwen3.6's native tool-call format (sglang ``qwen3_coder`` parser,
``qwen3`` reasoning parser). By default the user simulator is the policy model
itself, served by the rollout router, and so is retail's NL-assertion judge
(see ``tau2_agent_function``).

Requires: ``pip install -e <tau2-bench checkout> --no-deps`` plus its missing
deps. ``prepare`` downloads the model, converts it to torch_dist (the
``--ref-load``) and writes the prompt data with ``make_tau2_data.py``.

Args:
  --mode               normal | debug_rollout_only (eval + rollouts, no training)
  --num-rollout        RL steps
  --eval-interval      eval every N steps (eval also runs before step 0)
  --n-samples-per-eval-prompt  eval samples per task (temperature 1.0)
  --use-r3             rollout routing replay

wandb engages when `WANDB_API_KEY` is set; pass `--wandb-team` / `--wandb-host` through `--extra-args`.

Usage:
  python run_qwen3_6_35b_a3b_tau2.py prepare
  python run_qwen3_6_35b_a3b_tau2.py train --num-rollout 3
  python run_qwen3_6_35b_a3b_tau2.py train --mode debug_rollout_only --num-rollout 1
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

SCRIPT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

_FORWARDED_ENV = (
    "TAU2_USER_LLM",
    "TAU2_USER_API_BASE",
    "TAU2_JUDGE_LLM",
    "TAU2_JUDGE_API_BASE",
    "TAU2_MAX_STEPS",
    "TAU2_EPISODE_TIMEOUT_S",
    "TAU2_COMMUNICATE_SCORER",
)

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    model_name: str = "Qwen3.6-35B-A3B"
    megatron_model_type: str = "qwen3.6-35B-A3B"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/tau2"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    num_rollout: int = 3
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    # thinking exhausts 8192 tokens in ~10% of turns, leaving an empty reply tau2 rejects
    rollout_max_response_len: int = 16384
    # 49152 OOMs the logits on 8xH200
    max_seq_len: int = 32768
    n_samples_per_eval_prompt: int = 4
    eval_interval: int = 2
    use_r3: bool = True
    extra_args: str = ""

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def ref_load(self) -> str:
        return f"{self.model_dir}/{self.model_name}_torch_dist"


def _prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"test -e {args.hf_checkpoint} || hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}"
    )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )
    U.exec_command_cpu(f"python {SCRIPT_DIR}/make_tau2_data.py --output-dir {args.data_dir}")


def _train(args: ScriptArgs):
    U = args.create_backend()

    # no --save: a full checkpoint of this model is ~460 GB and these runs are short
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/all_train.jsonl "
        "--input-key prompt --metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    eval_args = (
        f"--eval-interval {args.eval_interval} "
        "--eval-prompt-data "
        + " ".join(f"{domain} {args.data_dir}/{domain}_test.jsonl" for domain in ("airline", "retail", "telecom"))
        + f" --n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
        f"--eval-max-response-len {args.rollout_max_response_len} "
        "--eval-temperature 1.0 "
    )

    grpo_args = (
        "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    if args.use_r3:
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam --lr 2e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_seq_len} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        f"--sglang-ep-size {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-mamba-radix-cache-strategy extra_buffer "
        "--sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 "
    )

    agent_args = (
        "--custom-generate-function-path tau2_generate.generate "
        "--custom-agent-function-path tau2_agent_function.run "
        "--custom-rm-path tau2_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-session-server --session-server-port 30000 --session-server-workers 32 "
        "--tito-model qwen36 --session-message-matcher role_content_only "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        "--moe-token-dispatcher-type flex "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )
    if args.mode == "debug_rollout_only":
        misc_args += "--debug-rollout-only "

    wandb_args = command_utils.get_default_wandb_args(__file__, run_id=args.run_id)

    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {optimizer_args} "
        f"{perf_args} {sglang_args} {agent_args} {misc_args} {wandb_args} {args.extra_args} "
    )

    policy_roles = [
        role
        for role, key in (("user simulator", "TAU2_USER_LLM"), ("judge", "TAU2_JUDGE_LLM"))
        if key not in os.environ
    ]
    if policy_roles:
        logger.warning(
            f"The policy also plays the tau2 {' and '.join(policy_roles)}, so it moves with every weight sync and "
            "pass rates are not comparable across steps or with the leaderboard. Set TAU2_USER_LLM / "
            "TAU2_JUDGE_LLM to a fixed external model (the leaderboard uses gpt-4.1) or a frozen copy of the "
            "initial weights."
        )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
            **{key: os.environ[key] for key in _FORWARDED_ENV if key in os.environ},
        },
    )


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Download Qwen3.6-35B-A3B, convert it to torch_dist and write the tau2 prompt data."""
    _prepare(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Run tau2 GRPO full-parameter training (eval before step 0 and every --eval-interval steps)."""
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
