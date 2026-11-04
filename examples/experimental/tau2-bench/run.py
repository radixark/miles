"""τ²-bench GRPO launcher: tau2 tasks through the TITO session server.

Trains on one tau2 domain's ``train`` split and evaluates on its ``test`` split.
Each sample is one tau2 task, played by ``tau2_agent_function.run``: tau2's own
orchestrator drives the user simulator, environment and evaluator, and the policy
answers through the session server's OpenAI-compatible endpoint.

The policy and how it runs -- checkpoint, GPUs and parallelism, SGLang parsers,
TITO family, chat-template settings, sequence budgets -- come from its entry in
``policy_models.py``; this file holds only the tau2 workload.

Requires tau2-bench in the training environment (see README.md), and the API
keys of the user simulator -- and, for retail and airline, of tau2's reward
judges -- exported in the launcher's environment, which ray workers inherit.
Self-hosted models served behind an OpenAI-compatible endpoint need no key (see
README.md). ``execute`` checks the keys before claiming any GPU.

Args:
  --model-name: policy model, a key of POLICY_MODELS in policy_models.py (default: Qwen3.8-27B).
  --domain: tau2 domain to train on: airline, retail or telecom (default: retail).
  --mode: normal, or debug_rollout_only to skip training.
  --skip-prepare: skip the model download, checkpoint conversion and task files.
  --model-dir / --data-dir: model and dataset directories.
  --enable-eval / --save: evaluate on the test split / write checkpoints (default: on).

Env vars forwarded to the agent: TAU2_USER_LLM, TAU2_USER_LLM_ARGS,
TAU2_JUDGE_LLM, TAU2_JUDGE_LLM_ARGS (see tau2_agent_function.py), and
TAU2_DATA_DIR when tau2-bench is not installed from its source checkout.

Usage:
    OPENAI_API_KEY=... MILES_SCRIPT_EXTERNAL_RAY=1 python run.py
    OPENAI_API_KEY=... MILES_SCRIPT_EXTERNAL_RAY=1 python run.py --domain telecom --mode debug_rollout_only
"""

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from policy_models import POLICY_MODELS, PolicyModel

from miles.utils.external_utils import command_utils

SCRIPT_DIR = Path(__file__).resolve().parent

_FORWARDED_ENV_VARS = (
    "TAU2_USER_LLM",
    "TAU2_USER_LLM_ARGS",
    "TAU2_JUDGE_LLM",
    "TAU2_JUDGE_LLM_ARGS",
    "TAU2_DATA_DIR",
)


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    domain: Literal["airline", "retail", "telecom"] = "retail"
    model_name: str = "Qwen3.8-27B"
    skip_prepare: bool = False
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    num_rollout: int = 200
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    enable_eval: bool = True
    eval_interval: int = 10
    save: bool = True
    save_interval: int = 20

    @property
    def model(self) -> PolicyModel:
        if self.model_name not in POLICY_MODELS:
            raise ValueError(f"--model-name {self.model_name!r} is not in policy_models.py: {sorted(POLICY_MODELS)}")
        return POLICY_MODELS[self.model_name]

    def task_file(self, split: str) -> str:
        return f"{self.data_dir}/tau2-bench/{self.domain}_{split}.jsonl"


def prepare(args: ScriptArgs):
    model = args.model
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}/tau2-bench")
    U.exec_command_cpu(f"hf download {model.hf_repo} --local-dir {args.model_dir}/{args.model_name}")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=model.megatron_model_type,
        num_gpus_per_node=model.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )
    for split in ("train", "test"):
        _write_task_file(args.domain, split, args.task_file(split))


def _write_task_file(domain: str, split: str, path: str) -> None:
    """One row per tau2 task; the agent loads the task itself from tau2 by its id."""
    from tau2.runner.helpers import get_tasks  # noqa: PLC0415 - tau2 is needed only where data is written

    rows = [
        {
            "prompt": [{"role": "user", "content": f"tau2 {domain} task {task.id}"}],
            "metadata": {"domain": domain, "task_id": task.id},
        }
        for task in get_tasks(domain, task_split_name=split)
    ]
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} tau2 {domain}/{split} tasks to {path}")


def _preflight_api_keys(domain: str) -> None:
    """Fail before claiming GPUs when an episode could not reach its user simulator or judges.

    Every episode would then fail the same way outside the policy, and training would only drop
    groups and refill them. The judges are the ones the domain's reward bases use.
    """
    import litellm  # noqa: PLC0415 - tau2's dependency; the launcher needs it only for this check
    from tau2.runner.helpers import get_tasks  # noqa: PLC0415 - tau2 is needed only where episodes are checked
    from tau2_agent_function import episode_llms  # noqa: PLC0415 - imports tau2 too

    bases = {
        basis
        for task in get_tasks(domain)
        if task.evaluation_criteria is not None
        for basis in task.evaluation_criteria.reward_basis
    }
    llms = episode_llms(bases)
    missing = {
        key
        for model, model_args in llms
        for key in litellm.validate_environment(
            model=model, api_key=model_args.get("api_key"), api_base=model_args.get("api_base")
        )["missing_keys"]
    }
    if missing:
        raise RuntimeError(
            f"tau2 {domain} episodes call {sorted({model for model, _ in llms})}, which need {sorted(missing)} set. "
            "Export them before launching (ray workers inherit the launcher's environment), or point "
            "TAU2_USER_LLM / TAU2_JUDGE_LLM at your own endpoint."
        )


def _model_args(model: PolicyModel) -> str:
    """How the policy runs: parallelism, optimizer placement, its SGLang engine and its chat template."""
    args = (
        f"--tensor-model-parallel-size {model.tensor_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {model.context_parallel_size} "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {model.max_seq_len // model.context_parallel_size} "
        f"--rollout-max-response-len {model.rollout_max_response_len} "
        f"--max-seq-len {model.max_seq_len} "
        f"--rollout-num-gpus-per-engine {model.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {model.sglang_mem_fraction_static} "
        f"--sglang-tool-call-parser {model.sglang_tool_call_parser} "
        f"--tito-model {model.tito_model} "
        f"--actor-num-nodes {model.num_nodes} "
        f"--actor-num-gpus-per-node {model.num_gpus_per_node} "
        f"--num-gpus-per-node {model.num_gpus_per_node} "
    )
    if model.sglang_reasoning_parser is not None:
        args += f"--sglang-reasoning-parser {model.sglang_reasoning_parser} "
    if model.chat_template_kwargs:
        args += f"--apply-chat-template-kwargs {shlex.quote(json.dumps(model.chat_template_kwargs))} "
    if model.optimizer_cpu_offload:
        args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    return args


def execute(args: ScriptArgs):
    model = args.model
    _preflight_api_keys(args.domain)
    U = args.create_backend()

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
    )
    if args.save:
        ckpt_args += f"--save {args.output_dir}/checkpoints --save-interval {args.save_interval} "

    rollout_args = (
        f"--prompt-data {args.task_file('train')} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data {args.domain}-test {args.task_file('test')} "
            "--n-samples-per-eval-prompt 4 "
            f"--eval-max-response-len {model.rollout_max_response_len} "
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

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path tau2_agent_function.run "
        "--custom-rm-path tau2_agent_function.reward_func "
        "--use-session-server "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{eval_args}"
        f"{_model_args(model)}"
        f"{grpo_args}"
        f"{optimizer_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{debug_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
        **{name: os.environ[name] for name in _FORWARDED_ENV_VARS if name in os.environ},
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=model.num_gpus_per_node,
        megatron_model_type=model.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
