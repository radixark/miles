"""Verifiers launcher (Qwen3-0.6B on code-golf-v1): Miles <-> Verifiers V1.

Defaults reproduce the two-GPU smoke configuration: 3 GRPO steps against the
`code-golf-v1` taskset. Scale --num-rollout and the batch sizes for real
training, and point --verifiers-config at your own EnvConfig TOML.

The Verifiers environment must already be installed in this interpreter, e.g.
`prime env install code-golf-v1` from a workspace with ./environments (see
README.md).

Usage:
    python run.py
    python run.py --verifiers-config /path/to/verifiers.toml --num-rollout 50
    python run.py --eval-interval 5      # evaluate the whole taskset every 5 rollouts
"""

import os
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_ROLLOUT_ENV = "MILES_USE_LEGACY_ROLLOUT_V1"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    megatron_model_type: str = "qwen3-0.6B"
    num_gpus_per_node: int = 2
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "Qwen3-0.6B"
    hf_checkpoint: str = "/root/models/Qwen3-0.6B"
    # Renderers resolves its renderer from the model identity, which a local
    # snapshot path does not carry; keep the registered id here.
    sglang_tokenizer_path: str = "Qwen/Qwen3-0.6B"
    ref_load: str = "/root/models/Qwen3-0.6B_torch_dist"
    save_dir: str = "/root/Qwen3-0.6B_verifiers/"

    # The Verifiers EnvConfig TOML. Written below when left at the default.
    verifiers_config: str = "/root/verifiers-code-golf.toml"
    taskset_id: str = "code-golf-v1"

    # Training settings (smoke scale)
    rollout_max_response_len: int = 512
    rollout_max_context_len: int = 2048
    num_rollout: int = 3
    rollout_batch_size: int = 3
    n_samples_per_prompt: int = 4
    global_batch_size: int = 12

    # Evaluation over the whole taskset, every N training rollouts. 0 disables it.
    # Group rewards need at least two rollouts per task to rank.
    eval_interval: int = 0
    n_samples_per_eval_prompt: int = 2


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=str(Path(args.hf_checkpoint).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    config_path = Path(args.verifiers_config)
    if not config_path.exists():
        config_path.write_text(f'[taskset]\nid = "{args.taskset_id}"\n')

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{U.repo_base_dir}",
        "VERIFIERS_CONFIG": str(config_path),
    }
    if LEGACY_ROLLOUT_ENV in os.environ:
        extra_env_vars[LEGACY_ROLLOUT_ENV] = os.environ[LEGACY_ROLLOUT_ENV]
    extra_env_vars = U.resolve_extra_env_vars(extra_env_vars, args)

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--sglang-tokenizer-path {args.sglang_tokenizer_path} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        "--save-interval 1000 "
    )

    # Verifiers owns the taskset, so Miles loads no prompt data; the rollout
    # function plug-point selects the adapter, which resolves as a bare module
    # because PYTHONPATH carries this directory into the rollout actor.
    rollout_fn = (
        "verifiers_rollout.generate_rollout"
        if bool(int(extra_env_vars.get(LEGACY_ROLLOUT_ENV, "0")))
        else "verifiers_rollout.VerifiersRolloutFn"
    )
    rollout_args = (
        f"--rollout-function-path {rollout_fn} "
        "--disable-rollout-global-dataset "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--over-sampling-batch-size {args.rollout_batch_size} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        "--rollout-temperature 0.8 "
        f"--rollout-num-gpus-per-engine 1 "
    )

    # Workaround: the taskset is the evaluation set, but Miles asserts that eval
    # datasets are configured whenever --eval-interval is set, so name the taskset
    # and point the placeholder at the config it is defined in -- the adapter serves
    # eval, so the built-in loader never opens this path. Worth replacing with a
    # Miles-side fix once a second rollout function owns its evaluation set.
    eval_args = (
        (
            f"--eval-interval {args.eval_interval} "
            f"--n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
            f"--eval-prompt-data verifiers-taskset {config_path} "
        )
        if args.eval_interval
        else ""
    )

    grpo_args = (
        "--advantage-estimator grpo "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        "--entropy-coef 0.0 "
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

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--no-gradient-accumulation-fusion "
    )

    sglang_args = "--sglang-mem-fraction-static 0.6 --sglang-enable-metrics "

    misc_args = (
        "--attention-backend flash "
        "--colocate "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args}{rollout_args}{eval_args}{grpo_args}{optimizer_args}{perf_args}{sglang_args}{misc_args}"
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
