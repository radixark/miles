from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import compute_model_args_overrides, encode_pseudo_file

SOLVER_MODEL_ID: str = "solver"
VERIFIER_MODEL_ID: str = "verifier"
MODEL_IDS: list[str] = [SOLVER_MODEL_ID, VERIFIER_MODEL_ID]
LEADER_MODEL_ID: str = MODEL_IDS[0]
EVAL_DATASET_NAME: str = "gsm8k"
TRAIN_SCRIPT: str = "train_multi_policy.py"
TRAIN_EXTRA_ENV_VARS: dict[str, str] = {"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"}

# Hyperparameters roughly follow test_qwen2.5_0.5B_gsm8k_async.py.
SHARED_TRAINER_OVERRIDES = dict(
    lr_decay_style="constant",
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    entropy_coef=0.0,
    eps_clip=0.2,
    eps_clip_high=0.28,
)


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_rollout: int = 3
    num_gpus_per_node: int = 4
    solver_model_name: str = "Qwen2.5-0.5B-Instruct"
    verifier_model_name: str = "Qwen3-0.6B"
    solver_megatron_model_type: str = "qwen2.5-0.5B"
    verifier_megatron_model_type: str = "qwen3-0.6B"
    rollout_num_gpus_per_model: int = 1
    actor_num_gpus_per_policy: int = 1
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    extra_args: str = ""

    @property
    def rollout_num_gpus(self) -> int:
        return self.rollout_num_gpus_per_model * len(MODEL_IDS)

    @property
    def model_path_of_model_id(self) -> dict[str, str]:
        return {
            SOLVER_MODEL_ID: f"{self.model_dir}/{self.solver_model_name}",
            VERIFIER_MODEL_ID: f"{self.model_dir}/{self.verifier_model_name}",
        }

    @property
    def megatron_model_type_of_model_id(self) -> dict[str, str]:
        return {
            SOLVER_MODEL_ID: self.solver_megatron_model_type,
            VERIFIER_MODEL_ID: self.verifier_megatron_model_type,
        }


def prepare(args: ScriptArgs) -> None:
    U = args.create_backend()
    model_path_of_model_id = args.model_path_of_model_id
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download Qwen/{args.solver_model_name} --local-dir {model_path_of_model_id[SOLVER_MODEL_ID]}"
    )
    U.exec_command_cpu(
        f"hf download Qwen/{args.verifier_model_name} --local-dir {model_path_of_model_id[VERIFIER_MODEL_ID]}"
    )
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def execute(args: ScriptArgs) -> None:
    launch_train(build_train_args(args), args)


def build_train_args(
    args: ScriptArgs,
    *,
    wandb_args: str | None = None,
    megatron_config: dict | None = None,
    sglang_config: dict | None = None,
    rollout_num_gpus: int | None = None,
) -> str:
    events_dir = compute_events_dir(args)
    solver_path = args.model_path_of_model_id[SOLVER_MODEL_ID]

    ckpt_args = f"--hf-checkpoint {solver_path}/ " f"--ref-load {solver_path}/ "

    policy_args = (
        f"--megatron-config {encode_pseudo_file(yaml.dump(megatron_config or compute_megatron_config(args)))} "
        f"--sglang-config {encode_pseudo_file(yaml.dump(sglang_config or compute_sglang_config(args)))} "
        "--custom-generate-function-path examples.multi_policy.solver_verifier.generate "
    )

    rollout_args = (
        "--fully-async "
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        # retract (default) can deadlock flush_cache in fully_async under load
        "--pause-generation-mode in_place "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data {EVAL_DATASET_NAME} {args.data_dir}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
        "--custom-eval-rollout-log-function-path examples.multi_policy.solver_verifier.split_eval_data_by_policy "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = "--advantage-estimator grpo " "--use-kl-loss "

    optimizer_args = "--optimizer adam " "--lr 1e-6 "

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.65 " "--sglang-enable-metrics "

    ci_args = "--ci-test " f"--save-debug-event-data {events_dir} "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_policy} "
        f"--rollout-num-gpus {args.rollout_num_gpus if rollout_num_gpus is None else rollout_num_gpus} "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"{ckpt_args} "
        f"{policy_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args if wandb_args is not None else command_utils.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )


def launch_train(train_args: str, args: ScriptArgs) -> None:
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type_of_model_id[LEADER_MODEL_ID],
        train_script=TRAIN_SCRIPT,
        extra_env_vars=dict(TRAIN_EXTRA_ENV_VARS),
    )


def compute_megatron_config(args: ScriptArgs, *, model_ids: list[str] | None = None) -> dict:
    return dict(trainers=[_compute_trainer_config(args, model_id) for model_id in model_ids or MODEL_IDS])


def compute_sglang_config(args: ScriptArgs, *, model_ids: list[str] | None = None) -> dict:
    return dict(sglang=[_compute_sglang_model_config(args, model_id) for model_id in model_ids or MODEL_IDS])


def compute_trainer_id(model_id: str) -> str:
    return f"{model_id}-{ACTOR_ROLE}"


def compute_events_dir(config: ExecuteTrainConfig) -> Path:
    return Path(config.output_dir) / "multi_policy_solver_verifier" / config.run_id / "events"


def _compute_trainer_config(args: ScriptArgs, model_id: str) -> dict:
    model_path = args.model_path_of_model_id[model_id]
    return dict(
        model_id=model_id,
        trainer_id=compute_trainer_id(model_id),
        overrides=dict(
            hf_checkpoint=model_path,
            ref_load=model_path,
            **compute_model_args_overrides(args.megatron_model_type_of_model_id[model_id]),
            **SHARED_TRAINER_OVERRIDES,
        ),
    )


def _compute_sglang_model_config(args: ScriptArgs, model_id: str) -> dict:
    return dict(
        name=model_id,
        model_path=args.model_path_of_model_id[model_id],
        update_weights=True,
        num_gpus_per_engine=1,
        server_groups=[dict(worker_type="regular", num_gpus=args.rollout_num_gpus_per_model)],
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    prepare(args)
    execute(args)


# TODO: unify this launcher when the example scripts are refactored
if __name__ == "__main__":
    typer.run(main)
