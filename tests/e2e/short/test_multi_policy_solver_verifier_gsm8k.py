import os
import statistics
from pathlib import Path
from typing import NamedTuple

import yaml
from tests.ci.ci_register import register_cuda_ci

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import compute_model_args_overrides, encode_pseudo_file

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["short", "multi-policy", "fully-async"])

SOLVER_MODEL_ID = "solver"
VERIFIER_MODEL_ID = "verifier"
SOLVER_MODEL_NAME = "Qwen2.5-0.5B-Instruct"
VERIFIER_MODEL_NAME = "Qwen3-0.6B"
SOLVER_MODEL_TYPE = "qwen2.5-0.5B"
VERIFIER_MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 8
NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "3"))

SOLVER_PATH = f"/root/models/{SOLVER_MODEL_NAME}"
VERIFIER_PATH = f"/root/models/{VERIFIER_MODEL_NAME}"

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

SGLANG_CONFIG = dict(
    sglang=[
        dict(
            name=SOLVER_MODEL_ID,
            model_path=SOLVER_PATH,
            update_weights=True,
            num_gpus_per_engine=1,
            server_groups=[dict(worker_type="regular", num_gpus=2)],
        ),
        dict(
            name=VERIFIER_MODEL_ID,
            model_path=VERIFIER_PATH,
            update_weights=True,
            num_gpus_per_engine=1,
            server_groups=[dict(worker_type="regular", num_gpus=2)],
        ),
    ]
)


class TrainRewardBounds(NamedTuple):
    initial_max: float
    final_min: float


TRAIN_REWARD_BOUNDS = {
    SOLVER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.01),
    VERIFIER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.01),
}


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{SOLVER_MODEL_NAME} --local-dir {SOLVER_PATH}")
    U.exec_command_cpu(f"hf download Qwen/{VERIFIER_MODEL_NAME} --local-dir {VERIFIER_PATH}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute(*, num_rollout: int = NUM_ROLLOUT, train_reward_bounds: dict[str, TrainRewardBounds] | None = None):
    config = command_utils.default_config()
    U = config.create_backend()
    events_dir = compute_events_dir(config)
    megatron_config = compute_megatron_config()

    ckpt_args = f"--hf-checkpoint {SOLVER_PATH}/ " f"--ref-load {SOLVER_PATH}/ "

    policy_args = (
        f"--megatron-config {encode_pseudo_file(yaml.dump(megatron_config))} "
        f"--sglang-config {encode_pseudo_file(yaml.dump(SGLANG_CONFIG))} "
        "--custom-generate-function-path examples.multi_policy.solver_verifier.generate "
    )

    rollout_args = (
        "--fully-async "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        # retract (default) can deadlock flush_cache in fully_async under load
        "--pause-generation-mode in_place "
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
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 4 "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{policy_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=SOLVER_MODEL_TYPE,
        train_script="train_multi_policy.py",
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )

    _assert_every_policy_learned(events_dir, bounds=train_reward_bounds or TRAIN_REWARD_BOUNDS)


def compute_megatron_config() -> dict:
    return dict(
        trainers=[
            _compute_trainer_config(model_id=SOLVER_MODEL_ID, model_type=SOLVER_MODEL_TYPE, model_path=SOLVER_PATH),
            _compute_trainer_config(
                model_id=VERIFIER_MODEL_ID, model_type=VERIFIER_MODEL_TYPE, model_path=VERIFIER_PATH
            ),
        ]
    )


def _compute_trainer_config(*, model_id: str, model_type: str, model_path: str) -> dict:
    return dict(
        model_id=model_id,
        overrides=dict(
            hf_checkpoint=model_path,
            ref_load=model_path,
            **compute_model_args_overrides(model_type),
            **SHARED_TRAINER_OVERRIDES,
        ),
    )


def compute_events_dir(config: ExecuteTrainConfig) -> Path:
    return Path(config.output_dir) / "multi_policy_solver_verifier" / config.run_id / "events"


def _assert_every_policy_learned(events_dir: Path, *, bounds: dict[str, TrainRewardBounds]) -> None:
    for model_id, model_bounds in bounds.items():
        rewards = _read_train_reward_series(events_dir, model_id=model_id)
        assert rewards, (
            f"no {_compute_train_reward_key(model_id)} value was logged under {events_dir}, so policy "
            f"{model_id!r} never reported a training reward and nothing about its learning can be checked"
        )

        initial = rewards[0]
        final = statistics.mean(rewards[-max(1, len(rewards) // 3) :])
        assert initial <= model_bounds.initial_max, (
            f"policy {model_id!r} starts at training reward {initial}, above {model_bounds.initial_max}; a run "
            f"that starts already solved cannot show that training moved it"
        )
        assert final >= model_bounds.final_min, (
            f"policy {model_id!r} ends at training reward {final}, below {model_bounds.final_min}; either its "
            f"reward function never fires, or training destroyed the model"
        )


def _read_train_reward_series(events_dir: Path, *, model_id: str) -> list[float]:
    reward_key = _compute_train_reward_key(model_id)
    step_key = f"{model_id}/rollout/step"
    points = [
        (event.metrics[step_key], event.metrics[reward_key])
        for event in read_events(events_dir)
        if isinstance(event, MetricEvent) and reward_key in event.metrics
    ]
    return [reward for _, reward in sorted(points)]


def _compute_train_reward_key(model_id: str) -> str:
    return f"{model_id}/rollout/raw_reward"


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
