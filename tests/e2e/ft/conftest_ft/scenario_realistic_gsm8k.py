# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import os
from typing import Annotated

import typer

from tests.e2e.ft.conftest_ft.fault_injection import CONTROL_SERVER_PORT, MEAN_INTERVAL_SECONDS, spawn_fault_injector

import miles.utils.external_utils.command_utils as U

app: typer.Typer = typer.Typer()

TEST_NAME: str = "trainer_ft_realistic_gsm8k"

_MODEL_NAME: str = "Qwen2.5-0.5B-Instruct"
_MODEL_TYPE: str = "qwen2.5-0.5B"
# Same disaggregated layout as the dp2_cp2_real_rollout mode: 2 cells x CP2 on
# 4 training GPUs, plus 4 rollout engines x 1 GPU.
_TRAIN_GPUS: int = 4
_ROLLOUT_GPUS: int = 4
# Must stay identical to the threshold asserted by the no-fault baseline
# tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py: fault recovery must not cost accuracy.
_DEFAULT_METRIC_THRESHOLD: float = 0.55


@app.command(name="run")
def run_ci(
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_rollout: Annotated[int, typer.Option(help="Number of rollouts")] = 250,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = 0.1,
    metric_threshold: Annotated[float, typer.Option(help="eval/gsm8k accuracy threshold")] = _DEFAULT_METRIC_THRESHOLD,
) -> None:
    mean_interval: float = MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Rollouts: {num_rollout}, Mean injection interval: {mean_interval:.1f}s")

    _prepare_gsm8k()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    train_args = _get_gsm8k_train_args(seed=seed, num_rollout=num_rollout, metric_threshold=metric_threshold)

    injector = spawn_fault_injector(seed=seed, mean_interval_seconds=mean_interval)

    try:
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=_TRAIN_GPUS + _ROLLOUT_GPUS,
            megatron_model_type=_MODEL_TYPE,
            extra_env_vars={
                "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
                # --ft-components train depends on cell-based indep_dp, which only
                # the v2 RayTrainGroup supports.
                "MILES_EXPERIMENTAL_FT_TRAINER": "1",
                # Same as run_training: a cell respawned after a crash cold-recompiles
                # its first forward, which is slow and memory-heavy enough to OOM.
                "TORCHDYNAMO_DISABLE": "1",
            },
        )
    finally:
        injector.stop_and_join(timeout_seconds=5)

    print(f"Random failure gsm8k accuracy test PASSED (seed={seed}, rollouts={num_rollout})")


def _prepare_gsm8k() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{_MODEL_NAME} --local-dir /root/models/{_MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _get_gsm8k_train_args(*, seed: int, num_rollout: int, metric_threshold: float) -> str:
    ckpt_args = f"--hf-checkpoint /root/models/{_MODEL_NAME}/ " f"--ref-load /root/models/{_MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        # Parallelism mirrors the dp2_cp2_real_rollout mode (2 cells x CP2), not
        # the no-fault baseline test.
        "--context-parallel-size 2 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
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

    sglang_args = (
        f"--rollout-num-gpus {_ROLLOUT_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-metrics "
    )

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--ft-components train "
        f"--control-server-port {CONTROL_SERVER_PORT} "
        "--mini-ft-controller-enable "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        f"--ci-metric-checker-threshold {metric_threshold} "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {_TRAIN_GPUS} "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(f'test_{TEST_NAME}.py', run_name_prefix=f'seed{seed}')} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fault_tolerance_args} "
        f"{ci_args} "
        f"{misc_args} "
    )


if __name__ == "__main__":
    app()
