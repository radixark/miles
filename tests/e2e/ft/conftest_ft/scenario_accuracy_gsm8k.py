# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import json
import os
import random
from typing import Annotated

import typer

import miles.utils.external_utils.command_utils as U

app: typer.Typer = typer.Typer()

TEST_NAME: str = "trainer_ft_accuracy_gsm8k"

MODEL_NAME: str = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE: str = "qwen2.5-0.5B"
NUM_GPUS: int = 2

# Same recipe as tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py (the no-fault baseline,
# whose wandb curves serve as the reference) plus train-side fault tolerance and a
# seeded random fault schedule. The fault schedule is fully determined by
# (--seed, --num-rollout), so a red run can be replayed exactly; the generated
# values also appear verbatim in the logged training command.
_DEFAULT_FAULT_SEED: int = 20260612
_NUM_TRAIN_FAULT_UNITS: int = 2
_NUM_ENGINE_KILLS: int = 2
# Provisional threshold pending calibration runs (the baseline asserts 0.55 at 250
# steps without faults); update after collecting the fault-run distribution.
_DEFAULT_METRIC_THRESHOLD: float = 0.45


def _generate_fault_schedule(*, seed: int, num_rollout: int) -> tuple[str, list[int]]:
    rng = random.Random(seed)
    num_faults = _NUM_TRAIN_FAULT_UNITS + _NUM_ENGINE_KILLS
    fault_lo = 6
    fault_hi = num_rollout - 4
    assert fault_hi - fault_lo >= num_faults, (
        f"num_rollout={num_rollout} is too small to schedule {num_faults} faults in "
        f"[{fault_lo}, {fault_hi}); the test would not exercise fault recovery at all"
    )
    max_feasible_gap = (fault_hi - fault_lo - 1) // max(1, num_faults - 1)
    min_gap = min(max(3, num_rollout // 20), max_feasible_gap)

    for _ in range(10000):
        rollouts = sorted(rng.sample(range(fault_lo, fault_hi), num_faults))
        if all(b - a >= min_gap for a, b in zip(rollouts, rollouts[1:], strict=False)):
            break
    else:
        raise RuntimeError(f"No fault schedule with gap >= {min_gap} found in range ({fault_lo}, {fault_hi})")

    # Each train fault unit mirrors the with_failure scenario: rank 0 of the last
    # cell crashes before allreduce (degraded-quorum commit on retry), then the
    # cell is stopped and restarted to exercise healing.
    train_actions: list[dict] = []
    for at_rollout in rollouts[:_NUM_TRAIN_FAULT_UNITS]:
        train_actions += [
            {"at_rollout": at_rollout, "action": "crash_before_allreduce", "cell_index": -1, "rank": 0, "attempt": 0},
            {"at_rollout": at_rollout, "action": "stop_cell_at_end", "cell_index": -1},
            {"at_rollout": at_rollout, "action": "start_cell_at_end", "cell_index": -1},
        ]

    engine_kill_rollout_ids: list[int] = rollouts[_NUM_TRAIN_FAULT_UNITS:]

    return json.dumps(train_actions), engine_kill_rollout_ids


def prepare() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


@app.command(name="run")
def run_ci(
    seed: Annotated[int, typer.Option(help="Random seed for the fault schedule")] = _DEFAULT_FAULT_SEED,
    num_rollout: Annotated[int, typer.Option(help="Number of rollouts")] = 250,
    metric_threshold: Annotated[
        float, typer.Option(help="eval/gsm8k accuracy threshold")
    ] = _DEFAULT_METRIC_THRESHOLD,
) -> None:
    """GSM8K accuracy-gain test under fault injection.

    Asserts that the model still reaches the eval/gsm8k threshold despite train
    cell crashes and rollout engine kills, i.e. that fault recovery preserves
    end-to-end learning, which dump-comparison FT tests cannot observe.

    Doubles as the CI entry point: the CI file calls ``run_ci()`` (defaults);
    manual smoke/calibration runs use the ``run`` CLI subcommand with
    --seed/--num-rollout/--metric-threshold.
    """
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    train_fault_actions, engine_kill_rollout_ids = _generate_fault_schedule(seed=seed, num_rollout=num_rollout)

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

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
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 " "--sglang-enable-metrics "

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--ft-components train "
        "--control-server-port 0 "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
        f"--ci-ft-test-actions '{train_fault_actions}' "
        f"--ci-engine-kill-rollout-ids {' '.join(str(x) for x in engine_kill_rollout_ids)} "
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
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
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

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            # --ft-components train depends on cell-based indep_dp, which only
            # the v2 RayTrainGroup supports.
            "MILES_EXPERIMENTAL_FT_TRAINER": "1",
            # Same as run_training in execution.py: a cell respawned after a crash
            # cold-recompiles its first forward, which is slow and memory-heavy
            # enough to OOM.
            "TORCHDYNAMO_DISABLE": "1",
        },
    )


if __name__ == "__main__":
    app()
