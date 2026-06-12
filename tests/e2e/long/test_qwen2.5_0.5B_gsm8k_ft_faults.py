"""GSM8K accuracy-gain test under fault injection.

Same training recipe as test_qwen2.5_0.5B_gsm8k.py (the no-fault baseline, whose
wandb curves serve as the reference), plus train-side fault tolerance and a
seeded random fault schedule (train cell crashes + rollout engine kills). The
test asserts that the model still reaches a calibrated eval/gsm8k accuracy, i.e.
that fault recovery preserves end-to-end learning, which dump-comparison FT
tests cannot observe.
"""

import json
import os
import random

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=9000, suite="stage-c-2-gpu-h200", labels=["long"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2
NUM_ENGINES = 2  # colocate, --rollout-num-gpus-per-engine 1

# Override knobs for smoke/calibration runs; CI uses the defaults.
NUM_ROLLOUT = U.get_int_env_var("MILES_TEST_NUM_ROLLOUT", 250)
# Provisional threshold pending calibration runs (baseline asserts 0.55 at 250
# steps without faults); update after collecting the fault-run distribution.
METRIC_THRESHOLD = U.get_float_env_var("MILES_TEST_METRIC_THRESHOLD", 0.45)

# The fault schedule is fully determined by this seed (and NUM_ROLLOUT), so a
# red run can be replayed exactly; the generated JSON also appears verbatim in
# the logged training command.
FAULT_SEED = 20260612
NUM_TRAIN_FAULT_UNITS = 2
NUM_ENGINE_KILLS = 2


def generate_fault_schedules() -> tuple[str, str]:
    rng = random.Random(FAULT_SEED)
    num_faults = NUM_TRAIN_FAULT_UNITS + NUM_ENGINE_KILLS
    fault_lo = 6
    fault_hi = NUM_ROLLOUT - 4
    assert fault_hi - fault_lo >= num_faults, (
        f"NUM_ROLLOUT={NUM_ROLLOUT} is too small to schedule {num_faults} faults in "
        f"[{fault_lo}, {fault_hi}); the test would not exercise fault recovery at all"
    )
    max_feasible_gap = (fault_hi - fault_lo - 1) // max(1, num_faults - 1)
    min_gap = min(max(3, NUM_ROLLOUT // 20), max_feasible_gap)

    for _ in range(10000):
        rollouts = sorted(rng.sample(range(fault_lo, fault_hi), num_faults))
        if all(b - a >= min_gap for a, b in zip(rollouts, rollouts[1:], strict=False)):
            break
    else:
        raise RuntimeError(f"No fault schedule with gap >= {min_gap} found in range ({fault_lo}, {fault_hi})")

    # Each train fault unit mirrors the FT e2e with_failure scenario: rank 0 of
    # the last cell crashes before allreduce (degraded-quorum commit on retry),
    # then the cell is stopped and restarted to exercise healing.
    train_actions: list[dict] = []
    for at_rollout in rollouts[:NUM_TRAIN_FAULT_UNITS]:
        train_actions += [
            {"at_rollout": at_rollout, "action": "crash_before_allreduce", "cell_index": -1, "rank": 0, "attempt": 0},
            {"at_rollout": at_rollout, "action": "stop_cell_at_end", "cell_index": -1},
            {"at_rollout": at_rollout, "action": "start_cell_at_end", "cell_index": -1},
        ]

    engine_kills: list[dict] = [
        {"at_rollout": at_rollout, "engine_index": rng.randrange(NUM_ENGINES)}
        for at_rollout in rollouts[NUM_TRAIN_FAULT_UNITS:]
    ]

    return json.dumps(train_actions), json.dumps(engine_kills)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    train_fault_actions, engine_kill_schedule = generate_fault_schedules()

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
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
        f"--ci-engine-kill-schedule '{engine_kill_schedule}' "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        f"--ci-metric-checker-threshold {METRIC_THRESHOLD} "
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
        f"{U.get_default_wandb_args(__file__)} "
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
            # Same as tests/e2e/ft: a cell respawned after a crash cold-recompiles
            # its first forward, which is slow and memory-heavy enough to OOM.
            "TORCHDYNAMO_DISABLE": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
