from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k

register_cuda_ci(
    est_time=9000,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    hardware=["hopper", "blackwell"],
)

if __name__ == "__main__":
    scenario_realistic_gsm8k.run_ci(
        seed=scenario_realistic_gsm8k.DEFAULT_SEED,
        num_rollout=scenario_realistic_gsm8k.DEFAULT_NUM_ROLLOUT,
        trainer_crash_interval_seconds=1e9,
        rollout_crash_interval_seconds=1e9,
        metric_threshold=scenario_realistic_gsm8k.DEFAULT_METRIC_THRESHOLD,
        fully_async=True,
        requested_triggers=None,
    )
