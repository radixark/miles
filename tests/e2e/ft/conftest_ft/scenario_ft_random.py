# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import logging
import os
import random
import threading
import time
from typing import Annotated

import requests
import typer

from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    materialize_cyclic_debug_rollout_data,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

import miles.utils.external_utils.command_utils as U
from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)

app: typer.Typer = typer.Typer()

TEST_NAME: str = "trainer_ft_random"

_CONTROL_SERVER_PORT: int = 18080
_MEAN_INTERVAL_SECONDS: float = 60.0
# Hard floor between consecutive injections so the FT controller has time to
# spawn the replacement actor and let it rejoin before the next crash. Without
# this, the exponential delay can produce several injections within a few
# seconds, causing the all-cells-dead cascade.
_MIN_GAP_BETWEEN_INJECTIONS_SECONDS: float = 30.0
_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]

_GSM8K_MODEL_NAME: str = "Qwen2.5-0.5B-Instruct"
_GSM8K_MODEL_TYPE: str = "qwen2.5-0.5B"
# Same disaggregated layout as the dp2_cp2_real_rollout mode: 2 cells x CP2 on
# 4 training GPUs, plus 4 rollout engines x 1 GPU.
_GSM8K_TRAIN_GPUS: int = 4
_GSM8K_ROLLOUT_GPUS: int = 4
# Provisional threshold pending calibration runs (the no-fault baseline
# tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py asserts 0.55 at 250 steps); update
# after collecting the fault-run distribution.
_GSM8K_DEFAULT_METRIC_THRESHOLD: float = 0.45


def _run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    stop_event: threading.Event,
) -> None:
    rng = random.Random(seed)
    last_injection_at: float = 0.0

    while not stop_event.is_set():
        delay = rng.expovariate(1.0 / mean_interval_seconds)
        if stop_event.wait(timeout=delay):
            break

        elapsed = time.monotonic() - last_injection_at
        if elapsed < _MIN_GAP_BETWEEN_INJECTIONS_SECONDS:
            logger.info(
                "Skipping injection: only %.1fs since last, need %.1fs",
                elapsed,
                _MIN_GAP_BETWEEN_INJECTIONS_SECONDS,
            )
            continue

        try:
            resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
            resp.raise_for_status()
            cells = resp.json()["items"]
        except Exception:
            logger.info("Failed to list cells from control server", exc_info=True)
            continue

        # A cell is "alive" iff its Healthy condition is TRUE. Note: phase=="Running"
        # is also true for StateAllocatedErrored (cell crashed mid-step but not yet
        # cleaned up), so phase alone is too permissive.
        def _is_alive(cell: dict) -> bool:
            return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])

        alive = [c for c in cells if _is_alive(c)]
        # Skip injection only when killing one more would leave us with no
        # redundancy left (≤1 alive). Otherwise inject — even if some peers
        # are still mid-recovery, we tolerate further reductions because dp
        # still has spare cells.
        if len(alive) <= 1:
            logger.info(
                "Skipping injection: %d/%d cells alive (need >1 to keep redundancy)",
                len(alive),
                len(cells),
            )
            continue

        target = rng.choice(alive)
        cell_name = target["metadata"]["name"]
        mode = rng.choice(_FAILURE_MODES)

        try:
            resp = requests.post(
                f"{base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": 0},
                timeout=5,
            )
            resp.raise_for_status()
            last_injection_at = time.monotonic()
        except Exception:
            logger.info("Failed to inject fault into %s", cell_name, exc_info=True)


@app.command(name="run")
def run_ci(
    mode: Annotated[str, typer.Option(help="Test mode variant")],
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_steps: Annotated[int, typer.Option(help="Number of train() calls")] = 30,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = 0.1,
) -> None:
    """Random failure soak test.

    Starts a background thread that injects faults at random intervals via the
    control server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    dump_dir: str = resolve_dump_dir(f"{TEST_NAME}_{mode}")
    print(f"Dump directory: {dump_dir}")
    mean_interval: float = _MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection interval: {mean_interval:.1f}s")

    prepare(ft_mode)

    # The recorded debug rollouts are fewer than the soak's step count; symlink them cyclically
    # into a temp dir so each rollout_id has a file, keeping the production load path unchanged.
    cyclic_data_dir = materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=cyclic_data_dir)
        + get_ft_args(ft_mode)
        + f"--control-server-port {_CONTROL_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    stop_event, injector_thread = _spawn_fault_injector(seed=seed, mean_interval_seconds=mean_interval)

    try:
        run_training(train_args=train_args, mode=ft_mode)
    finally:
        stop_event.set()
        injector_thread.join(timeout=5)

    print(f"Random failure soak test PASSED (seed={seed}, steps={num_steps})")


@app.command(name="run-gsm8k")
def run_ci_gsm8k(
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_rollout: Annotated[int, typer.Option(help="Number of rollouts")] = 250,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = 0.1,
    metric_threshold: Annotated[
        float, typer.Option(help="eval/gsm8k accuracy threshold")
    ] = _GSM8K_DEFAULT_METRIC_THRESHOLD,
) -> None:
    """Random failure soak on the real gsm8k RL recipe, asserting eval accuracy.

    Same external fault injection as ``run``, but the workload is the recipe of
    tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py (whose regular CI runs serve as the
    no-fault reference wandb curves) with train-side fault tolerance. Besides
    surviving the crashes, the run must reach the eval/gsm8k accuracy threshold —
    i.e. fault recovery preserves end-to-end learning, which the comparison
    scenarios cannot observe.

    Doubles as the CI entry point: the CI file calls ``run_ci_gsm8k()`` (defaults);
    manual smoke/calibration runs use the ``run-gsm8k`` CLI subcommand.
    """
    mean_interval: float = _MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Rollouts: {num_rollout}, Mean injection interval: {mean_interval:.1f}s")

    _prepare_gsm8k()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    train_args = _get_gsm8k_train_args(seed=seed, num_rollout=num_rollout, metric_threshold=metric_threshold)

    stop_event, injector_thread = _spawn_fault_injector(seed=seed, mean_interval_seconds=mean_interval)

    try:
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=_GSM8K_TRAIN_GPUS + _GSM8K_ROLLOUT_GPUS,
            megatron_model_type=_GSM8K_MODEL_TYPE,
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
        stop_event.set()
        injector_thread.join(timeout=5)

    print(f"Random failure gsm8k accuracy test PASSED (seed={seed}, rollouts={num_rollout})")


def _spawn_fault_injector(*, seed: int, mean_interval_seconds: float) -> tuple[threading.Event, threading.Thread]:
    base_url = f"http://localhost:{_CONTROL_SERVER_PORT}"
    stop_event = threading.Event()
    injector_thread = threading.Thread(
        target=_run_fault_injection_loop,
        kwargs={
            "base_url": base_url,
            "seed": seed,
            "mean_interval_seconds": mean_interval_seconds,
            "stop_event": stop_event,
        },
        daemon=True,
        name="ft-random-fault-injector",
    )
    injector_thread.start()
    return stop_event, injector_thread


def _prepare_gsm8k() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{_GSM8K_MODEL_NAME} --local-dir /root/models/{_GSM8K_MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _get_gsm8k_train_args(*, seed: int, num_rollout: int, metric_threshold: float) -> str:
    ckpt_args = f"--hf-checkpoint /root/models/{_GSM8K_MODEL_NAME}/ " f"--ref-load /root/models/{_GSM8K_MODEL_NAME}/ "

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
        f"--rollout-num-gpus {_GSM8K_ROLLOUT_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-metrics "
    )

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--ft-components train "
        f"--control-server-port {_CONTROL_SERVER_PORT} "
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
        f"--actor-num-gpus-per-node {_GSM8K_TRAIN_GPUS} "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(f'test_{TEST_NAME}_gsm8k.py', run_name_prefix=f'seed{seed}')} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fault_tolerance_args} "
        f"{ci_args} "
        f"{misc_args} "
    )


if __name__ == "__main__":
    app()
