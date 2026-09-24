import asyncio
from collections.abc import Callable
from pathlib import Path

from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    get_train_env_vars_arg,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode
from tests.utils.cluster_backends import create_backend_for_run
from tests.utils.soak.core.config import POLL_INTERVAL_SECONDS, SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.utils import (
    API_SERVER_ARGS,
    compute_release_of_config,
    evidence_directory,
    resolve_dump_dir,
)
from tests.utils.soak.ft.actions.resize import ResizePoolForm, ScalingStep, assert_schedule_leaves_room
from tests.utils.soak.ft.checkers.resize import assert_observed_cells, assert_resizes_follow_schedule
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, POOL_TARGET_KIND, ROLLOUT_CELL_TYPE, CellTarget, Moment

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.test_utils.comparisons.metrics import assert_gradients_nonzero
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, LEADER_WORKER_SET_KIND
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

SCALING_NUM_ROLLOUTS: int = 12

_COUNTS_OF_CELL_TYPE: dict[str, Callable[[CellTarget], bool]] = {
    ROLLOUT_CELL_TYPE: lambda target: target.ready,
    ACTOR_CELL_TYPE: lambda target: target.alive,
}


def compute_scaling_mode(ft_components: tuple[str, ...]) -> FTTestMode:
    return FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=2,
        rollout_gpus_per_engine=1,
        ft_components=ft_components,
        parallel_args="--context-parallel-size 2",
    )


def compute_scaling_schedule(moment: Moment) -> tuple[ScalingStep, ...]:
    return (
        ScalingStep(at_rollout=2, replicas=3, moment=moment),
        ScalingStep(at_rollout=7, replicas=2, moment=moment),
    )


def run_scaling_scenario(
    *,
    test_name: str,
    mode: FTTestMode,
    schedule: tuple[ScalingStep, ...],
    pool_id: str,
    cell_type: str,
    assert_scaled: Callable[[list[SoakEvent], str], None],
) -> None:
    config = command_utils.default_config()
    assert config.cluster_backend is ClusterBackend.KUBERNETES, (
        f"a scaling scenario resizes the LeaderWorkerSet a pool is deployed as, which only the "
        f"{ClusterBackend.KUBERNETES.value} backend has, and this environment declares the "
        f"{config.cluster_backend.value} backend"
    )
    assert config.namespace, "resizing a pool needs the namespace the run is installed into"
    _assert_may_patch_pools(namespace=config.namespace)
    create_backend_for_run(config)
    initial_replicas = {ROLLOUT_CELL_TYPE: mode.rollout_num_engines, ACTOR_CELL_TYPE: mode.num_cells}[cell_type]
    assert_schedule_leaves_room(schedule, initial_replicas=initial_replicas, num_rollouts=SCALING_NUM_ROLLOUTS)

    dump_dir: str = resolve_dump_dir(test_name, run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    prepare(mode, config=config)

    train_args = (
        get_common_train_args(mode, dump_dir=dump_dir, num_steps=SCALING_NUM_ROLLOUTS, enable_dumper=False)
        + get_ft_args(mode, api_server_args=API_SERVER_ARGS)
        + "--mini-ft-controller-enable "
        + get_train_env_vars_arg(mode, deterministic=False)
    )

    sizes = [initial_replicas, *(step.replicas for step in schedule)]
    evidence_dir = evidence_directory(Path(dump_dir))
    injector = asyncio.run(
        run_cell_soak(
            config=config,
            dump_dir=Path(dump_dir),
            sut_run=asyncio.to_thread(
                run_training, train_args=train_args, mode=mode, dump_dir=dump_dir, config=config
            ),
            runner_config=SoakRunnerConfig(
                seed=0,
                target_configs={
                    POOL_TARGET_KIND: SoakTargetConfig(expected_count=1, mean_interval_seconds=POLL_INTERVAL_SECONDS)
                },
                tail=SoakTailConfig.create(num_rollout=SCALING_NUM_ROLLOUTS),
                quiescent_polls_required=1,
            ),
            event_log=EventLog(evidence_dir / "events.jsonl"),
            evidence_dir=evidence_dir,
            forms={
                POOL_TARGET_KIND: [
                    ResizePoolForm(
                        namespace=config.namespace,
                        workload=component_name(compute_release_of_config(config), pool_id),
                        cell_type=cell_type,
                        schedule=schedule,
                        min_replicas=min(sizes),
                        max_replicas=max(sizes),
                    )
                ]
            },
        )
    )
    events = injector.event_log.events

    assert_resizes_follow_schedule(events, schedule=schedule, initial_replicas=initial_replicas)
    assert_observed_cells(
        events,
        cell_type=cell_type,
        initial=initial_replicas,
        schedule=schedule,
        counts=_COUNTS_OF_CELL_TYPE[cell_type],
    )
    assert_gradients_nonzero(side=test_name, dump_dir=dump_dir, min_trained_rollouts=SCALING_NUM_ROLLOUTS)
    assert_scaled(events, dump_dir)
    print(f"Scaling test PASSED ({test_name}, rollouts={SCALING_NUM_ROLLOUTS}, schedule={schedule})")


def _assert_may_patch_pools(*, namespace: str) -> None:
    result = run_process(
        ["kubectl", "auth", "can-i", "patch", LEADER_WORKER_SET_KIND, "--namespace", namespace],
        capture_output=True,
        check=False,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    assert result.stdout.strip() == "yes", (
        f"this account may not patch {LEADER_WORKER_SET_KIND} in namespace {namespace}, so no pool of the "
        f"run can be resized: {result.stderr.strip() or result.stdout.strip()}"
    )
