import asyncio
from collections.abc import Callable
from functools import partial
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.common.example_args import with_replaced_value
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    get_train_env_vars_arg,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode
from tests.utils.soak.core.config import POLL_INTERVAL_SECONDS, SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.utils import (
    API_SERVER_ARGS,
    assert_fresh_dump_dir,
    compute_release_of_config,
    evidence_directory,
    note_launch_outcome,
    resolve_dump_dir,
)
from tests.utils.soak.deploy.session import LauncherChain, follow_launchers
from tests.utils.soak.ft.actions.resize import ResizePoolForm
from tests.utils.soak.ft.checkers.resize import (
    assert_counts_follow_schedule,
    assert_resizes_follow_schedule,
    assert_schedule_fits,
)
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.types import POOL_TARGET_KIND, ResizeStep

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.comparisons.metrics import assert_gradients_nonzero
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

SCALING_NUM_ROLLOUTS: int = 12


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


SCALING_SCHEDULE: tuple[ResizeStep, ...] = (ResizeStep(at_rollout=2, replicas=3), ResizeStep(at_rollout=7, replicas=2))


def run_scaling_scenario(
    *,
    test_name: str,
    mode: FTTestMode,
    pool_id: str,
    cell_type: str,
    scaling_flag: str,
    num_gpus_per_cell: int,
    read_counts_of_rollout_id: Callable[[list[SoakEvent], Path], dict[int, int]],
    counted: str,
) -> None:
    config = command_utils.default_config()
    assert config.cluster_backend is ClusterBackend.KUBERNETES, (
        f"a scaling scenario resizes the LeaderWorkerSet a pool is deployed as, which only the "
        f"{ClusterBackend.KUBERNETES.value} backend has, and this environment declares the "
        f"{config.cluster_backend.value} backend"
    )
    assert config.namespace, "resizing a pool needs the namespace the run is installed into"
    initial_replicas = mode.cell_counts_of_type[cell_type]
    assert_schedule_fits(SCALING_SCHEDULE, initial_replicas=initial_replicas, num_rollouts=SCALING_NUM_ROLLOUTS)

    dump_dir = Path(resolve_dump_dir(test_name, run_id=config.run_id))
    print(f"Dump directory: {dump_dir}")
    prepare(mode, config=config)
    assert_fresh_dump_dir(dump_dir)

    train_args = (
        get_common_train_args(mode, dump_dir=str(dump_dir), num_steps=SCALING_NUM_ROLLOUTS, enable_dumper=False)
        + get_ft_args(mode, api_server_args=API_SERVER_ARGS)
        + "--mini-ft-controller-enable "
        + get_train_env_vars_arg(mode, deterministic=False)
    )

    evidence_dir = evidence_directory(dump_dir)
    event_log = EventLog(evidence_dir / "events.jsonl")
    chain = LauncherChain()
    injector = asyncio.run(
        run_cell_soak(
            config=config,
            dump_dir=dump_dir,
            sut_run=follow_launchers(
                note_launch_outcome(
                    event_log=event_log,
                    request_id=None,
                    launching=asyncio.to_thread(run_training, train_args=train_args, mode=mode, config=config),
                ),
                chain=chain,
            ),
            runner_config=SoakRunnerConfig(
                seed=0,
                target_configs={
                    POOL_TARGET_KIND: SoakTargetConfig(expected_count=1, mean_interval_seconds=POLL_INTERVAL_SECONDS)
                },
                tail=SoakTailConfig.create(num_rollout=SCALING_NUM_ROLLOUTS),
                quiescent_polls_required=1,
            ),
            event_log=event_log,
            evidence_dir=evidence_dir,
            forms={
                POOL_TARGET_KIND: [
                    ResizePoolForm(
                        namespace=config.namespace,
                        workload=component_name(compute_release_of_config(config), pool_id),
                        cell_type=cell_type,
                        schedule=SCALING_SCHEDULE,
                        relaunch_at_size=partial(
                            _relaunch_at_size,
                            train_args=train_args,
                            mode=mode,
                            config=config,
                            scaling_flag=scaling_flag,
                            num_gpus_per_cell=num_gpus_per_cell,
                        ),
                        event_log=event_log,
                        chain=chain,
                    )
                ]
            },
        )
    )
    events = injector.event_log.events

    assert_resizes_follow_schedule(events, schedule=SCALING_SCHEDULE, initial_replicas=initial_replicas)
    assert_gradients_nonzero(side=test_name, dump_dir=str(dump_dir), min_trained_rollouts=SCALING_NUM_ROLLOUTS)
    assert_counts_follow_schedule(
        read_counts_of_rollout_id(events, dump_dir),
        initial=initial_replicas,
        schedule=SCALING_SCHEDULE,
        num_rollouts=SCALING_NUM_ROLLOUTS,
        subject=counted,
    )
    print(f"Scaling test PASSED ({test_name}, rollouts={SCALING_NUM_ROLLOUTS}, schedule={SCALING_SCHEDULE})")


async def _relaunch_at_size(
    replicas: int,
    *,
    train_args: str,
    mode: FTTestMode,
    config: command_utils.ExecuteTrainConfig,
    scaling_flag: str,
    num_gpus_per_cell: int,
) -> None:
    await asyncio.to_thread(
        run_training,
        train_args=with_replaced_value(train_args, flag=scaling_flag, value=str(replicas * num_gpus_per_cell)),
        mode=mode,
        config=config,
    )
