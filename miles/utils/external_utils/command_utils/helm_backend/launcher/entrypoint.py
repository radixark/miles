from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args
from miles.utils.external_utils.command_utils.base_backend import (
    CLUSTER_BACKEND_FLAG,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import ArgvManipulator, chart_dir, repo_base_dir, train_env_vars
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import pod_phase
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import InfraInfo, LaunchPlan
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, RunNames
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import wait_for_run
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.run_uuid import derive_run_uuid
from miles.utils.workers.serving.utils import override_argv, override_env
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)

_RUN_UUID_FLAG = "--run-uuid"
_WANDB_RUN_ID_FLAG = "--wandb-run-id"
_RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def execute_train(*, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None:
    run_id = config.run_id
    assert _RUN_ID_PATTERN.fullmatch(
        run_id
    ), f"run_id {run_id!r} names every object this run installs, so it has to match {_RUN_ID_PATTERN.pattern}"

    namespace = config.namespace
    release = RunNames.release(run_id=run_id)
    env = train_env_vars(request, {}, config=config)
    pod_argv, args = _compute_train_argv(request, run_id=run_id, env=env)

    specs = compute_specs(args)
    chart = chart_dir(repo_base_dir=repo_base_dir)
    shared_root = InfraInfo.shared_root(InfraInfo.load(chart, list(config.helm_values)))
    run_directory = RunFiles.run_dir(shared_root=shared_root, run_id=run_id)

    Helm.build_dependencies(chart)

    state_file = RunFiles.new_state_file(run_directory=run_directory)

    plan = LaunchPlan(
        run_id=run_id,
        release=release,
        namespace=namespace,
        state_file=str(state_file),
        orchestrator_command=["python", request.train_script, *pod_argv],
        worker_argv=pod_argv,
        env=env,
        prepare_cmd=request.prepare_cmd,
    )
    values_path = RunFiles.new_values_file(run_directory=run_directory)
    _write_helm_values(values_path, build_values(specs, plan).as_values())
    values_files: list[str | Path] = [*config.helm_values, values_path]

    Helm.upgrade(
        release=release,
        namespace=namespace,
        chart=chart,
        values_files=values_files,
    )

    _follow_until_finished(release=release, namespace=namespace, state_file=state_file)


def _follow_until_finished(*, release: str, namespace: str, state_file: Path) -> None:
    logger.info(f"Waiting for {release} to report its verdict; ctrl+c stops watching, not the run")
    orchestrator_workload = naming.component_name(release, naming.ORCHESTRATOR_COMPONENT)

    outcome = wait_for_run(
        state_file=state_file,
        read_pod_phase=lambda: pod_phase(namespace, orchestrator_workload),
    )

    if outcome.exit_code != 0:
        raise SystemExit(outcome.exit_code)


def _compute_train_argv(request: ExecuteTrainRequest, *, run_id: str, env: dict[str, str]) -> tuple[list[str], Any]:
    argv = [*shlex.split(shell_safe_model_args(request.megatron_model_type)), *shlex.split(request.train_args)]
    argv = ArgvManipulator.with_flag(argv, CLUSTER_BACKEND_FLAG, ClusterBackend.KUBERNETES.value)
    # TODO: generate different run_uuid even for same run_id, but at the same time allow helm upgrading
    argv = ArgvManipulator.with_flag(argv, _RUN_UUID_FLAG, derive_run_uuid(run_id))

    with override_argv(argv), override_env(env):
        args = parse_args()

    # TODO: remove after args refactor handles wandb ids
    if args.use_wandb and args.wandb_run_id is None:
        args.wandb_run_id = _generate_wandb_run_id()
        argv = ArgvManipulator.with_flag(argv, _WANDB_RUN_ID_FLAG, args.wandb_run_id)

    return argv, args


def _generate_wandb_run_id() -> str:
    from wandb.sdk.lib.runid import generate_id

    return generate_id()


def _write_helm_values(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, default_flow_style=False, sort_keys=True))
