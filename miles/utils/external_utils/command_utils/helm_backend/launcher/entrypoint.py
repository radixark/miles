from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.train import (
    TRAINER_CONTROLLER_ADDRS_FLAG,
    compute_trainer_controller_pool_id,
    compute_trainer_ids,
    specs_trainer_controller,
)
from miles.utils.arguments import parse_args
from miles.utils.external_utils.command_utils.base_backend import (
    CLUSTER_BACKEND_FLAG,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_INIT_KWARGS_FLAG,
    ArgvManipulator,
    chart_dir,
    repo_base_dir,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import manifest_diff
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import CI_LABEL, Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.hot_restart import (
    compute_orchestrator_object_key,
    plan_hot_restart,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.launch_record import (
    LaunchRecord,
    installed_launch_record_file,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest, ManifestObjectKey
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import farewell, with_observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.diagnosis import collect_diagnosis
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import pod_phase
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    InfraInfo,
    LaunchPlan,
    MooncakeInfo,
    MooncakePlan,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName, RunFiles, RunNames
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import wait_for_run
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.object_store import ObjectStoreBackend
from miles.utils.run_uuid import generate_run_uuid, validate_run_uuid
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import RPC_PORT_NAME

logger = logging.getLogger(__name__)

_RUN_UUID_FLAG = "--run-uuid"
_ENV_REPORT_FLAG = "--env-report"
_RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def execute_train(*, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None:
    run_id = config.run_id
    assert _RUN_ID_PATTERN.fullmatch(
        run_id
    ), f"run_id {run_id!r} names every object this run installs, so it has to match {_RUN_ID_PATTERN.pattern}"

    namespace = config.namespace
    release = ReleaseName(
        run_id=run_id,
        deploy_component=config.deploy_component,
        deploy_instance_id=config.deploy_instance_id,
    ).serialize()
    installed_manifest = Helm.get_manifest(release, namespace)
    run_uuid = _resolve_run_uuid(config, installed_manifest=installed_manifest, release=release)
    pod_argv, args = _compute_train_argv(request, run_uuid=run_uuid, release=release, namespace=namespace)
    deploy_component = DeployComponent(args.deploy_component)
    assert (deploy_component, args.deploy_instance_id) == (config.deploy_component, config.deploy_instance_id), (
        f"the run's pods are told {deploy_component.value}/{args.deploy_instance_id!r}, the release is named "
        f"{config.deploy_component.value}/{config.deploy_instance_id!r}"
    )
    deploys_orchestration_script = deploy_component.deploys_orchestration_script()

    specs = compute_specs(args)
    chart = chart_dir(repo_base_dir=repo_base_dir)
    shared_root = InfraInfo.shared_root(InfraInfo.load(chart, list(config.helm_values)), namespace=namespace)
    run_directory = RunFiles.run_dir(shared_root=shared_root, run_id=run_id)

    if config.ci_run:
        _uninstall_leftover_ci_releases(namespace, keep_run_id=run_id)
    Helm.build_dependencies(chart)

    orchestrator_command = ["python", request.train_script, *pod_argv] if deploys_orchestration_script else []
    hot_restart_plan = plan_hot_restart(
        components=config.parsed_hot_restart,
        deploy_component=deploy_component,
        release=release,
        installed_manifest=installed_manifest,
    )
    reachable_at = (
        _compute_trainer_controller_addrs(args, release=release, namespace=namespace)
        if deploy_component is DeployComponent.TRAINER
        else {}
    )
    values_path = RunFiles.new_values_file(run_directory=run_directory)
    values_files: list[str | Path] = [*config.helm_values, values_path]
    record_path = RunFiles.new_record_file(run_directory=run_directory)

    def render_and_propose(state_file: Path | None) -> tuple[LaunchRecord, Manifest]:
        plan = LaunchPlan(
            run_id=run_id,
            release=release,
            namespace=namespace,
            state_file=str(state_file) if state_file is not None else "",
            orchestrator_command=orchestrator_command,
            worker_argv=pod_argv,
            env=train_env_vars(request, {}, config=config),
            colocate=bool(args.colocate),
            mooncake_plan=_compute_mooncake_plan(args),
            prepare_cmd=request.prepare_cmd,
            extra_manifests=request.extra_manifests,
            restart_at=hot_restart_plan.restart_at,
            stamped_components=hot_restart_plan.stamped_components,
        )
        computed = LaunchRecord.compute(plan=plan, values_file=values_path, reachable_at=reachable_at)
        rendered = plan.model_copy(
            update={
                "launch_record": _compute_pod_record_file(
                    installed_manifest=installed_manifest, record_path=record_path
                ),
            }
        )
        _write_helm_values(values_path, build_values(specs, rendered).as_values())
        return computed, Helm.render_upgrade(
            release=release, namespace=namespace, chart=chart, values_files=values_files
        )

    carried_state_file = (
        _carried_state_file(installed_manifest=installed_manifest, release=release)
        if deploys_orchestration_script
        else None
    )

    rebuilds_orchestrator = True
    if installed_manifest is not None:
        _, carried_manifest = render_and_propose(carried_state_file)
        carried_diff = manifest_diff.diff_manifests(
            before=installed_manifest,
            after=carried_manifest,
            allow_diff_object_keys=hot_restart_plan.allow_diff_object_keys,
        )
        rebuilds_orchestrator = carried_diff.rebuilds(key=compute_orchestrator_object_key(release))
    state_file = (
        RunFiles.new_state_file(run_directory=run_directory)
        if rebuilds_orchestrator and deploys_orchestration_script
        else carried_state_file
    )
    record, proposed_manifest = render_and_propose(state_file)

    if installed_manifest is not None:
        _assert_upgrade_is_allowed(
            diff=(
                carried_diff
                if state_file == carried_state_file
                else manifest_diff.diff_manifests(
                    before=installed_manifest,
                    after=proposed_manifest,
                    allow_diff_object_keys=hot_restart_plan.allow_diff_object_keys,
                )
            ),
            release=release,
            skip_upgrade_check=config.skip_upgrade_check,
            allow_diff_object_keys=hot_restart_plan.allow_diff_object_keys,
        )
    if rebuilds_orchestrator:
        _remove_pending_uninstall(release, namespace=namespace)

    record.write(path=record_path)
    logger.info(f"What this launch launched is recorded under {record_path}")

    Helm.upgrade(
        release=release,
        namespace=namespace,
        chart=chart,
        values_files=values_files,
        ci_run=config.ci_run,
    )

    if not deploys_orchestration_script:
        logger.info(
            f"Installed {release}, which carries no orchestration script: it has no training to finish, so it stays "
            f"up until you uninstall it with `helm uninstall {release} --namespace {namespace}`"
        )
        if reachable_at:
            logger.info(f"Reach it with {_describe_trainer_controller_addrs(reachable_at)}")
        return

    if deploy_component.is_split():
        logger.info(
            f"The other deployments of this run share the object store of {release}, so give each of them "
            f"{_describe_shared_object_store(_compute_mooncake_plan(args), release=release, namespace=namespace)}"
        )

    _follow_until_finished(release=release, namespace=namespace, state_file=state_file)


def _compute_trainer_controller_addrs(args: Any, *, release: str, namespace: str) -> dict[str, str]:
    specs_by_pool_id = {spec.name: spec for spec in specs_trainer_controller(args)}
    addrs = {}
    for trainer_id in compute_trainer_ids(args):
        spec = specs_by_pool_id[compute_trainer_controller_pool_id(trainer_id)]
        addr = static_cell_addrs(spec=spec, release=release, cell_index=0)[RPC_PORT_NAME]
        host = RunNames.service_fqdn(name=addr.host, namespace=namespace)
        addrs[trainer_id] = f"{host}:{addr.port}"
    return addrs


def _describe_trainer_controller_addrs(reachable_at: dict[str, str]) -> str:
    entries = " ".join(f"{trainer_id}={addr}" for trainer_id, addr in reachable_at.items())
    return f"{TRAINER_CONTROLLER_ADDRS_FLAG} {entries}"


def _describe_shared_object_store(plan: MooncakePlan | None, *, release: str, namespace: str) -> str:
    assert plan is not None, (
        f"{release} carries the orchestration script of a split run, so it runs the object store master the other "
        f"deployments redeem their references at, and a run without one shares nothing"
    )
    init_kwargs = MooncakeInfo.cluster_init_kwargs(plan, host=MooncakeInfo.master_service_host(release, namespace))
    return (
        f"--object-store-backend {ObjectStoreBackend.MOONCAKE.value} "
        f"{MOONCAKE_INIT_KWARGS_FLAG} {shlex.quote(json.dumps(init_kwargs))}"
    )


def _follow_until_finished(*, release: str, namespace: str, state_file: Path) -> None:
    logger.info(f"Following every pod of {release}; ctrl+c stops watching, not the run")
    orchestrator_workload = naming.component_name(release, naming.ORCHESTRATOR_COMPONENT)

    with with_observability(namespace=namespace, selector=Kubectl.release_selector(release)):
        outcome = wait_for_run(
            state_file=state_file,
            read_pod_phase=lambda: pod_phase(namespace, orchestrator_workload),
        )

    if outcome.exit_code != 0:
        _collect_diagnosis(release=release, namespace=namespace, state_file=state_file)

    logger.info(farewell(namespace=namespace, release=release, workload=orchestrator_workload))
    if outcome.exit_code != 0:
        raise SystemExit(outcome.exit_code)


def _resolve_run_uuid(config: ExecuteTrainConfig, *, installed_manifest: Manifest | None, release: str) -> str:
    if (given := config.run_uuid) is not None:
        return validate_run_uuid(given)

    assert not config.deploy_component.is_split(), (
        f"--deploy-component {config.deploy_component.value} installs one part of a run whose other parts are "
        f"installed by other launches, and they are joined by nothing but the run uuid, so the layer that deploys "
        f"them all has to name it with --run-uuid"
    )

    if installed_manifest is not None:
        installed = installed_manifest.flag_value(
            _RUN_UUID_FLAG,
            stateful_set=RunNames.orchestrator_object(release=release),
            container=naming.ORCHESTRATOR_COMPONENT,
        )
        if installed is not None:
            return installed

    return generate_run_uuid()


def _compute_train_argv(
    request: ExecuteTrainRequest, *, run_uuid: str, release: str, namespace: str
) -> tuple[list[str], Any]:
    argv = [*shlex.split(shell_safe_model_args(request.megatron_model_type)), *shlex.split(request.train_args)]
    assert not ArgvManipulator.is_defined(argv, _ENV_REPORT_FLAG), (
        f"{_ENV_REPORT_FLAG} is what this launcher tells the pods about the launch that installed them, and an "
        f"argument of that name outranks it, so the pods would report a launch that never happened; drop it"
    )
    if not ArgvManipulator.is_defined(argv, CLUSTER_BACKEND_FLAG):
        argv = ArgvManipulator.set(argv, CLUSTER_BACKEND_FLAG, ClusterBackend.KUBERNETES.value)
    if not ArgvManipulator.is_defined(argv, _RUN_UUID_FLAG):
        argv = ArgvManipulator.set(argv, _RUN_UUID_FLAG, run_uuid)

    with override_argv(argv):
        args = parse_args()

    pod_argv = MooncakeInfo.with_cluster_master(
        argv, plan=_compute_mooncake_plan(args), host=MooncakeInfo.master_service_host(release, namespace)
    )
    return pod_argv, args


def _compute_mooncake_plan(args) -> MooncakePlan | None:
    if not DeployComponent(args.deploy_component).deploys_orchestration_script():
        return None
    return MooncakeInfo.plan_of_args(args)


def _compute_pod_record_file(*, installed_manifest: Manifest | None, record_path: Path) -> str | None:
    if installed_manifest is None:
        return str(record_path)
    return installed_launch_record_file(manifest=installed_manifest)


def _carried_state_file(*, installed_manifest: Manifest | None, release: str) -> Path | None:
    if installed_manifest is None:
        return None

    attached_state_file = installed_manifest.state_file(
        stateful_set=RunNames.orchestrator_object(release=release), container=naming.ORCHESTRATOR_COMPONENT
    )
    assert attached_state_file is not None, (
        f"Run {release} is installed but its orchestrator names no state file, so this launch cannot tell what it "
        f"is watching; uninstall it, or launch under a new run id"
    )
    return attached_state_file


def _assert_upgrade_is_allowed(
    *,
    diff: manifest_diff.ManifestDiffs,
    release: str,
    skip_upgrade_check: bool,
    allow_diff_object_keys: frozenset[ManifestObjectKey],
) -> None:
    if diff.is_allowed:
        logger.info(
            f"Run {release} already exists; upgrading it with these allowed changes:\n{diff.summarize_allowed_changes()}"
        )
        return

    allowed = ", ".join(sorted(f"{key.kind}/{key.name}" for key in allow_diff_object_keys))
    rebuildable = f", and more than the objects this launch is allowed to rebuild ({allowed})" if allowed else ""
    message = (
        f"Run {release} already exists and the relaunch would change more than its size{rebuildable}:\n"
        f"{diff.describe()}\n"
        f"launch under a new run id, or pass --skip-upgrade-check to apply this anyway and accept the restarts"
    )
    if not skip_upgrade_check:
        raise SystemExit(message)
    logger.warning(f"upgrade check skipped: {message}")


def _belongs_to_run(release: str, *, run_id: str) -> bool:
    return (parsed := ReleaseName.parse(release)) is not None and parsed.run_id == run_id


def _uninstall_leftover_ci_releases(namespace: str, *, keep_run_id: str) -> list[str]:
    listed = Helm.list_releases(namespace=namespace, selector=f"{CI_LABEL}=true")
    releases = [release for release in listed if not _belongs_to_run(release, run_id=keep_run_id)]
    for release in releases:
        logger.info(f"Uninstalling the leftover ci release {release} before this run installs its own")
        Helm.uninstall(release=release, namespace=namespace)
    return releases


def _remove_pending_uninstall(release: str, *, namespace: str) -> None:
    job = RunNames.uninstall_job(release=release)
    logger.info(f"Deleting {job} if it is pending, so it cannot uninstall the release this launch installs")
    Kubectl.delete_job(job, namespace=namespace, check=True)


def _collect_diagnosis(*, release: str, namespace: str, state_file: Path) -> None:
    releases = _releases_of_run(release=release, namespace=namespace)
    if len(releases) > 1:
        logger.info(
            f"This run is deployed as {len(releases)} releases, and a deployment other than {release} can be what "
            f"failed, so the diagnosis covers all of them: {', '.join(releases)}"
        )

    try:
        diagnosis = collect_diagnosis(
            namespace=namespace,
            output_dir=state_file.parent,
            selector=Kubectl.releases_selector(releases),
            state_file=state_file,
        )
    except Exception:
        logger.warning("Could not collect a diagnosis of the failed run", exc_info=True)
        return

    logger.info(f"The pods of this failed run are described under {diagnosis.directory}")
    if not diagnosis.is_complete:
        logger.warning(f"The diagnosis is incomplete, these could not be collected: {', '.join(diagnosis.missing)}")


def _releases_of_run(*, release: str, namespace: str) -> list[str]:
    parsed = ReleaseName.parse(release)
    if parsed is None:
        return [release]

    try:
        listed = Helm.list_releases(namespace=namespace)
    except Exception:
        logger.warning(f"Could not list what else run {parsed.run_id} installed; diagnosing {release} alone")
        logger.debug("Listing the releases of the run failed", exc_info=True)
        return [release]

    return sorted({one for one in listed if _belongs_to_run(one, run_id=parsed.run_id)} | {release})


def _write_helm_values(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(values, Dumper=_HelmValuesDumper, default_flow_style=False, sort_keys=True))


class _HelmValuesDumper(yaml.SafeDumper):
    def represent_mapping(self, tag: str, mapping: Any, flow_style: bool | None = None) -> yaml.MappingNode:
        node = super().represent_mapping(tag, mapping, flow_style=flow_style)
        for key, _ in node.value:
            key.style = None
        return node


def _represent_helm_value_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="'")


_HelmValuesDumper.add_representer(str, _represent_helm_value_str)
