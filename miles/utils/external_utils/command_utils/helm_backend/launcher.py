from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import (
    elastic,
    helm,
    kube,
    mooncake,
    naming,
    observe,
    run_state,
    watch,
)
from miles.utils.external_utils.command_utils.helm_backend.values import RunLayout, build_values
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_spec import BaseWorkerSpec

_LOG_STREAM_JOIN_SECONDS = 5.0


class LaunchedRun(FrozenStrictBaseModel):
    release: str
    namespace: str
    exit_file: Path
    generation: int = 0

    @property
    def orchestrator_workload(self) -> str:
        return naming.component_name(self.release, "orchestrator")


def launch(
    request: ExecuteTrainRequest,
    *,
    specs: list[BaseWorkerSpec],
    run_id: str,
    namespace: str,
    infra_values_files: list[str],
    repo_base_dir: str | Path,
    train_argv: list[str],
    colocate: bool = False,
    shared_root: str = "",
    stage_to_local: tuple[str, ...] = (),
    node_local_root: str = "",
    force: bool = False,
    ci_run: bool = False,
) -> LaunchedRun:
    release = naming.release_name(run_id)
    infra_values = merged_infra_values(infra_values_files)
    resolved_root = resolve_shared_root(infra_values, override=shared_root)
    run_directory = run_state.run_dir(resolved_root, run_id)
    exit_file = run_state.orchestrator_exit_path(run_directory)

    pod_argv = mooncake.with_cluster_master(train_argv, mooncake.master_service_host(release, namespace))
    values = build_values(
        specs,
        RunLayout(
            run_id=run_id,
            release=release,
            orchestrator_command=orchestrator_command(request, pod_argv),
            worker_argv=pod_argv,
            env=runtime_env(request),
            colocate=colocate,
            uses_mooncake=mooncake.uses_mooncake(train_argv),
            mooncake_port=mooncake.master_port_of(train_argv, default_port=0),
            stage_to_local=stage_to_local,
            node_local_root=node_local_root,
        ),
    )

    run_values_file = run_state.values_path(run_directory)
    write_values(run_values_file, values)

    if ci_run:
        helm.uninstall_ci_releases(namespace)

    chart = helm.chart_dir(repo_base_dir)
    helm.run(helm.dependency_build_command(chart))
    values_files: list[str | Path] = [*infra_values_files, run_values_file]
    baseline = run_state.current_generation(exit_file)
    if helm.release_exists(release, namespace):
        _check_upgrade_only_scales(
            release=release,
            namespace=namespace,
            chart=chart,
            values_files=values_files,
            proposed=_deep_merge(infra_values, values),
            force=force,
        )
    helm.run(
        helm.upgrade_command(
            release=release,
            namespace=namespace,
            chart=chart,
            values_files=values_files,
            ci_run=ci_run,
        )
    )
    generation = run_state.reset_for_new_generation(exit_file, baseline)

    return LaunchedRun(release=release, namespace=namespace, exit_file=exit_file, generation=generation)


def runtime_env(request: ExecuteTrainRequest) -> dict[str, str]:
    from miles.utils.external_utils.command_utils.common import build_train_env_vars

    return build_train_env_vars(request, {})


def resolve_shared_root(infra_values: dict[str, Any], *, override: str = "") -> str:
    derived = run_state.shared_root_of(infra_values)
    if not override:
        return derived
    assert override == derived, (
        f"ExecuteTrainConfig.shared_root is {override!r} but the infra values put the run directory under "
        f"{derived!r}; the launcher and the pods would then read and write different files, so set one or the other"
    )
    return derived


def write_values(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_values(values))


def dump_values(values: dict[str, Any]) -> str:
    return yaml.safe_dump(values, default_flow_style=False, sort_keys=True)


def merged_infra_values(infra_values_files: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for values_file in infra_values_files:
        merged = _deep_merge(merged, yaml.safe_load(Path(values_file).read_text()) or {})
    return merged


def merged_values(infra_values_files: list[str], run_values: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(merged_infra_values(infra_values_files), run_values)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def installed_values(release: str, namespace: str) -> dict[str, Any]:
    listed = helm.run(
        ["helm", "get", "values", release, "--namespace", namespace, "--output", "json"], capture_output=True
    )
    return json.loads(listed.stdout or "null") or {}


def _check_upgrade_only_scales(
    *,
    release: str,
    namespace: str,
    chart: Path,
    values_files: list[str | Path],
    proposed: dict[str, Any],
    force: bool,
) -> None:
    values_diff = elastic.diff_values(installed_values(release, namespace), proposed)
    manifest_diff = _manifest_diff(release=release, namespace=namespace, chart=chart, values_files=values_files)

    if values_diff.is_allowed and manifest_diff.is_allowed:
        print(f"[launcher] run {release} already exists; upgrading it:", flush=True)
        print(manifest_diff.summarize_scaling(), flush=True)
        return

    message = (
        f"run {release} already exists and the relaunch would change more than its size:\n"
        f"  values:\n{values_diff.describe()}\n"
        f"  manifests:\n{manifest_diff.describe()}\n"
        f"launch under a new run id, or pass force=True to apply this anyway and accept the restarts"
    )
    if not force:
        raise SystemExit(message)
    print(f"[launcher] forced: {message}", flush=True)


def _manifest_diff(
    *, release: str, namespace: str, chart: Path, values_files: list[str | Path]
) -> elastic.ManifestDiff:
    installed = helm.run(["helm", "get", "manifest", release, "--namespace", namespace], capture_output=True).stdout
    proposed = helm.run(
        helm.upgrade_command(
            release=release, namespace=namespace, chart=chart, values_files=values_files, dry_run=True
        ),
        capture_output=True,
    ).stdout
    return elastic.diff_manifests(installed, elastic.manifest_of(proposed))


def follow_until_finished(run: LaunchedRun, log: Callable[[str], None] = print) -> int:
    _report_startup(run, log=log)

    log(f"[launcher] following {run.orchestrator_workload}; ctrl+c stops watching, not the run")
    stream = LogStream(run)
    stream.start()
    try:
        outcome = watch.wait_for_run(
            exit_file=run.exit_file,
            read_pod_phase=lambda: kube.pod_phase(run.namespace, run.orchestrator_workload),
            log=log,
            min_generation=run.generation,
        )
    finally:
        stream.stop()

    log(observe.farewell(namespace=run.namespace, release=run.release, workload=run.orchestrator_workload))
    return outcome.exit_code


class LogStream:
    def __init__(self, run: LaunchedRun) -> None:
        self._command = observe.follow_log_command(namespace=run.namespace, workload=run.orchestrator_workload)
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._stream, daemon=True)
        self._thread.start()

    def _stream(self) -> None:
        try:
            self._process = subprocess.Popen(self._command)
            self._process.wait()
        except KeyboardInterrupt:
            print("[launcher] stopped following the log; the run continues", flush=True)
        except OSError as error:
            print(f"[launcher] could not follow the log ({error}); polling the exit file instead", flush=True)

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_LOG_STREAM_JOIN_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._thread is not None:
            self._thread.join(timeout=_LOG_STREAM_JOIN_SECONDS)


def _report_startup(run: LaunchedRun, log: Callable[[str], None], budget_seconds: float = 300.0) -> None:
    deadline = time.monotonic() + budget_seconds
    last_summary = None
    while time.monotonic() < deadline:
        pods = kube.release_pods(run.namespace, run.release)
        events = kube.pod_events(namespace=run.namespace, pods=pods)
        if (summary := observe.startup_summary(pods, events)) != last_summary:
            log(f"[launcher] {summary}")
            last_summary = summary
        if hint := observe.scale_hint(pods):
            log(f"[launcher] {hint}")
            return
        if observe.is_settled(pods):
            return
        time.sleep(watch.POLL_INTERVAL_SECONDS)


def orchestrator_command(request: ExecuteTrainRequest, train_argv: list[str]) -> list[str]:
    return ["python", request.train_script, *train_argv]
