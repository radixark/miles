from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Callable

from miles.utils.external_utils.command_utils.common import run_shell_command, substitute_placeholders
from miles.utils.external_utils.command_utils.helm_backend import helm, naming
from miles.utils.pydantic_utils import FrozenStrictBaseModel

_JOB_TEMPLATE = "templates/adhoc-job.yaml"
_JOB_NAME_LABEL = "batch.kubernetes.io/job-name"
_COMPLETION_INDEX_KEY = "batch.kubernetes.io/job-completion-index"
_TERMINAL_LOG_LINES = 200
_POLL_INTERVAL_SECONDS = 5.0
_TIMEOUT_SECONDS = 3 * 60 * 60


class AdhocContext(FrozenStrictBaseModel):
    namespace: str
    chart_dir: str
    infra_values_files: tuple[str, ...] = ()
    release: str = "miles-run-adhoc"
    gpus_per_node: int
    timeout_seconds: float = _TIMEOUT_SECONDS


def run_locally(cmd: str, capture_output: bool = False) -> str | None:
    return run_shell_command(cmd, capture_output=capture_output)


def run_on_one_gpu_node(context: AdhocContext, cmd: str, capture_output: bool = False) -> str | None:
    return run_on_nodes(
        context, cmd, capture_output=capture_output, completions=1, gpus_per_pod=context.gpus_per_node, step="gpu"
    )[0]


def run_on_nodes(
    context: AdhocContext,
    cmd: str,
    *,
    capture_output: bool,
    completions: int,
    gpus_per_pod: int,
    step: str,
) -> list[str | None]:
    prepared = substitute_placeholders(
        cmd,
        node_rank="${JOB_COMPLETION_INDEX}",
        nnodes=str(completions),
        master_addr=master_address(context.release, step, context.namespace),
        node_ip="$(hostname -i)",
    )
    return run_job(
        command=["bash", "-c", prepared],
        namespace=context.namespace,
        chart_dir=context.chart_dir,
        infra_values_files=list(context.infra_values_files),
        release=context.release,
        step=step,
        completions=completions,
        gpus_per_pod=gpus_per_pod,
        capture_output=capture_output,
        timeout_seconds=context.timeout_seconds,
        poll_interval_seconds=_POLL_INTERVAL_SECONDS,
        sleep=time.sleep,
    )


def job_object_name(release: str, step: str) -> str:
    return naming.component_name(release, step)


def master_address(release: str, step: str, namespace: str) -> str:
    name = job_object_name(release, step)
    return f"{name}-0.{name}.{namespace}.svc.cluster.local"


def render_job(
    *,
    command: list[str],
    namespace: str,
    chart_dir: str,
    infra_values_files: list[str],
    release: str,
    step: str,
    completions: int,
    gpus_per_pod: int,
    active_deadline_seconds: int,
) -> str:
    arguments = [
        "helm",
        "template",
        release,
        chart_dir,
        "--namespace",
        namespace,
        "--show-only",
        _JOB_TEMPLATE,
        "--set",
        "adhoc.enabled=true",
        "--set",
        f"adhoc.name={step}",
        "--set",
        f"adhoc.objectName={job_object_name(release, step)}",
        "--set",
        f"adhoc.completions={completions}",
        "--set",
        f"adhoc.gpusPerPod={gpus_per_pod}",
        "--set",
        f"adhoc.activeDeadlineSeconds={active_deadline_seconds}",
        "--set-json",
        f"adhoc.command={json.dumps(command)}",
        "--set",
        "run.id=adhoc",
    ]
    for values_file in infra_values_files:
        arguments += ["--values", values_file]

    return helm.run(arguments, capture_output=True).stdout


def run_job(
    *,
    command: list[str],
    namespace: str,
    chart_dir: str,
    infra_values_files: list[str],
    release: str,
    step: str,
    completions: int,
    gpus_per_pod: int,
    capture_output: bool,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None],
    kubectl: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> list[str | None]:
    kubectl = kubectl or _kubectl
    manifest = render_job(
        command=command,
        namespace=namespace,
        chart_dir=chart_dir,
        infra_values_files=infra_values_files,
        release=release,
        step=step,
        completions=completions,
        gpus_per_pod=gpus_per_pod,
        active_deadline_seconds=int(timeout_seconds),
    )

    kubectl(["delete", "job", job_object_name(release, step), "--namespace", namespace, "--ignore-not-found"])
    _apply(manifest, namespace=namespace, kubectl=kubectl)

    outcome = _wait(
        release=release,
        step=step,
        namespace=namespace,
        kubectl=kubectl,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )
    logs = _logs_per_completion(
        release=release, step=step, namespace=namespace, kubectl=kubectl, completions=completions
    )

    if outcome != "complete":
        raise RuntimeError(f"job {job_object_name(release, step)} {outcome}; last log lines:\n{_joined(logs)}")

    kubectl(["delete", "job", job_object_name(release, step), "--namespace", namespace, "--ignore-not-found"])
    return [log if capture_output else None for log in logs]


def _apply(manifest: str, namespace: str, kubectl: Callable) -> None:
    print(f"EXEC: kubectl apply -f - -n {namespace}", flush=True)
    result = subprocess.run(
        ["kubectl", "apply", "--namespace", namespace, "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"could not submit the job: {result.stderr}"


def _wait(
    *,
    release: str,
    step: str,
    namespace: str,
    kubectl: Callable,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None],
) -> str:
    waited = 0.0
    while waited < timeout_seconds:
        status = _job_status(release=release, step=step, namespace=namespace, kubectl=kubectl)
        if status in ("complete", "failed"):
            return status
        sleep(poll_interval_seconds)
        waited += poll_interval_seconds
    return f"did not finish within {timeout_seconds:.0f}s"


def _job_status(*, release: str, step: str, namespace: str, kubectl: Callable) -> str:
    result = kubectl(
        [
            "get",
            "job",
            job_object_name(release, step),
            "--namespace",
            namespace,
            "--output",
            "json",
            "--ignore-not-found",
        ]
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "pending"

    conditions = (json.loads(result.stdout).get("status") or {}).get("conditions") or []
    for condition in conditions:
        if condition.get("status") != "True":
            continue
        if condition.get("type") in ("Complete", "SuccessCriteriaMet"):
            return "complete"
        if condition.get("type") in ("Failed", "FailureTarget"):
            return "failed"
    return "running"


def _logs_per_completion(*, release: str, step: str, namespace: str, kubectl: Callable, completions: int) -> list[str]:
    pods = _pods_by_completion_index(release=release, step=step, namespace=namespace, kubectl=kubectl)
    if not pods:
        return [_logs_of(f"job/{job_object_name(release, step)}", namespace=namespace, kubectl=kubectl)] * completions

    logs = {index: _logs_of(name, namespace=namespace, kubectl=kubectl) for index, name in pods}
    return [logs.get(index, f"no pod of this job reported completion index {index}") for index in range(completions)]


def _pods_by_completion_index(*, release: str, step: str, namespace: str, kubectl: Callable) -> list[tuple[int, str]]:
    result = kubectl(
        [
            "get",
            "pods",
            "--namespace",
            namespace,
            "--selector",
            f"{_JOB_NAME_LABEL}={job_object_name(release, step)}",
            "--output",
            "json",
            "--ignore-not-found",
        ]
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    pods = [(_completion_index_of(pod), pod["metadata"]["name"]) for pod in json.loads(result.stdout).get("items", [])]
    return sorted(pods)


def _completion_index_of(pod: dict) -> int:
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    raw = labels.get(_COMPLETION_INDEX_KEY, annotations.get(_COMPLETION_INDEX_KEY))
    return int(raw) if raw is not None else 0


def _logs_of(target: str, *, namespace: str, kubectl: Callable) -> str:
    result = kubectl(
        [
            "logs",
            target,
            "--namespace",
            namespace,
            "--all-containers",
            "--tail",
            str(_TERMINAL_LOG_LINES),
        ]
    )
    return result.stdout or result.stderr


def _joined(logs: list[str]) -> str:
    if len(logs) == 1:
        return logs[0]
    return "\n".join(f"[completion index {index}]\n{log}" for index, log in enumerate(logs))


def _kubectl(arguments: list[str]) -> subprocess.CompletedProcess:
    print(f"EXEC: {shlex.join(['kubectl', *arguments])}", flush=True)
    return subprocess.run(["kubectl", *arguments], capture_output=True, text=True)
