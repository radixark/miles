from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from miles.utils.external_utils.command_utils.common import substitute_placeholders
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import with_observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import CommandJobValues
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Job, Pod, PodList

logger = logging.getLogger(__name__)

_JOB_TEMPLATE = "templates/command-job.yaml"
_COMPLETION_INDEX_KEY = "batch.kubernetes.io/job-completion-index"
_TERMINAL_LOG_LINES = 200
_POLL_INTERVAL_SECONDS = 5.0
_TIMEOUT_SECONDS = 3 * 60 * 60
_RELEASE = "miles-run-command"


class CommandJobContext(FrozenStrictBaseModel):
    namespace: str
    chart_dir: Path
    helm_values_files: tuple[str, ...] = ()
    gpus_per_node: int
    timeout_seconds: float = _TIMEOUT_SECONDS


class _CommandJob(FrozenStrictBaseModel):
    context: CommandJobContext
    step: str
    completions: int
    gpus_per_pod: int

    @property
    def object_name(self) -> str:
        return naming.component_name(_RELEASE, self.step)

    @property
    def master_address(self) -> str:
        return RunNames.service_fqdn(name=f"{self.object_name}-0.{self.object_name}", namespace=self.context.namespace)

    @property
    def pod_selector(self) -> str:
        return Kubectl.job_selector(self.object_name)


def run_on_nodes(
    context: CommandJobContext,
    cmd: str,
    *,
    capture_output: bool,
    completions: int,
    step: str,
) -> list[str | None]:
    job = _CommandJob(context=context, step=step, completions=completions, gpus_per_pod=context.gpus_per_node)
    prepared = substitute_placeholders(
        cmd,
        node_rank="${JOB_COMPLETION_INDEX}",
        nnodes=str(completions),
        master_addr=job.master_address,
        node_ip="$(hostname -i)",
    )
    return _run_job(job, command=["bash", "-c", prepared], capture_output=capture_output)


def _run_job(job: _CommandJob, *, command: list[str], capture_output: bool) -> list[str | None]:
    manifest = _render_job(job, command=command)

    Kubectl.delete_job(job.object_name, namespace=job.context.namespace)
    Kubectl.apply(manifest, namespace=job.context.namespace)

    with with_observability(namespace=job.context.namespace, selector=job.pod_selector):
        outcome = _wait(job)

    if outcome != "complete":
        raise RuntimeError(f"Job {job.object_name} {outcome}; last log lines:\n{_joined(_logs_per_completion(job))}")

    logs = _logs_per_completion(job) if capture_output else [None] * job.completions
    Kubectl.delete_job(job.object_name, namespace=job.context.namespace)
    return logs


def _render_job(job: _CommandJob, *, command: list[str]) -> str:
    Helm.build_dependencies(job.context.chart_dir)

    command_job = CommandJobValues(
        enabled=True,
        name=job.step,
        object_name=job.object_name,
        command=command,
        completions=job.completions,
        gpus_per_pod=job.gpus_per_pod,
        active_deadline_seconds=int(job.context.timeout_seconds),
    )
    values: dict[str, Any] = {
        f"commandJob.{key}": value for key, value in command_job.model_dump(by_alias=True, exclude_none=True).items()
    }
    values["run.id"] = "command-job"

    return Helm.template(
        release=_RELEASE,
        chart=job.context.chart_dir,
        namespace=job.context.namespace,
        show_only=_JOB_TEMPLATE,
        values=values,
        values_files=list(job.context.helm_values_files),
    )


def _wait(job: _CommandJob) -> str:
    waited = 0.0
    while waited < job.context.timeout_seconds:
        status = _job_status(job)
        if status in ("complete", "failed"):
            return status
        time.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
    return f"did not finish within {job.context.timeout_seconds:.0f}s"


def _job_status(job: _CommandJob) -> str:
    described = Kubectl.get_json("job", return_type=Job, name=job.object_name, namespace=job.context.namespace)
    if described is None:
        return "pending"

    for condition in described.status.conditions:
        if condition.status != "True":
            continue
        if condition.type in ("Complete", "SuccessCriteriaMet"):
            return "complete"
        if condition.type in ("Failed", "FailureTarget"):
            return "failed"
    return "running"


def _logs_per_completion(job: _CommandJob) -> list[str]:
    pods = _pods_by_completion_index(job)
    if not pods:
        return [_logs_of_target(f"job/{job.object_name}", namespace=job.context.namespace)] * job.completions

    logs = {index: _logs_of_target(name, namespace=job.context.namespace) for index, name in pods}
    return [
        logs.get(index, f"no pod of this job reported completion index {index}") for index in range(job.completions)
    ]


def _pods_by_completion_index(job: _CommandJob) -> list[tuple[int, str]]:
    listed = Kubectl.get_json("pods", return_type=PodList, namespace=job.context.namespace, selector=job.pod_selector)
    if listed is None:
        return []

    return sorted((_completion_index_of_pod(pod), pod.metadata.name) for pod in listed.items)


def _completion_index_of_pod(pod: Pod) -> int:
    raw = pod.metadata.labels.get(_COMPLETION_INDEX_KEY, pod.metadata.annotations.get(_COMPLETION_INDEX_KEY))
    return int(raw) if raw is not None else 0


def _logs_of_target(target: str, *, namespace: str) -> str:
    return Kubectl.logs(target, namespace=namespace, tail=_TERMINAL_LOG_LINES)


def _joined(logs: list[str]) -> str:
    if len(logs) == 1:
        return logs[0]
    return "\n".join(f"[completion index {index}]\n{log}" for index, log in enumerate(logs))
