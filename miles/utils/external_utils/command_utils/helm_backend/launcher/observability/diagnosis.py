from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import PodList

logger = logging.getLogger(__name__)

_NO_PREVIOUS_CONTAINER = "previous terminated container"


class Diagnosis(FrozenStrictBaseModel):
    directory: Path
    missing: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing


def collect_diagnosis(
    *, namespace: str, output_dir: Path, selector: str | None = None, state_file: Path | None = None
) -> Diagnosis:
    directory = output_dir / f"miles-diagnosis-{namespace}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    directory.mkdir(parents=True)

    missing: list[str] = []
    if not _capture(
        path=directory / "events.txt",
        command=["kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ):
        missing.append("events")

    pods = _pod_names(namespace=namespace, selector=selector)
    if pods is None:
        missing.append(f"pod listing in namespace {namespace}")
    elif not pods:
        missing.append(f"pods of the run in namespace {namespace}")
    for pod in pods or []:
        missing += _capture_pod(pod, namespace=namespace, directory=directory)

    if state_file is not None:
        text = state_file.read_text() if state_file.is_file() else f"{state_file} does not exist\n"
        (directory / state_file.name).write_text(text)

    return Diagnosis(directory=directory, missing=tuple(missing))


def _capture_pod(pod: str, *, namespace: str, directory: Path) -> list[str]:
    missing = []
    if not _capture(
        path=directory / f"{pod}.log", command=["kubectl", "logs", pod, "-n", namespace, "--all-containers"]
    ):
        missing.append(f"logs of {pod}")
    if not _capture(
        path=directory / f"{pod}.previous.log",
        command=["kubectl", "logs", pod, "-n", namespace, "--all-containers", "--previous"],
        ignored_failure=_NO_PREVIOUS_CONTAINER,
    ):
        missing.append(f"previous logs of {pod}")
    if not _capture(
        path=directory / f"{pod}.describe.txt", command=["kubectl", "describe", "pod", pod, "-n", namespace]
    ):
        missing.append(f"describe of {pod}")
    return missing


def _pod_names(*, namespace: str, selector: str | None) -> list[str] | None:
    try:
        listed = Kubectl.get_json("pods", return_type=PodList, namespace=namespace, selector=selector)
        return [pod.metadata.name for pod in listed.items] if listed is not None else []
    except Exception:
        logger.warning(f"Could not list the pods of namespace {namespace}", exc_info=True)
        return None


def _capture(*, path: Path, command: list[str], ignored_failure: str | None = None) -> bool:
    result = run_process(command, capture_output=True, check=False)
    if result.returncode != 0 and ignored_failure is not None and ignored_failure in result.stderr:
        return True
    path.write_text(result.stdout + result.stderr)
    return result.returncode == 0
