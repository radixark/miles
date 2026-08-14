from __future__ import annotations

import logging
import time
from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import PodList

logger = logging.getLogger(__name__)


class Diagnosis(FrozenStrictBaseModel):
    directory: Path
    missing: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing


def collect_diagnosis(
    *, namespace: str, output_dir: Path, selector: str | None = None, state_file: Path | None = None
) -> Diagnosis:
    directory = output_dir / f"miles-diagnosis-{namespace}-{time.strftime('%Y%m%d-%H%M%S')}"
    directory.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    if not _capture(
        path=directory / "events.txt",
        command=["kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ):
        missing.append("events")

    pods = _pod_names(namespace=namespace, selector=selector)
    if pods is None:
        missing.append(f"pod listing in namespace {namespace}")
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
    _capture(
        path=directory / f"{pod}.previous.log",
        command=["kubectl", "logs", pod, "-n", namespace, "--all-containers", "--previous"],
        skip_when_it_fails=True,
    )
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


def _capture(path: Path, command: list[str], skip_when_it_fails: bool = False) -> bool:
    result = run_process(command, capture_output=True, check=False)
    if result.returncode != 0 and skip_when_it_fails:
        return True
    path.write_text(result.stdout + result.stderr)
    return result.returncode == 0
