import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tests.utils.deploy.hot_restart.cluster_observer import (
    WORKLOAD_KINDS,
    ClusterRead,
    compute_hot_restart_workloads,
    compute_trainer_rpc_url,
    read_cluster_snapshot,
)
from tests.utils.deploy.hot_restart.evidence import RunProgress, read_run_progress
from tests.utils.soak.core.events import SoakObservationEvent
from tests.utils.soak.core.types import SoakObserver
from tests.utils.soak.core.utils import recording_error
from tests.utils.soak.deploy.types import DeploymentObservationDetails, DeploymentTarget

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT, RunNames
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS


@dataclass(frozen=True, kw_only=True)
class DeploymentObserver(SoakObserver):
    namespace: str
    release: str
    trainer_id: str
    checkpoint_dir: Path
    events_dir: Path

    async def observe(self) -> SoakObservationEvent:
        observed_at = datetime.now(timezone.utc)
        errors: dict[str, str] = {}

        read, uninstall_job_uid, progress = await asyncio.gather(
            self._read_cluster(),
            self._observe_uninstall_job(errors=errors),
            self._observe_progress(errors=errors),
        )
        snapshot = read.snapshot
        errors.update({kind: "Read failed" for kind in snapshot.reads_missing})

        targets = self._create_targets(
            read=read, progress=progress, uninstall_job_uid=uninstall_job_uid, errors=errors
        )

        return SoakObservationEvent(
            timestamp=observed_at,
            targets=targets,
            details=DeploymentObservationDetails(cluster=snapshot),
            errors=errors,
        )

    async def _read_cluster(self) -> ClusterRead:
        return await asyncio.to_thread(
            read_cluster_snapshot,
            release=self.release,
            namespace=self.namespace,
            trainer_rpc_url=compute_trainer_rpc_url(
                release=self.release, namespace=self.namespace, trainer_id=self.trainer_id
            ),
        )

    async def _observe_uninstall_job(self, *, errors: dict[str, str]) -> str | None:
        uninstall_job_uid: str | None = None
        with recording_error(errors, "uninstall_job"):
            uninstall_job_uid = await asyncio.to_thread(self._read_uninstall_job)
        return uninstall_job_uid

    async def _observe_progress(self, *, errors: dict[str, str]) -> RunProgress | None:
        progress: RunProgress | None = None
        with recording_error(errors, "progress"):
            progress = await asyncio.to_thread(
                read_run_progress, checkpoint_dir=self.checkpoint_dir, events_dir=self.events_dir
            )
        return progress

    def _create_targets(
        self,
        *,
        read: ClusterRead,
        progress: RunProgress | None,
        uninstall_job_uid: str | None,
        errors: dict[str, str],
    ) -> list[DeploymentTarget]:
        snapshot = read.snapshot
        if not snapshot.describes_whole_release or snapshot.describes_gone_release:
            return []

        items = [item for kind in WORKLOAD_KINDS for item in read.payload_of_kind[kind]["items"]]
        workload_stamps = {one.name: one.restart_at for one in snapshot.workloads}
        return [
            DeploymentTarget(
                identity=self.release,
                incarnation=_compute_incarnation(workload_stamps),
                alive=True,
                ready=(
                    progress is not None
                    and "uninstall_job" not in errors
                    and compute_hot_restart_workloads(self.release) <= set(snapshot.workload_names)
                ),
                namespace=self.namespace,
                release=self.release,
                workload_stamps=workload_stamps,
                workload_uids={item["metadata"]["name"]: item["metadata"]["uid"] for item in items},
                saved_iteration=None if progress is None else progress.last_saved_iteration,
                finished_rollout_id=None if progress is None else progress.last_finished_rollout_id,
                state_file=Manifest(objects=items, namespace=self.namespace).state_file(
                    stateful_set=RunNames.orchestrator_object(release=self.release),
                    container=ORCHESTRATOR_COMPONENT,
                ),
                uninstall_job_uid=uninstall_job_uid,
            )
        ]

    def _read_uninstall_job(self) -> str | None:
        result = run_process(
            [
                "kubectl",
                "get",
                "job",
                RunNames.uninstall_job(release=self.release),
                "--namespace",
                self.namespace,
                "--output",
                "json",
                "--ignore-not-found",
            ],
            capture_output=True,
            check=True,
            timeout=KUBECTL_TIMEOUT_SECONDS,
        )
        return json.loads(result.stdout)["metadata"]["uid"] if result.stdout.strip() else None


def _compute_incarnation(workload_stamps: dict[str, str | None]) -> str:
    return hashlib.sha256(json.dumps(workload_stamps, sort_keys=True).encode()).hexdigest()
