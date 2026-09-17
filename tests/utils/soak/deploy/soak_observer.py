import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    BOOT_UUID_TIMEOUT_SECONDS,
    POD_KIND,
    WORKLOAD_KINDS,
    ClusterSnapshot,
    compute_hot_restart_workloads,
    compute_trainer_rpc_url,
    parse_pod_facts,
    parse_workload_facts,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import observe_run_progress
from tests.utils.soak.action import run_command
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.state import SoakDeploymentTarget, SoakObservation

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT, RunNames
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class HotRestartSoakObserver(SoakObserver):
    trainer_id: str
    checkpoint_dir: Path
    events_dir: Path

    async def observe(self) -> SoakObservation:
        assert self.release is not None and self.namespace is not None
        observed_at = datetime.now(timezone.utc)
        kinds = (POD_KIND, *WORKLOAD_KINDS)
        boot_uuid, job, *payloads = await asyncio.gather(
            self._read_boot_uuid(), self._read_uninstall_job(), *(self._read_objects(kind) for kind in kinds)
        )
        payload_of_kind = dict(zip(kinds, payloads, strict=True))
        pods = payload_of_kind[POD_KIND]
        workloads = tuple(
            fact
            for kind in WORKLOAD_KINDS
            if (payload := payload_of_kind[kind]) is not None
            for fact in parse_workload_facts(payload, kind=kind)
        )
        snapshot = ClusterSnapshot(
            pods=parse_pod_facts(pods) if pods is not None else (),
            workloads=tuple(sorted(workloads, key=lambda one: (one.kind, one.name))),
            trainer_boot_uuid=boot_uuid,
            reads_missing=tuple(kind for kind, payload in payload_of_kind.items() if payload is None),
        )
        errors = {
            kind: "Read failed" for kind in (*snapshot.reads_missing, *(("uninstall_job",) if job is None else ()))
        }
        try:
            progress = await observe_run_progress(
                checkpoint_dir=self.checkpoint_dir,
                events_dir=self.events_dir,
                timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
            )
        except Exception as error:
            logger.warning("Failed to observe hot restart progress", exc_info=True)
            errors["progress"] = repr(error)
            progress = None
        deployments = []
        expected = compute_hot_restart_workloads(self.release)
        if (
            snapshot.describes_whole_release
            and not snapshot.describes_gone_release
            and expected <= set(snapshot.workload_names)
            and job is not None
            and progress is not None
        ):
            deployments.append(
                SoakDeploymentTarget(
                    namespace=self.namespace,
                    release=self.release,
                    workload_stamps={one.name: one.restart_at for one in snapshot.workloads},
                    workload_uids={
                        item["metadata"]["name"]: item["metadata"]["uid"]
                        for kind in WORKLOAD_KINDS
                        for item in payload_of_kind[kind]["items"]
                    },
                    saved_iteration=progress.last_saved_iteration,
                    finished_rollout_id=progress.last_finished_rollout_id,
                    state_file=Manifest(
                        objects=[item for kind in WORKLOAD_KINDS for item in payload_of_kind[kind]["items"]],
                        namespace=self.namespace,
                    ).state_file(
                        stateful_set=RunNames.orchestrator_object(release=self.release),
                        container=ORCHESTRATOR_COMPONENT,
                    ),
                    uninstall_job_uid=job.get("uid"),
                )
            )
        return SoakObservation(
            timestamp=observed_at,
            cells=[],
            deployments=deployments,
            details={"hot_restart_cluster": snapshot.model_dump(mode="json")},
            errors=errors,
        )

    async def _read_objects(self, kind: str) -> dict | None:
        assert self.namespace is not None and self.release is not None
        try:
            result = await run_command(
                [
                    "kubectl",
                    "get",
                    kind,
                    "--namespace",
                    self.namespace,
                    "--selector",
                    compute_release_selector(release=self.release),
                    "--output",
                    "json",
                ],
                timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
            )
            return json.loads(result.stdout)
        except Exception:
            logger.warning("Failed to read %s of %s", kind, self.release, exc_info=True)
            return None

    async def _read_uninstall_job(self) -> dict[str, str] | None:
        assert self.namespace is not None and self.release is not None
        try:
            result = await run_command(
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
                timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
            )
            return {"uid": json.loads(result.stdout)["metadata"]["uid"]} if result.stdout.strip() else {}
        except Exception:
            logger.warning("Failed to observe the uninstall job", exc_info=True)
            return None

    async def _read_boot_uuid(self) -> str | None:
        assert self.namespace is not None and self.release is not None
        url = compute_trainer_rpc_url(release=self.release, namespace=self.namespace, trainer_id=self.trainer_id)
        try:
            async with httpx.AsyncClient(timeout=BOOT_UUID_TIMEOUT_SECONDS) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.headers.get(BOOT_UUID_HEADER)
        except Exception:
            logger.warning("Failed to read trainer boot UUID from %s", url, exc_info=True)
            return None
