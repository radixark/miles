import hashlib
import json
import logging
from dataclasses import dataclass, field

import requests

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.ray.specs.train import compute_trainer_controller_pool_id
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT, RunNames
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.kubectl_reads import read_objects_of_release
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, HEALTH_PATH
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name, static_worker_host
from miles.utils.workers.worker_spec import DEFAULT_RPC_PORT


# ================================= constants ==================================


logger = logging.getLogger(__name__)

POD_KIND: str = "pods"
STATEFUL_SET_KIND: str = "statefulsets"
LEADER_WORKER_SET_KIND: str = "leaderworkersets.leaderworkerset.x-k8s.io"
WORKLOAD_KINDS: tuple[str, ...] = (STATEFUL_SET_KIND, LEADER_WORKER_SET_KIND)
BOOT_UUID_TIMEOUT_SECONDS: float = 10.0


# ============================ what a cluster holds ============================


class PodFact(FrozenStrictBaseModel):
    name: str
    uid: str
    restart_count: int


class WorkloadFact(FrozenStrictBaseModel):
    kind: str
    name: str
    generation: int
    pod_template_fingerprint: str
    restart_at: str | None


class ClusterSnapshot(FrozenStrictBaseModel):
    pods: tuple[PodFact, ...]
    workloads: tuple[WorkloadFact, ...]
    trainer_boot_uuid: str | None
    reads_missing: tuple[str, ...] = ()

    @property
    def workload_names(self) -> tuple[str, ...]:
        return tuple(one.name for one in self.workloads)

    @property
    def describes_whole_release(self) -> bool:
        return not ({POD_KIND, *WORKLOAD_KINDS} & set(self.reads_missing))

    @property
    def describes_gone_release(self) -> bool:
        return not self.pods or not self.workloads


# ============================== observing a run ===============================


@dataclass
class ClusterObserver:
    release: str
    namespace: str
    trainer_id: str
    snapshots: list[ClusterSnapshot] = field(default_factory=list)
    attempts: int = 0
    failures: int = 0

    def observe_once_or_warn(self) -> None:
        try:
            self.observe_once()
        except BaseException:
            self.failures += 1
            logger.warning("Failed to observe the cluster of a run being hot restarted", exc_info=True)

    def observe_once(self) -> None:
        self.attempts += 1
        snapshot = read_cluster_snapshot(
            release=self.release,
            namespace=self.namespace,
            trainer_rpc_url=compute_trainer_rpc_url(
                release=self.release, namespace=self.namespace, trainer_id=self.trainer_id
            ),
        )
        if not snapshot.describes_whole_release:
            self.failures += 1
            logger.warning(
                f"Observing {self.release} could not read {list(snapshot.reads_missing)}, so this is a read that "
                f"failed rather than a run whose pods changed"
            )
            return
        if snapshot.describes_gone_release:
            logger.warning(
                f"Observed {self.release} with {len(snapshot.pods)} pod(s) and {len(snapshot.workloads)} "
                f"workload(s), which is a release being uninstalled rather than a run whose pods were replaced"
            )
            return
        self.snapshots.append(snapshot)


# ============================== reading a cluster =============================


def compute_trainer_rpc_url(*, release: str, namespace: str, trainer_id: str) -> str:
    host = RunNames.service_fqdn(
        name=static_worker_host(release, compute_trainer_controller_pool_id(trainer_id), 0), namespace=namespace
    )
    return f"http://{host}:{DEFAULT_RPC_PORT}{HEALTH_PATH}"


def read_cluster_snapshot(*, release: str, namespace: str, trainer_rpc_url: str) -> ClusterSnapshot:
    boot_uuid = read_boot_uuid(trainer_rpc_url)
    payload_of_kind = {
        kind: _read_objects(kind=kind, release=release, namespace=namespace) for kind in (POD_KIND, *WORKLOAD_KINDS)
    }

    pods = payload_of_kind[POD_KIND]
    workloads = tuple(
        fact
        for kind in WORKLOAD_KINDS
        if (payload := payload_of_kind[kind]) is not None
        for fact in parse_workload_facts(payload, kind=kind)
    )
    return ClusterSnapshot(
        pods=parse_pod_facts(pods) if pods is not None else (),
        workloads=tuple(sorted(workloads, key=lambda one: (one.kind, one.name))),
        trainer_boot_uuid=boot_uuid,
        reads_missing=tuple(kind for kind, payload in payload_of_kind.items() if payload is None),
    )


def compute_hot_restart_workloads(release: str) -> frozenset[str]:
    return frozenset(
        component_name(release, component) for component in (ORCHESTRATOR_COMPONENT, ROLLOUT_EXECUTOR_POOL_ID)
    )


def read_restart_stamp_of_workload(*, release: str, namespace: str) -> dict[str, str | None] | None:
    stamps: dict[str, str | None] = {}
    for kind in WORKLOAD_KINDS:
        payload = _read_objects(kind=kind, release=release, namespace=namespace)
        if payload is None:
            return None
        stamps.update({fact.name: fact.restart_at for fact in parse_workload_facts(payload, kind=kind)})
    return stamps


def read_boot_uuid(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=BOOT_UUID_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.headers.get(BOOT_UUID_HEADER)
    except Exception:
        logger.warning(f"Failed to read the rpc server boot uuid from {url}", exc_info=True)
        return None


def parse_pod_facts(payload: dict) -> tuple[PodFact, ...]:
    facts = [
        PodFact(
            name=item["metadata"]["name"],
            uid=item["metadata"]["uid"],
            restart_count=sum(
                int(one.get("restartCount", 0)) for one in item.get("status", {}).get("containerStatuses", [])
            ),
        )
        for item in payload["items"]
    ]
    return tuple(sorted(facts, key=lambda one: one.name))


def parse_workload_facts(payload: dict, *, kind: str) -> tuple[WorkloadFact, ...]:
    facts = [
        WorkloadFact(
            kind=kind,
            name=item["metadata"]["name"],
            generation=int(item["metadata"]["generation"]),
            pod_template_fingerprint=_compute_pod_template_fingerprint(item),
            restart_at=_read_restart_at(item),
        )
        for item in payload["items"]
    ]
    return tuple(sorted(facts, key=lambda one: one.name))


def _read_objects(*, kind: str, release: str, namespace: str) -> dict | None:
    try:
        return json.loads(read_objects_of_release(kind=kind, release=release, namespace=namespace, output="json"))
    except Exception:
        logger.warning(f"Failed to list the {kind} of {release} in {namespace}", exc_info=True)
        return None


def _read_restart_at(item: dict) -> str | None:
    templates = _read_pod_templates(item)
    stamps = {
        stamp
        for template in templates
        if (stamp := template.get("metadata", {}).get("annotations", {}).get(RESTART_AT_ANNOTATION)) is not None
    }
    assert len(stamps) <= 1, (
        f"{item['metadata']['name']} carries the restart stamps {sorted(stamps)} on the templates of one object, "
        f"and a hot restart writes one stamp per object"
    )
    return next(iter(stamps), None)


def _compute_pod_template_fingerprint(item: dict) -> str:
    text = json.dumps(_read_pod_templates(item), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _read_pod_templates(item: dict) -> tuple[dict, ...]:
    spec = item.get("spec", {})
    leader_worker_template = spec.get("leaderWorkerTemplate", {})
    templates = [
        spec.get("template"),
        leader_worker_template.get("leaderTemplate"),
        leader_worker_template.get("workerTemplate"),
    ]
    present = tuple(template for template in templates if template is not None)
    assert present, (
        f"{item['metadata']['name']} has no StatefulSet or LeaderWorkerSet pod template, so a hot restart cannot "
        f"tell whether that workload was rewritten"
    )
    return present
