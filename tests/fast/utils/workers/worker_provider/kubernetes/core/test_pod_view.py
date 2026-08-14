from miles.utils.workers.k8s_types import ContainerStatus, Pod, PodCondition, PodMetadata, PodSpec, PodStatus
from miles.utils.workers.worker_provider.kubernetes.core import pod_view
from miles.utils.workers.worker_provider.kubernetes.helm import env as helm_env
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS


def make_pod(
    name: str = "engine-0-0",
    cell_id_suffix: str = "0",
    pod_in_cell_index: str = "0",
    pool_id: str = "engine",
    ready: bool = True,
    **kwargs,
) -> Pod:
    pod_labels = {
        helm_env.DEFAULT_LABEL_KEYS.pool_id: pool_id,
        helm_env.DEFAULT_LABEL_KEYS.cell_index: cell_id_suffix,
        helm_env.DEFAULT_LABEL_KEYS.pod_in_cell_index: pod_in_cell_index,
    }
    pod_labels.update(kwargs.pop("labels", {}))

    pod = Pod(
        metadata=PodMetadata(
            name=name,
            uid=kwargs.pop("uid", f"uid-{name}"),
            labels=pod_labels,
            annotations=kwargs.pop("annotations", {}),
            deletion_timestamp=kwargs.pop("deletion_timestamp", None),
        ),
        spec=PodSpec(node_name=kwargs.pop("node_name", "gpu-1"), subdomain=kwargs.pop("subdomain", None)),
        status=PodStatus(
            pod_ip=kwargs.pop("pod_ip", "10.0.0.1"),
            conditions=[PodCondition(type="Ready", status="True" if ready else "False")],
            container_statuses=[ContainerStatus(restart_count=kwargs.pop("restarts", 0))],
        ),
    )
    assert not kwargs, f"make_pod does not know {sorted(kwargs)}, so the fixture would not be what the test meant"
    return pod


def make_unlabelled_pod(name: str, labels: dict[str, str] | None = None) -> Pod:
    return Pod(
        metadata=PodMetadata(name=name, uid=f"uid-{name}", labels=labels or {}),
        spec=PodSpec(),
        status=PodStatus(),
    )


def parse(pod, keys=None):
    return pod_view.parse_pod(pod, keys or DEFAULT_LABEL_KEYS)


class TestParsePod:
    def test_reads_the_cell_a_pod_belongs_to(self):
        """A cell is a pool_id and a group index, which is what its consumers address."""
        parsed = parse(make_pod(pool_id="inference-engine-0-0", cell_id_suffix="2"))

        assert parsed.cell_id == "inference-engine-0-0-2"

    def test_ignores_a_pod_that_carries_no_cell_labels(self):
        """A namespace holds other pods, and treating one as a worker would invent a cell."""
        assert parse(make_unlabelled_pod("prometheus-0")) is None

    def test_reads_the_pool_a_pod_was_deployed_for(self):
        """Every consumer addresses cells by pool_id, and only this label says which pool_id a pod serves."""
        assert parse(make_pod(pool_id="trainer-engine-actor")).pool_id == "trainer-engine-actor"

    def test_ignores_a_pod_that_names_a_pool_but_no_cell(self):
        """A static worker of the chart carries the pool label with no group index, and belongs to no cell."""
        pod = make_unlabelled_pod("router-0", labels={helm_env.DEFAULT_LABEL_KEYS.pool_id: "inference-router-0"})

        assert parse(pod) is None

    def test_ignores_a_pod_that_names_no_pool(self):
        """A pod without the pool_id label cannot be placed, and guessing one would invent a cell."""
        pod = make_unlabelled_pod("engine-0-0", labels={helm_env.DEFAULT_LABEL_KEYS.cell_index: "0"})

        assert parse(pod) is None

    def test_reports_a_pod_the_apiserver_is_deleting(self):
        """Deletion is graceful: the pod keeps answering and stays Ready while it is on its way out."""
        pod = make_pod(deletion_timestamp="2026-08-10T12:00:00Z")

        assert parse(pod).deleting is True
        assert parse(pod).ready is True

    def test_reports_a_pod_nobody_asked_to_delete(self):
        """Every pod of a healthy cell must read as staying, or no cell is ever alive."""
        assert parse(make_pod()).deleting is False

    def test_reports_a_pod_that_is_not_ready_yet(self):
        """A cell whose workers are still loading must not be given work."""
        assert parse(make_pod(ready=False)).ready is False

    def test_reads_the_keys_a_platform_configured(self):
        """A platform that already labels its pods should not have to relabel them for miles."""
        keys = DEFAULT_LABEL_KEYS.model_copy(update={"pool_id": "acme.io/group", "cell_index": "acme.io/index"})
        pod = make_unlabelled_pod("p", labels={"acme.io/group": "engine", "acme.io/index": "3"})

        assert parse(pod, keys).cell_id == "engine-3"

    def test_reads_how_many_pods_the_cell_should_have(self):
        """A group still being created has ready pods but not all of them, and must not be given work."""
        pod = make_pod(annotations={helm_env.DEFAULT_LABEL_KEYS.cell_size_annotation: "4"})

        assert parse(pod).cell_size == 4

    def test_the_size_is_not_looked_for_among_the_labels(self):
        """LeaderWorkerSet publishes group-index as a label but size as an annotation, and only one of them is read."""
        pod = make_pod(labels={helm_env.DEFAULT_LABEL_KEYS.cell_size_annotation: "4"})

        assert parse(pod).cell_size == 0

    def test_reports_no_size_when_the_platform_publishes_none(self):
        """A platform that does not say cannot be second-guessed, so the cell is judged on readiness alone."""
        assert parse(make_pod()).cell_size == 0

    def test_reads_the_domain_facts_a_platform_attached(self):
        """An engine's model id reaches miles through the pod, not through the launcher's memory."""
        pod = make_pod(
            annotations={f"{helm_env.DEFAULT_LABEL_KEYS.meta_annotation_prefix}model_id": "glm", "other": "ignored"}
        )

        assert parse(pod).meta == {"model_id": "glm"}

    def test_reads_nothing_from_a_pod_without_annotations(self):
        """Most pods carry none, and a missing annotation block is not an error."""
        assert parse(make_pod()).meta == {}

    def test_reads_the_gpus_the_platform_assigned(self):
        """A worker owns a share of its pod's gpus, which only the platform's annotation can say."""
        pod = make_pod(
            annotations={
                f"{helm_env.DEFAULT_LABEL_KEYS.meta_annotation_prefix}{helm_env.DEFAULT_LABEL_KEYS.gpu_ids_meta}": "0,1,2,3"
            }
        )

        assert parse(pod).gpu_ids == (0, 1, 2, 3)


class TestParsePods:
    def test_returns_the_pods_of_a_cell_in_index_order(self):
        """Everything downstream numbers workers off this order, so the store's order must not leak through."""
        raw = [make_pod(name=f"engine-0-{index}", pod_in_cell_index=str(index)) for index in (2, 0, 1)]

        assert [pod.pod_in_cell_index for pod in pod_view.parse_pods(raw, keys=DEFAULT_LABEL_KEYS)] == [0, 1, 2]

    def test_drops_the_pods_that_are_not_workers(self):
        """The label selector is coarser than the cell labels, so unrelated pods reach this function."""
        raw = [make_pod(), make_unlabelled_pod("prometheus-0")]

        assert len(pod_view.parse_pods(raw, keys=DEFAULT_LABEL_KEYS)) == 1


class TestCellMembersHash:
    def test_is_stable_for_the_same_membership(self):
        """A consumer compares this across polls, so noise would look like a healed cell every time."""
        pods = [parse(make_pod(name=f"engine-0-{index}", pod_in_cell_index=str(index))) for index in range(2)]

        assert pod_view.cell_members_hash(pods) == pod_view.cell_members_hash(list(reversed(pods)))

    def test_changes_when_a_pod_is_replaced(self):
        """A new pod lost whatever the old one held in memory, and its consumers must resynchronise."""
        before = [parse(make_pod(uid="uid-a"))]
        after = [parse(make_pod(uid="uid-b"))]

        assert pod_view.cell_members_hash(before) != pod_view.cell_members_hash(after)

    def test_changes_when_a_pod_restarts_in_place(self):
        """The uid survives a restart but the process does not, so the hash has to notice."""
        before = [parse(make_pod(restarts=0))]
        after = [parse(make_pod(restarts=1))]

        assert pod_view.cell_members_hash(before) != pod_view.cell_members_hash(after)
