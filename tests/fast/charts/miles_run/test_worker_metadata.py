import json

import pytest
from tests.fast.charts.utils import RUN_RELEASE_NAME, named_object, render_run, requires_helm, with_object_names


@requires_helm
class TestWorkerMetadataProjection:
    @pytest.mark.parametrize(
        "section,kind",
        [
            ("staticWorkers", "StatefulSet"),
            ("inferenceEngines", "LeaderWorkerSet"),
            ("trainerEngines", "LeaderWorkerSet"),
        ],
    )
    def test_each_pool_projects_its_exact_worker_metadata_through_the_downward_api(
        self, section: str, kind: str
    ) -> None:
        """Every worker receives the JSON annotation from its own pod template through the exact metadata environment key."""
        annotation = "miles.radixark.io/worker-metadata"
        metadata = json.dumps(
            {
                "workers_per_pod": 2,
                "pods_per_cell": 1,
                "gpu_slots_per_worker": 0 if section == "staticWorkers" else 1,
                "dynamic_pool": section != "staticWorkers",
                "worker_class": None,
                "port_infos": [],
                "static_meta": {"values": {"note": 'quoted "value"'}},
            }
        )
        pools = with_object_names(
            [{"name": "metadata-worker", "command": ["worker"], "annotations": {annotation: metadata}}]
        )
        objects = render_run("--set-json", f"run.{section}={json.dumps(pools)}")
        workload = named_object(objects, kind, f"{RUN_RELEASE_NAME}-miles-run-metadata-worker")
        template = (
            workload["spec"]["template"]
            if kind == "StatefulSet"
            else workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]
        )
        environment = template["spec"]["containers"][0]["env"]

        assert template["metadata"]["annotations"][annotation] == metadata
        assert [item for item in environment if item["name"] == "MILES_WORKER_METADATA"] == [
            {
                "name": "MILES_WORKER_METADATA",
                "valueFrom": {"fieldRef": {"fieldPath": f"metadata.annotations['{annotation}']"}},
            }
        ]
