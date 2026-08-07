from __future__ import annotations

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import RELEASE, make_engine_spec, make_trainer_spec

from miles.utils.external_utils.command_utils.helm_backend.values import RunLayout, build_values
from miles.utils.workers.worker_provider.kubernetes.run import kubernetes_run
from miles.utils.workers.worker_spec import BaseWorkerSpec

NAMESPACE = "rl"


class TestRanksPerPod:
    @pytest.mark.parametrize(("num_workers_per_cell", "num_gpus_per_node"), [(1, 8), (4, 8), (8, 8), (16, 8), (2, 1)])
    def test_matches_the_number_of_ranks_the_launcher_supervises_in_one_pod(
        self, num_workers_per_cell: int, num_gpus_per_node: int
    ) -> None:
        """The provider fans a pod out into the ranks its own command started, so the two rules cannot drift."""
        spec = make_trainer_spec(num_workers_per_cell=num_workers_per_cell)

        assert _ranks_per_pod(spec, num_gpus_per_node=num_gpus_per_node) == _launched_ranks_per_pod(
            spec, num_gpus_per_node=num_gpus_per_node
        )

    def test_an_engine_pod_runs_one_command_and_therefore_holds_one_rank(self) -> None:
        """An engine pod is a single server process spanning its gpus, whatever its spec counts as a worker."""
        spec = make_engine_spec()

        assert _launched_ranks_per_pod(spec, num_gpus_per_node=8) == 1
        assert _ranks_per_pod(spec, num_gpus_per_node=8) == 1


def _ranks_per_pod(spec: BaseWorkerSpec, *, num_gpus_per_node: int) -> int:
    run = kubernetes_run(
        specs=[spec],
        namespace=NAMESPACE,
        release=RELEASE,
        kubernetes_client_factory=lambda: object(),
        num_gpus_per_node=num_gpus_per_node,
    )
    return run.pools[spec.name].ranks_per_pod


def _launched_ranks_per_pod(spec: BaseWorkerSpec, *, num_gpus_per_node: int) -> int:
    values = build_values(
        [spec],
        RunLayout(
            run_id="260101-000000-000",
            release=RELEASE,
            orchestrator_command=["python", "train.py"],
            worker_argv=[],
            num_gpus_per_node=num_gpus_per_node,
        ),
    )
    command = next(
        entry["command"]
        for section in ("trainers", "inferenceEngines", "staticWorkers")
        for entry in values["run"][section]
        if entry["pool_id"] == spec.name
    )
    if "--ranks-per-pod" not in command:
        return 1
    return int(command[command.index("--ranks-per-pod") + 1])
