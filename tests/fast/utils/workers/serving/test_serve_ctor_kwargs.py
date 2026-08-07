from __future__ import annotations

from typing import Any

import pytest

from miles.utils.function_registry import function_registry
from miles.utils.workers.pod_context import SUBPROCESS_INDEX_ENV_VAR, read_pod_rank
from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.serve_common import build_base_parser, split_worker_argv
from miles.utils.workers.worker_spec import WorkerCtorContext

CTOR_KWARGS_FN = "test:ctor_kwargs"
WORKER_FN = "test:worker"
CONTEXT_FN = "test:context_kwargs"


class KeywordOnlyWorker:
    def __init__(self, *, args: str, rank: int, cell_id: str) -> None:
        self.args = args
        self.rank = rank
        self.cell_id = cell_id


def worker_of(argv: list[str], *, environ: dict[str, str]) -> Any:
    own_argv, worker_argv = split_worker_argv(argv)
    args = build_base_parser("test").parse_args(own_argv)
    rank = read_pod_rank(ranks_per_pod=args.ranks_per_pod, gpu_slots_per_rank=args.gpu_slots_per_rank, environ=environ)
    return serve_inner.create_worker(args, worker_argv=worker_argv, rank=rank)


def ctor_kwargs(*, pool_id: str, worker_argv: list[str], context: WorkerCtorContext) -> dict[str, Any]:
    return dict(args=f"{pool_id}:{' '.join(worker_argv)}", rank=context.worker_in_cell_index, cell_id=context.cell_id)


class ContextWorker:
    def __init__(self, *, context: WorkerCtorContext) -> None:
        self.context = context


def context_kwargs(*, pool_id: str, worker_argv: list[str], context: WorkerCtorContext) -> dict[str, Any]:
    return dict(context=context)


@pytest.fixture
def registered_functions():
    with function_registry.temporary(CTOR_KWARGS_FN, ctor_kwargs):
        with function_registry.temporary(WORKER_FN, KeywordOnlyWorker):
            yield


@pytest.fixture
def registered_context_functions():
    with function_registry.temporary(CONTEXT_FN, context_kwargs):
        with function_registry.temporary(WORKER_FN, ContextWorker):
            yield


class TestCreateWorker:
    def test_builds_a_keyword_only_worker_from_the_computed_kwargs(self, registered_functions):
        """Every real served worker takes keyword arguments, so handing it the argv positionally is a TypeError."""
        worker = worker_of(
            [
                "--worker",
                WORKER_FN,
                "--pool-id",
                "trainer-actor",
                "--ctor-kwargs-fn",
                CTOR_KWARGS_FN,
                "--",
                "--rollout-num-gpus",
                "8",
            ],
            environ={},
        )

        assert isinstance(worker, KeywordOnlyWorker)
        assert worker.args == "trainer-actor:--rollout-num-gpus 8"

    def test_hands_the_kwargs_the_rank_this_process_runs_as(self, registered_functions):
        """Every rank of a pod runs the same command, so the rank has to come from the process, not the argv."""
        worker = worker_of(
            [
                "--worker",
                WORKER_FN,
                "--pool-id",
                "trainer-actor",
                "--ctor-kwargs-fn",
                CTOR_KWARGS_FN,
                "--ranks-per-pod",
                "8",
                "--gpu-slots-per-rank",
                "1",
                "--",
                "--x",
            ],
            environ={SUBPROCESS_INDEX_ENV_VAR: "5"},
        )

        assert worker.rank == 5

    def test_keeps_passing_the_argv_when_no_ctor_kwargs_are_computed(self, registered_functions):
        """The plain argv factory is what every rpc test worker uses, and it must keep working."""
        assert worker_of(["--worker", "builtins.list", "--", "a", "b"], environ={}) == ["a", "b"]

    def test_the_context_carries_the_providers_the_spec_may_address_workers_through(
        self, registered_context_functions
    ):
        """A spec that needs another worker's address asks the context, not a global."""
        worker = worker_of(
            [
                "--worker",
                WORKER_FN,
                "--pool-id",
                "trainer-actor",
                "--ctor-kwargs-fn",
                CONTEXT_FN,
                "--",
                "--rollout-num-gpus",
                "8",
            ],
            environ={},
        )

        assert worker.context.capability is not None

    def test_refuses_to_compute_kwargs_without_knowing_which_spec(self, registered_functions):
        """The function resolves one named spec out of the run, so an unnamed one would pick arbitrarily."""
        with pytest.raises(AssertionError, match="--pool-id"):
            worker_of(["--worker", WORKER_FN, "--ctor-kwargs-fn", CTOR_KWARGS_FN, "--", "--x"], environ={})


class TestRpcPortOfARank:
    def test_the_ranks_of_one_pod_listen_on_different_ports(self):
        """The supervisor runs them all in one network namespace, so a shared port is a bind failure."""
        ports = [
            8000
            + read_pod_rank(
                ranks_per_pod=4, gpu_slots_per_rank=1, environ={SUBPROCESS_INDEX_ENV_VAR: str(index)}
            ).rank_in_pod
            for index in range(4)
        ]

        assert ports == [8000, 8001, 8002, 8003]

    def test_the_first_rank_keeps_the_port_the_address_book_predicts(self):
        """The provider addresses a pod at the spec's static rpc port, which rank zero has to answer on."""
        rank = read_pod_rank(ranks_per_pod=4, gpu_slots_per_rank=1, environ={})

        assert rank.rank_in_pod == 0
