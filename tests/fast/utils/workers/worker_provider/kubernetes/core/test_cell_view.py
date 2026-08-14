from __future__ import annotations

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import make_pool_spec

from miles.utils.workers.worker_provider.kubernetes.core.cell_view import compute_cell_info, compute_worker_infos
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys, ParsedPod
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo

CELL_ID = "engine-0"
ENGINE_CLASS = f"{__name__}.FakeEngine"


class FakeEngine:
    def generate(self, prompt: str) -> str:
        return prompt


def make_parsed_pod(*, pod_in_cell_index: int, cell_size: int = 2, ready: bool = True, **kwargs) -> ParsedPod:
    pod_ip = kwargs.pop("pod_ip", f"10.0.0.{pod_in_cell_index + 1}")
    return ParsedPod(
        name=f"engine-0-{pod_in_cell_index}",
        cell_id=CELL_ID,
        cell_index=0,
        pool_id="engine",
        pod_in_cell_index=pod_in_cell_index,
        ready=ready,
        pod_ip=pod_ip,
        uid=f"uid-{pod_in_cell_index}",
        restart_count=0,
        cell_size=cell_size,
        subdomain=kwargs.pop("subdomain", None),
        gpu_ids=kwargs.pop("gpu_ids", ()),
        meta=kwargs.pop("meta", {}),
    )


def make_run(*, workers_per_pod: int = 1, worker_class: str | None = ENGINE_CLASS) -> KubernetesRunInfo:
    return KubernetesRunInfo(
        namespace="rl",
        label_selector="app.kubernetes.io/instance=r",
        specs={
            "engine": make_pool_spec(
                "engine", ports={"rpc": 8000}, worker_class=worker_class, workers_per_pod=workers_per_pod
            )
        },
        label_keys=CellLabelKeys(
            pool_id="pool-id",
            cell_index="cell-index",
            pod_in_cell_index="pod-index",
            cell_size_annotation="cell-size",
            meta_annotation_prefix="meta-",
            gpu_ids_meta="gpu_ids",
        ),
    )


def build_cell_info(pods: list[ParsedPod]):
    return compute_cell_info(CELL_ID, pods=pods, run=make_run())


def build_worker_infos(pods: list[ParsedPod], *, workers_per_pod: int = 1):
    return compute_worker_infos(CELL_ID, pods=pods, run=make_run(workers_per_pod=workers_per_pod))


class TestCellLiveness:
    def test_a_cell_whose_pods_all_arrived_and_are_ready_is_alive(self):
        """This is the only state in which a consumer may hand the cell work."""
        assert (
            build_cell_info([make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1)]).alive is True
        )

    def test_a_cell_still_missing_a_pod_is_not_alive(self):
        """A group being created has ready pods long before it has all of them."""
        assert build_cell_info([make_parsed_pod(pod_in_cell_index=0)]).alive is False

    def test_a_cell_that_shows_the_same_pod_index_twice_is_not_alive(self):
        """Mid-replacement the old pod's deletion may lag its replacement, and a count would call that complete."""
        pods = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=0)]

        assert build_cell_info(pods).alive is False

    def test_a_cell_whose_platform_publishes_no_size_is_judged_on_readiness_alone(self):
        """A platform that does not say how big a cell is cannot be second-guessed."""
        assert build_cell_info([make_parsed_pod(pod_in_cell_index=0, cell_size=0)]).alive is True

    def test_a_cell_with_an_unready_pod_is_not_alive(self):
        """A worker still loading its weights would drop whatever it is given."""
        pods = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1, ready=False)]

        assert build_cell_info(pods).alive is False


class TestCellMeta:
    def test_merges_the_facts_every_pod_of_the_cell_agrees_on(self):
        """A cell reports one value per key, and its pods carry the same annotations."""
        pods = [make_parsed_pod(pod_in_cell_index=index, meta={"model_id": "glm"}) for index in range(2)]

        assert build_cell_info(pods).meta == {"model_id": "glm"}

    def test_refuses_a_cell_whose_pods_disagree_about_a_key(self):
        """Whichever pod won would be whatever order the store happened to hand them back in."""
        pods = [
            make_parsed_pod(pod_in_cell_index=0, meta={"model_id": "glm"}),
            make_parsed_pod(pod_in_cell_index=1, meta={"model_id": "qwen"}),
        ]

        with pytest.raises(AssertionError, match="model_id"):
            build_cell_info(pods)


class TestWorkerInfos:
    def test_fans_a_pod_out_into_one_worker_per_worker_it_serves(self):
        """A pod runs several workers, and each of them is a Miles worker of its own."""
        pods = [make_parsed_pod(pod_in_cell_index=index, gpu_ids=(0, 1)) for index in range(2)]

        infos = build_worker_infos(pods, workers_per_pod=2)

        assert [info.name for info in infos] == [f"engine-0-{index}" for index in range(4)]
        assert [info.gpu_ids for info in infos] == [[0], [1], [0], [1]]

    def test_offsets_the_rpc_port_of_each_worker_the_way_its_process_binds_it(self):
        """The workers of a pod share its ip, so only the port tells them apart."""
        infos = build_worker_infos([make_parsed_pod(pod_in_cell_index=0, gpu_ids=(0, 1))], workers_per_pod=2)

        assert [info.self_addrs["rpc"].port for info in infos] == [8000, 8001]

    def test_a_command_worker_is_reported_without_a_handle(self):
        """An engine pod runs no rpc server, so the dashboard reads its addresses but cannot call it."""
        run = make_run(worker_class=None)

        (info,) = compute_worker_infos(CELL_ID, pods=[make_parsed_pod(pod_in_cell_index=0, cell_size=1)], run=run)

        assert info.handle is None
        assert info.self_addrs["rpc"].host == "10.0.0.1"

    def test_an_ipv6_pod_is_addressed_in_brackets(self):
        """An unbracketed v6 address makes every url built from it unparseable."""
        pod = make_parsed_pod(pod_in_cell_index=0, cell_size=1, pod_ip="fd00::5")

        (info,) = compute_worker_infos(CELL_ID, pods=[pod], run=make_run())

        assert info.self_addrs["rpc"].host == "[fd00::5]"

    def test_refuses_a_cell_that_is_missing_a_pod(self):
        """Numbering workers off a gapped pod list would name workers that belong to another pod."""
        with pytest.raises(AssertionError, match="missing pods"):
            build_worker_infos([make_parsed_pod(pod_in_cell_index=1)])
