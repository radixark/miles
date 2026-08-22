import asyncio
from dataclasses import dataclass, field

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod, make_unlabelled_pod
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import make_pool_spec

from miles.utils.workers.reconcile.k8s_api import PodListPage, PodWatchEvent
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import HostAndPort

NAMESPACE = "rl"
SELECTOR = "app.kubernetes.io/instance=r"


@dataclass
class FakePodApi:
    pods: list = field(default_factory=list)
    events: list = field(default_factory=list)
    resource_version: str = "1"
    event_delay: float = 0.0

    selectors: list = field(default_factory=list)

    async def list_pods(self, *, namespace, label_selector):
        self.selectors.append(label_selector)
        return PodListPage(pods=list(self.pods), resource_version=self.resource_version)

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        self.selectors.append(label_selector)
        await asyncio.sleep(self.event_delay)
        for event in self.events:
            yield event
        await asyncio.sleep(3600)


@pytest.fixture(autouse=True)
def _fake_pod_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pod_api.reset()
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)


def _run(api, *, ports, worker_classes=None, spec_metas=None, workers_per_pod=None) -> KubernetesRunInfo:
    fake_pod_api.install(api)
    worker_classes = worker_classes or {}
    spec_metas = spec_metas or {}
    workers_per_pod = workers_per_pod or {}
    return KubernetesRunInfo(
        namespace=NAMESPACE,
        label_selector=SELECTOR,
        label_keys=DEFAULT_LABEL_KEYS,
        specs={
            pool_id: make_pool_spec(
                pool_id,
                ports=spec_ports,
                worker_class=worker_classes.get(pool_id),
                meta=spec_metas.get(pool_id),
                workers_per_pod=workers_per_pod.get(pool_id, 1),
            )
            for pool_id, spec_ports in ports.items()
        },
    )


def _provider(api, worker_ports=None, pool_ids=("engine",), **kwargs):
    return KubernetesWorkerProvider(
        run=_run(api, ports=worker_ports or {"engine": {"primary": 8000}}, **kwargs),
        pool_ids=list(pool_ids),
        resync_period=None,
    )


def _cell_info(provider, cell_id="engine-00000"):
    async def scenario():
        stop = await _watch(provider, [])
        try:
            return provider.cell_info(cell_id)
        finally:
            await stop()

    return asyncio.run(scenario())


async def _watch(provider, reported):
    async def reconcile(cell_id, info):
        reported.append((cell_id, info))

    return await provider.watch_cells(reconcile)


def _relabelled(pool_id):
    pod = make_pod(name="engine-0-0", labels={DEFAULT_LABEL_KEYS.pool_id: pool_id})
    return PodWatchEvent(type="MODIFIED", pod=pod, resource_version="2", rejects_cursor=False)


async def _run_watch(provider, reported):
    async def reconcile(cell_id, info):
        reported.append((cell_id, info))

    stop = await provider.watch_cells(reconcile)
    await asyncio.sleep(0.1)
    await stop()


class TestGetAddrs:
    def test_answers_a_cell_member_from_its_pod(self):
        """A cell's members only exist once scheduled, so their addresses come from observation."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return (await provider.get_addrs("engine-00000-00000"))["primary"]
            finally:
                await stop()

        assert asyncio.run(scenario()) == HostAndPort(host="10.1.2.3", port=8000)

    def test_refuses_a_worker_it_has_never_seen(self):
        """Returning a guess would send traffic to whatever happens to answer at that address."""
        provider = _provider(FakePodApi())

        async def scenario():
            stop = await _watch(provider, [])
            try:
                await provider.get_addrs("engine-00009-00009")
            finally:
                await stop()

        with pytest.raises(AssertionError, match="no observed pod serves"):
            asyncio.run(scenario())


class TestWatchCells:
    def test_reports_the_cells_that_already_existed(self):
        """A controller starting against a running release must not think the cluster is empty."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-1-0", cell_id_suffix="1")])
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert sorted(cell_id for cell_id, _ in reported) == ["engine-00000", "engine-00001"]

    def test_finishes_the_initial_listing_before_returning(self):
        """Otherwise a caller reading the cells right after cannot tell "not listed yet" from "not there"."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.cell_ids()
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-00000"]

    def test_hides_a_cell_of_another_controller(self):
        """Several controllers share a namespace, and each must see only the specs it was given."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="trainer-0-0", pool_id="trainer")])
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert [cell_id for cell_id, info in reported if info is not None] == ["engine-00000"]

    def test_reports_a_cell_that_is_not_alive_yet_as_gone(self):
        """A cell that is still starting has nothing a consumer can drive, which is what None says."""
        api = FakePodApi(
            pods=[make_pod(name="engine-0-0"), make_pod(name="engine-0-1", pod_in_cell_index="1", ready=False)]
        )
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert reported == [("engine-00000", None)]

    def test_says_nothing_about_a_cell_of_another_pool(self):
        """A pod outside this view's pools is keyed to no cell, so there is nothing to report about it."""
        api = FakePodApi(pods=[make_pod(name="trainer-0-0", pool_id="trainer")])
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert reported == []

    def test_the_api_client_is_closed_when_the_watch_stops(self):
        """Nothing else owns the session, so a watch that stops without closing it leaks one per call."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            await asyncio.sleep(0.05)
            assert fake_pod_api.CLOSE_CALLS == []
            await stop()

        asyncio.run(scenario())

        assert fake_pod_api.CLOSE_CALLS == [api]

    def test_ignores_a_pod_that_is_not_a_worker(self):
        """A namespace holds other pods, and one of them must not become a phantom cell."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_unlabelled_pod("prometheus-0")])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.cell_ids()
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-00000"]

    def test_pushes_exactly_what_a_direct_query_would_answer(self):
        """One watch and one set of pure functions: a pushed cell and a pulled one cannot be allowed to disagree."""
        api = FakePodApi(
            pods=[
                make_pod(
                    name="engine-0-0", pod_in_cell_index="0", annotations={"miles.radixark.io/meta-model_id": "glm"}
                ),
                make_pod(
                    name="engine-0-1", pod_in_cell_index="1", annotations={"miles.radixark.io/meta-model_id": "glm"}
                ),
            ]
        )
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            try:
                return provider.cell_info("engine-00000")
            finally:
                await stop()

        queried = asyncio.run(scenario())

        assert [info for _, info in reported] == [queried]


class TestWatchedPods:
    def test_asks_the_apiserver_only_for_the_pools_it_was_built_for(self):
        """A run holds every pool_id's pods, and streaming all of them into every component would not scale."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, worker_ports={"engine": {"primary": 8000}, "router": {"primary": 9000}})

        asyncio.run(_run_watch(provider, []))

        assert all(
            selector == f"{SELECTOR},{DEFAULT_LABEL_KEYS.pool_id} in (engine)" for selector in api.selectors
        ), api.selectors

    def test_names_every_pool_it_watches_in_one_selector(self):
        """One component may watch several specs, and each extra watch is another stream to the apiserver."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(
            api,
            worker_ports={"engine": {"primary": 8000}, "router": {"primary": 9000}},
            pool_ids=("router", "engine"),
        )

        asyncio.run(_run_watch(provider, []))

        assert api.selectors[0] == f"{SELECTOR},{DEFAULT_LABEL_KEYS.pool_id} in (engine,router)"

    def test_a_provider_over_no_pool_asks_for_no_pod_at_all(self):
        """A train-only run still builds the controller, and it must not end up watching the whole release."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, pool_ids=())

        asyncio.run(_run_watch(provider, []))

        assert all(selector != SELECTOR for selector in api.selectors), api.selectors


class TestWatchCellsStateCommit:
    def test_a_reported_cell_leaving_the_wanted_set_is_reported_as_gone(self):
        """From this view the cell is gone, and only None says so; dropping it silently strands the consumer."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")], events=[_relabelled("someone-else")], event_delay=0.02)
        provider = _provider(api)
        reported = []

        asyncio.run(_run_watch(provider, reported))

        assert [info is None for _, info in reported] == [False, True]
        assert [cell_id for cell_id, _ in reported] == ["engine-00000", "engine-00000"]


class TestCellInfo:
    def test_orders_the_workers_by_rank(self):
        """Consumers index this list by rank, so an arbitrary order would scramble the mapping."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-1", pod_in_cell_index="1"),
                make_pod(name="engine-0-0", pod_in_cell_index="0"),
            ]
        )
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.cell_info("engine-00000")
            finally:
                await stop()

        assert asyncio.run(scenario()).worker_names == ["engine-00000-00000", "engine-00000-00001"]

    def test_carries_the_meta_a_platform_annotated(self):
        """An engine's model id is a domain fact its consumers need and the pod is where it travels."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "glm"})])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.cell_info("engine-00000")
            finally:
                await stop()

        assert asyncio.run(scenario()).meta == {"model_id": "glm"}

    def test_names_every_rank_of_a_pod_that_serves_more_than_one(self):
        """Consumers index this list by rank, so a list of pods would hide every rank but the first of each."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-0-1", pod_in_cell_index="1")])
        provider = _provider(api, workers_per_pod={"engine": 2})

        assert _cell_info(provider).worker_names == [
            "engine-00000-00000",
            "engine-00000-00001",
            "engine-00000-00002",
            "engine-00000-00003",
        ]

    def test_still_reports_the_pods_themselves_for_the_operations_that_delete_them(self):
        """Healing recreates pods, and a rank name is not something kubernetes could delete."""
        api = FakePodApi(pods=[make_pod(name="engine-0-1", pod_in_cell_index="1"), make_pod(name="engine-0-0")])
        provider = _provider(api, workers_per_pod={"engine": 2})

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.pod_names_of_cell("engine-00000")
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-0-0", "engine-0-1"]

    def test_is_absent_for_a_cell_with_no_pods(self):
        """A cell that was deleted must read as gone rather than as an empty cell."""
        provider = _provider(FakePodApi())

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.cell_info("engine-00007")
            finally:
                await stop()

        assert asyncio.run(scenario()) is None


def _spec_meta(context) -> dict:
    return {"role": "actor", "cell_index": context.cell_index, "needs_offload": False, "model_id": "glm"}


class TestSpecMeta:
    def test_evaluates_the_meta_of_the_spec_for_every_cell_that_was_observed(self):
        """Two cells of one pool are two different cells, so one evaluation would collapse them into one."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-1-0", cell_id_suffix="1")])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        first = _cell_info(provider, cell_id="engine-00000")
        second = _cell_info(provider, cell_id="engine-00001")

        assert (first.meta["cell_index"], second.meta["cell_index"]) == (0, 1)
        assert (first.meta["role"], second.meta["role"]) == ("actor", "actor")

    def test_keeps_the_python_types_the_spec_computed(self):
        """A chart can only carry strings, which is why this meta is computed here rather than rendered."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        meta = _cell_info(provider).meta

        assert meta["needs_offload"] is False

    def test_reports_nothing_of_its_own_for_a_spec_that_declares_no_meta(self):
        """Most specs have no facts to add, and an invented key would look like a fact to a consumer."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])

        assert _cell_info(_provider(api)).meta == {}

    def test_lets_a_pod_annotation_override_a_key_the_spec_also_computed(self):
        """The pod is what a platform actually created, so its own account of itself wins."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "qwen"})])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        assert _cell_info(provider).meta["model_id"] == "qwen"

    def test_refuses_a_cell_whose_pods_annotate_one_key_differently(self):
        """A cell reports one value per key, and picking one silently would depend on the store's order."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "glm"}),
                make_pod(
                    name="engine-0-1", pod_in_cell_index="1", annotations={"miles.radixark.io/meta-model_id": "qwen"}
                ),
            ]
        )

        with pytest.raises(AssertionError, match="'model_id': \\('glm', 'qwen'\\)"):
            _cell_info(_provider(api))

    def test_accepts_pods_that_agree_about_a_key(self):
        """Every pod of a pool_id carries the same values entry, so agreement is the normal case."""
        annotations = {"miles.radixark.io/meta-model_id": "glm"}
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", annotations=annotations),
                make_pod(name="engine-0-1", pod_in_cell_index="1", annotations=annotations),
            ]
        )

        assert _cell_info(_provider(api)).meta == {"model_id": "glm"}


class _FakeTrainWorker:
    def init(self, rank: int) -> int:
        return rank

    def kill_self(self) -> None:
        return None


def _trainer_provider(api, pool_ids=("engine",), **kwargs):
    return KubernetesWorkerProvider(
        run=_run(
            api,
            ports={"engine": {"master": 9000, "rpc": 8000}},
            worker_classes={"engine": f"{__name__}._FakeTrainWorker"},
            **kwargs,
        ),
        pool_ids=list(pool_ids),
        resync_period=None,
    )


def _worker_infos(provider, cell_id="engine-00000"):
    async def scenario():
        stop = await _watch(provider, [])
        try:
            return provider.get_worker_infos(cell_ids=[cell_id])[0]
        finally:
            await stop()

    return asyncio.run(scenario())


def _worker_infos_if_generation(provider, *, expected_workers_hash: str, cell_id="engine-00000"):
    async def scenario():
        stop = await _watch(provider, [])
        try:
            return provider.get_worker_infos_if_generation(
                cell_id=cell_id,
                expected_workers_hash=expected_workers_hash,
            )
        finally:
            await stop()

    return asyncio.run(scenario())


def _worker_handle(provider, cell_id="engine-00000"):
    async def scenario():
        stop = await _watch(provider, [])
        try:
            (infos,) = provider.get_worker_infos(cell_ids=[cell_id])
            return provider.get_handles_of_worker_infos(infos)[infos[0].name]
        finally:
            await stop()

    return asyncio.run(scenario())


class TestGetWorkerInfos:
    def test_a_matching_generation_returns_workers_from_the_same_pod_snapshot(self):
        """Generation preconditions and worker selection must derive from one immutable observation."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])
        provider = _trainer_provider(api)
        expected_workers_hash = _cell_info(provider).workers_hash

        infos = _worker_infos_if_generation(provider, expected_workers_hash=expected_workers_hash)

        assert infos is not None
        assert [info.name for info in infos] == ["engine-00000-00000"]

    def test_a_stale_generation_returns_no_workers(self):
        """A stale mutation precondition must not expose any replacement worker handle to the caller."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        infos = _worker_infos_if_generation(
            _trainer_provider(api),
            expected_workers_hash="stale-generation",
        )

        assert infos is None

    def test_orders_the_workers_by_the_rank_label(self):
        """A trainer cell reads rank 0 as its master, so an arbitrary pod order would misconfigure it."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-2", pod_in_cell_index="2", pod_ip="10.0.0.3"),
                make_pod(name="engine-0-0", pod_in_cell_index="0", pod_ip="10.0.0.1"),
                make_pod(name="engine-0-1", pod_in_cell_index="1", pod_ip="10.0.0.2"),
            ]
        )

        infos = _worker_infos(_trainer_provider(api))

        assert [info.name for info in infos] == ["engine-00000-00000", "engine-00000-00001", "engine-00000-00002"]

    def test_addresses_a_worker_at_its_pod_ip_on_the_spec_ports(self):
        """Every rank has its own network namespace, so each publishes the spec's ports at its own ip."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        infos = _worker_infos(_trainer_provider(api))

        assert infos[0].self_addrs == {
            "master": HostAndPort(host="10.1.2.3", port=9000),
            "rpc": HostAndPort(host="10.1.2.3", port=8000),
        }

    def test_falls_back_to_the_headless_service_name_of_a_pod_without_an_ip(self):
        """A pod's ip appears late, but its dns name is stable from the moment the workload names it."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip=None, subdomain="engine")])

        infos = _worker_infos(_trainer_provider(api))

        assert infos[0].self_addrs["rpc"].host == f"engine-0-0.engine.{NAMESPACE}.svc"

    def test_hands_out_an_rpc_handle_pointed_at_the_worker(self):
        """A trainer cell drives its ranks through these handles, so they must talk to the right pod."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        handle = _worker_handle(_trainer_provider(api))

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._transport._server_url == "http://10.1.2.3:8000"

    def test_the_handle_knows_the_methods_of_the_worker_class(self):
        """A typo would otherwise become a 404 at call time, deep inside a training step."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])

        handle = _worker_handle(_trainer_provider(api))

        assert callable(handle.init)
        with pytest.raises(AttributeError):
            handle.__getattr__("nonexistent_method")

    def test_reports_the_gpus_a_platform_annotated(self):
        """Colocation is verified against these ids, so they travel with the worker rather than beside it."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "2,3"})])

        assert _worker_infos(_trainer_provider(api))[0].gpu_ids == [2, 3]

    def test_counts_a_pod_restart_as_a_new_worker_generation(self):
        """A restarted pod kept its name and lost its memory, and a consumer must be able to tell."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", restarts=2)])

        assert _worker_infos(_trainer_provider(api))[0].generation == 2

    def test_refuses_a_cell_that_is_missing_a_pod(self):
        """Driving half a cell would let the missing ranks' collective hang the whole run."""
        api = FakePodApi(pods=[make_pod(name="engine-0-1", pod_in_cell_index="1")])

        with pytest.raises(AssertionError, match="missing pods"):
            _worker_infos(_trainer_provider(api))

    def test_refuses_a_cell_it_has_never_observed(self):
        """Returning no workers would read as a cell with nothing to do rather than as an error."""
        with pytest.raises(AssertionError, match="no observed worker pods"):
            _worker_infos(_trainer_provider(FakePodApi()), cell_id="engine-00009")

    def test_a_worker_that_is_not_served_names_no_class_and_refuses_a_handle(self):
        """A command worker has no rpc surface at all, so the refusal belongs where one is asked for."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, worker_ports={"engine": {"rpc": 8000}})

        (info,) = _worker_infos(provider)

        assert info.worker_class is None
        with pytest.raises(AssertionError, match="is not served"):
            build_rpc_handle_of_worker_info(info)

    def test_fans_a_pod_out_into_one_worker_per_rank_it_serves(self):
        """A supervised pod runs one worker process per rank, and each of them has to be driven separately."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 3}))

        assert [info.name for info in infos] == ["engine-00000-00000", "engine-00000-00001", "engine-00000-00002"]

    def test_offsets_the_rpc_port_of_each_rank_the_way_the_process_binds_it(self):
        """serve_inner listens on port + worker_in_pod_index, so any other guess reaches the wrong rank or nothing."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

        assert [info.self_addrs["rpc"].port for info in infos] == [8000, 8001]
        assert [info.self_addrs["master"].port for info in infos] == [9000, 9000]

    def test_numbers_the_ranks_of_the_second_pod_after_those_of_the_first(self):
        """A rank's name is its index in the cell, which spans the pods rather than restarting in each."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-1", pod_in_cell_index="1", pod_ip="10.0.0.2"),
                make_pod(name="engine-0-0", pod_in_cell_index="0", pod_ip="10.0.0.1"),
            ]
        )

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

        assert [info.name for info in infos] == [
            "engine-00000-00000",
            "engine-00000-00001",
            "engine-00000-00002",
            "engine-00000-00003",
        ]
        assert [info.self_addrs["rpc"].host for info in infos] == ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"]

    def test_gives_each_rank_its_own_share_of_the_gpus_of_the_pod(self):
        """A rank takes the gpu slots at its own offset, exactly as the pod's own process numbering does."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "0,1,2,3"})])

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

        assert [info.gpu_ids for info in infos] == [[0, 1], [2, 3]]

    def test_refuses_a_pod_whose_gpus_do_not_divide_among_its_ranks(self):
        """Handing a rank a partial share would silently point two ranks at one gpu."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "0,1,2"})])

        with pytest.raises(AssertionError, match="equal share"):
            _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

    def test_resolves_the_address_of_a_rank_that_no_pod_is_named_after(self):
        """Consumers hold the rank names this provider handed out, so it has to answer for them."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])
        provider = _trainer_provider(api, workers_per_pod={"engine": 2})

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return await provider.get_addrs("engine-00000-00001")
            finally:
                await stop()

        assert asyncio.run(scenario())["rpc"] == HostAndPort(host="10.0.0.1", port=8001)

    def test_answers_for_several_cells_at_once(self):
        """A controller resynchronises every cell it owns, and one round trip per cell is the shape."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", cell_id_suffix="0"),
                make_pod(name="engine-1-0", cell_id_suffix="1"),
            ]
        )
        provider = _trainer_provider(api)

        async def scenario():
            stop = await _watch(provider, [])
            try:
                return provider.get_worker_infos(cell_ids=["engine-00000", "engine-00001"])
            finally:
                await stop()

        assert [[info.name for info in infos] for infos in asyncio.run(scenario())] == [
            ["engine-00000-00000"],
            ["engine-00001-00000"],
        ]
