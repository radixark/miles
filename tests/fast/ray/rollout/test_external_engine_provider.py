from __future__ import annotations

from argparse import Namespace
from typing import Any

import httpx
import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import external_engine_provider as external_engine_provider_module
from miles.ray.rollout.external_engine_provider import (
    _EXTERNAL_ENGINE_POOL_ID,
    StaticInferenceEngineWorkerProvider,
    _compute_external_engine_urls,
    _discover_external_engine,
    _fetch_server_info_with_retry,
    static_inference_engine_provider,
)
from miles.ray.rollout.inference_controller import _compute_server_cell_meta_from_info
from miles.utils.workers.worker_provider.base import CellInfo


def _make_args(urls: list[str], *, num_gpus_per_engine: int = 1, **overrides: Any) -> Namespace:
    defaults = dict(
        rollout_external=True,
        rollout_external_engine_addrs=urls,
        rollout_num_gpus=len(urls) * num_gpus_per_engine,
        rollout_num_gpus_per_engine=num_gpus_per_engine,
    )
    return make_args(**{**defaults, **overrides})


def _regular_payload(*, num_gpus: int = 1) -> dict[str, Any]:
    return dict(internal_states=[dict(world_size=num_gpus)], disaggregation_mode="null")


def _prefill_payload(*, num_gpus: int = 1, bootstrap_port: int = 12090) -> dict[str, Any]:
    return {
        **_regular_payload(num_gpus=num_gpus),
        "disaggregation_mode": "prefill",
        "disaggregation_bootstrap_port": bootstrap_port,
    }


def _decode_payload(*, num_gpus: int = 1) -> dict[str, Any]:
    return {**_regular_payload(num_gpus=num_gpus), "disaggregation_mode": "decode"}


def _install_payloads(monkeypatch, payloads: dict[str, dict[str, Any]]) -> None:
    async def _fetch(url: str) -> dict[str, Any]:
        return payloads[url]

    monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)


async def _collect_watched_cells(provider: StaticInferenceEngineWorkerProvider) -> list[CellInfo]:
    observed: list[CellInfo] = []

    async def _reconcile(cell_id: str, info: CellInfo | None) -> None:
        assert info is not None and info.cell_id == cell_id
        observed.append(info)

    stop = await provider.watch_cells(_reconcile)
    await stop()
    return observed


class TestStaticInferenceEngineProviderFactory:
    def test_the_factory_builds_the_provider_from_args_alone(self, monkeypatch):
        """The loadable factory contract passes capability, which the static provider has no use for."""
        recorded: dict[str, Any] = {}

        class _Recording:
            def __init__(self, *, args: Namespace) -> None:
                recorded["args"] = args

        monkeypatch.setattr(external_engine_provider_module, "StaticInferenceEngineWorkerProvider", _Recording)
        args = _make_args(["host1:8000"])

        provider = static_inference_engine_provider(args, capability=object())

        assert isinstance(provider, _Recording)
        assert recorded["args"] is args


class TestComputeExternalEngineUrls:
    def test_addresses_are_normalized_to_http_urls(self):
        """host:port and http://host:port/ must both land on one canonical url form."""
        args = _make_args(["host1:8000", "http://host2:8001/"])

        assert _compute_external_engine_urls(args) == ["http://host1:8000", "http://host2:8001"]

    def test_an_address_without_a_port_is_rejected(self):
        """The engine port cannot be guessed, so a bare hostname must fail fast."""
        args = _make_args(["host1"])

        with pytest.raises(AssertionError, match="invalid external engine address"):
            _compute_external_engine_urls(args)

    def test_the_same_engine_listed_twice_is_rejected(self):
        """One engine behind two cells registers with the router twice and eats two rank ranges."""
        args = _make_args(["host1:8000", "http://host1:8000"])

        with pytest.raises(AssertionError, match="more than once"):
            _compute_external_engine_urls(args)

    @pytest.mark.parametrize(
        "aliases",
        [
            ["HOST1:8000", "http://host1:8000/"],
            ["HTTP://host1:8000", "host1:8000"],
            ["[fd00:0:0::1]:8000", "[fd00::1]:8000"],
        ],
    )
    def test_two_spellings_of_one_endpoint_are_rejected(self, aliases):
        """Deduplicating raw strings would let a case or ipv6 alias dial one engine from two cells,
        which then claims two rank ranges in the weight-update group and stalls the rendezvous."""
        args = _make_args(aliases)

        with pytest.raises(AssertionError, match="more than once"):
            _compute_external_engine_urls(args)

    def test_addresses_are_reported_in_their_canonical_form(self):
        """The canonical url is what the cell dials and what workers_hash identifies it by."""
        args = _make_args(["HOST1:8000", "[fd00:0:0::2]:8001"])

        assert _compute_external_engine_urls(args) == ["http://host1:8000", "http://[fd00::2]:8001"]


class TestDiscoverExternalEngine:
    async def test_num_gpus_comes_from_the_world_size_the_engine_reports(self, monkeypatch):
        """The parallel sizes on their own do not give the total, and deriving it here would put a
        copy of sglang's own formula in miles for every dimension it grows."""
        _install_payloads(monkeypatch, {"http://host1:8000": _regular_payload(num_gpus=8)})

        engine = await _discover_external_engine("http://host1:8000")

        assert (engine.num_gpus, engine.worker_type) == (8, "regular")
        assert (engine.host, engine.port) == ("host1", 8000)

    async def test_a_prefill_engine_reports_its_type_and_bootstrap_port(self, monkeypatch):
        """PD registration needs both the role and the bootstrap port from discovery."""
        _install_payloads(
            monkeypatch,
            {"http://host1:8000": {**_prefill_payload(num_gpus=2), "disaggregation_bootstrap_port": "12090"}},
        )

        engine = await _discover_external_engine("http://host1:8000")

        assert (engine.worker_type, engine.disaggregation_bootstrap_port) == ("prefill", 12090)

    async def test_a_decode_engine_reports_its_type_without_a_bootstrap_port(self, monkeypatch):
        """Only prefill engines own a bootstrap port; decode must come back None."""
        _install_payloads(monkeypatch, {"http://host1:8000": _decode_payload(num_gpus=4)})

        engine = await _discover_external_engine("http://host1:8000")

        assert (engine.worker_type, engine.disaggregation_bootstrap_port) == ("decode", None)


class _FakeApiClient:
    calls: list[str] = []
    answers: list[Any] = []

    def __init__(self, *, server_url: str) -> None:
        self.server_url = server_url

    async def get_server_info(self) -> dict[str, Any]:
        _FakeApiClient.calls.append(self.server_url)
        answer = _FakeApiClient.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def fake_api_client(monkeypatch):
    _FakeApiClient.calls = []
    _FakeApiClient.answers = []
    monkeypatch.setattr(external_engine_provider_module, "SGLangApiClient", _FakeApiClient)
    return _FakeApiClient


class TestFetchServerInfoWithRetry:
    async def test_discovery_waits_for_an_engine_that_is_still_booting(self, fake_api_client):
        """An engine that answers on the second try must not fail the run."""
        fake_api_client.answers = [httpx.ConnectError("still booting"), _regular_payload()]

        payload = await _fetch_server_info_with_retry("http://host1:8000")

        assert payload == _regular_payload()
        assert fake_api_client.calls == ["http://host1:8000", "http://host1:8000"]

    async def test_discovery_gives_up_after_the_deadline(self, fake_api_client):
        """A dead address must raise instead of retrying forever."""
        fake_api_client.answers = [httpx.ConnectError("nobody home")]

        with pytest.raises(TimeoutError, match="server_info"):
            await _fetch_server_info_with_retry("http://host1:8000", timeout_seconds=0.01)


class TestStaticInferenceEngineWorkerProvider:
    async def _make_provider(self, monkeypatch, args: Namespace, payloads: dict[str, dict[str, Any]]):
        _install_payloads(monkeypatch, payloads)
        provider = StaticInferenceEngineWorkerProvider(args=args)
        await provider.init()
        return provider

    async def test_one_unreachable_engine_publishes_no_cells_at_all(self, monkeypatch):
        """A half fleet would still satisfy the router and the weight updater, which would then wait
        on ranks the missing engine was supposed to claim."""
        args = _make_args(["host1:8000", "host2:8000"])

        async def _fetch(url: str) -> dict[str, Any]:
            if url == "http://host2:8000":
                raise TimeoutError(f"External engine {url} did not answer /server_info")
            return _regular_payload()

        monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)
        provider = StaticInferenceEngineWorkerProvider(args=args)

        with pytest.raises(TimeoutError):
            await provider.init()

        with pytest.raises(AssertionError, match="which has not run"):
            _ = provider.cell_infos

    async def test_each_url_becomes_one_cell_with_the_discovered_meta(self, monkeypatch):
        """The cell meta must carry what the controller reads, sourced from discovery."""
        args = _make_args(["host1:8000"], num_gpus_per_engine=2, sglang_api_key="secret")

        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload(num_gpus=2)})

        (info,) = provider.cell_infos
        assert info.pool_id == _EXTERNAL_ENGINE_POOL_ID
        assert info.meta == dict(
            model_id="default",
            worker_type="regular",
            num_gpus_per_engine=2,
            gpu_offset=0,
            sglang_api_key="secret",
            needs_offload=False,
            update_weights=True,
        )

    async def test_gpu_offsets_accumulate_linearly_in_listed_order(self, monkeypatch):
        """The weight-update NCCL layout assigns each engine the ranks after its predecessors."""
        args = _make_args(["host1:8000", "host2:8000", "host3:8000"], rollout_num_gpus=7)
        payloads = {
            "http://host1:8000": _regular_payload(num_gpus=2),
            "http://host2:8000": _regular_payload(num_gpus=4),
            "http://host3:8000": _regular_payload(num_gpus=1),
        }

        provider = await self._make_provider(monkeypatch, args, payloads)

        offsets = [info.meta["gpu_offset"] for info in provider.cell_infos]
        assert offsets == [0, 2, 6]

    async def test_a_fleet_with_a_different_gpu_total_is_rejected(self, monkeypatch):
        """--rollout-num-gpus sizes the placement group and the router, so a fleet that is smaller
        than it claims must fail at startup instead of hanging in NCCL."""
        args = _make_args(["host1:8000", "host2:8000"], num_gpus_per_engine=4)
        payloads = {
            "http://host1:8000": _regular_payload(num_gpus=4),
            "http://host2:8000": _regular_payload(num_gpus=2),
        }

        with pytest.raises(AssertionError, match="6 gpus in total"):
            await self._make_provider(monkeypatch, args, payloads)

    async def test_a_pd_fleet_without_the_router_flag_is_rejected(self, monkeypatch):
        """The router was already launched non-PD, so serving a PD fleet behind it would misroute."""
        args = _make_args(["host1:8000", "host2:8000"], rollout_num_gpus=2)
        payloads = {
            "http://host1:8000": _prefill_payload(),
            "http://host2:8000": _decode_payload(),
        }

        with pytest.raises(AssertionError, match="rollout-external-router-pd"):
            await self._make_provider(monkeypatch, args, payloads)

    async def test_the_router_flag_without_a_pd_fleet_is_rejected(self, monkeypatch):
        """A PD router in front of regular engines is a misdeclaration, not a fleet to serve."""
        args = _make_args(["host1:8000"], rollout_external_router_pd=True)

        with pytest.raises(AssertionError, match="are none"):
            await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

    async def test_a_declared_pd_fleet_carries_roles_and_bootstrap_ports(self, monkeypatch):
        """External PD needs the discovered role and bootstrap port on every cell."""
        args = _make_args(["host1:8000", "host2:8000"], rollout_num_gpus=6, rollout_external_router_pd=True)
        payloads = {
            "http://host1:8000": _prefill_payload(num_gpus=2, bootstrap_port=12090),
            "http://host2:8000": _decode_payload(num_gpus=4),
        }

        provider = await self._make_provider(monkeypatch, args, payloads)

        assert [info.meta["worker_type"] for info in provider.cell_infos] == ["prefill", "decode"]

    async def test_expected_num_cells_counts_the_urls_of_the_model(self, monkeypatch):
        """The startup barrier waits for exactly the announced fleet, one cell per url."""
        args = _make_args(["host1:8000", "host2:8000"])
        payloads = {
            "http://host1:8000": _regular_payload(),
            "http://host2:8000": _regular_payload(),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        assert provider.expected_num_cells(model_id="default") == 2
        assert provider.expected_num_cells(model_id="ghost") == 0

    async def test_the_first_announcement_completes_before_the_watch_is_established(self, monkeypatch):
        """The controller treats a returned watch as the initial sync being done."""
        args = _make_args(["host1:8000", "host2:8000"])
        payloads = {
            "http://host1:8000": _regular_payload(),
            "http://host2:8000": _regular_payload(),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        observed = await _collect_watched_cells(provider)

        assert [info.cell_id for info in observed] == [info.cell_id for info in provider.cell_infos]

    async def test_get_addrs_answers_the_primary_engine_address(self, monkeypatch):
        """The cell dials its node-0 engine through exactly this address."""
        args = _make_args(["host1:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        (info,) = provider.cell_infos
        addrs = await provider.get_addrs(worker_name=info.worker_names[0])

        assert (addrs["primary"].host, addrs["primary"].port) == ("host1", 8000)
        assert "gate" not in addrs

    async def test_a_prefill_engines_addrs_carry_the_bootstrap_port(self, monkeypatch):
        """The router needs the bootstrap port when it registers a prefill worker."""
        args = _make_args(["host1:8000", "host2:8000"], rollout_num_gpus=2, rollout_external_router_pd=True)
        payloads = {
            "http://host1:8000": _prefill_payload(bootstrap_port=12090),
            "http://host2:8000": _decode_payload(),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        (prefill_info, _decode_info) = provider.cell_infos
        addrs = await provider.get_addrs(worker_name=prefill_info.worker_names[0])

        assert addrs["disaggregation_bootstrap"].port == 12090

    async def test_asking_for_an_unknown_worker_is_rejected(self, monkeypatch):
        """Answering a made-up name with a guess would dial the wrong machine."""
        args = _make_args(["host1:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        with pytest.raises(AssertionError, match="not one of the external engines"):
            await provider.get_addrs(worker_name="ghost-0-0")

    async def test_worker_infos_expose_addresses_but_no_rpc_handle(self, monkeypatch):
        """External engines are plain http servers; nothing can call rpc methods on them."""
        args = _make_args(["host1:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        (info,) = provider.cell_infos
        ((worker_info,),) = provider.get_worker_infos(cell_ids=[info.cell_id])

        assert worker_info.name == info.worker_names[0]
        assert worker_info.worker_class is None
        assert worker_info.self_addrs["primary"].port == 8000

    async def test_an_ipv6_engine_address_stays_bracketed(self, monkeypatch):
        """An unbracketed ipv6 host would render an unparseable server url."""
        args = _make_args(["http://[fd00::1]:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://[fd00::1]:8000": _regular_payload()})

        (info,) = provider.cell_infos
        addrs = await provider.get_addrs(worker_name=info.worker_names[0])

        assert addrs["primary"].host == "[fd00::1]"

    async def test_the_cell_info_satisfies_the_server_cell_meta_contract(self, monkeypatch):
        """A provider whose meta drifts from ServerCellMetadata would crash reconcile at runtime."""
        args = _make_args(["host1:8000"], num_gpus_per_engine=2, rollout_external_router_pd=True)
        payload = _decode_payload(num_gpus=2)
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": payload})

        (info,) = provider.cell_infos
        meta = _compute_server_cell_meta_from_info(info)

        assert (meta.worker_type, meta.num_gpus_per_engine, meta.workers_hash) == (
            "decode",
            2,
            "http://host1:8000",
        )
        assert meta.worker_name == info.worker_names[0]
