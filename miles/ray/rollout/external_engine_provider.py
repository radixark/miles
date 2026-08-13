import asyncio
import ipaddress
import logging
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.http_utils import _wrap_ipv6
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

logger = logging.getLogger(__name__)

DISCOVERY_TIMEOUT_SECONDS = 600.0

_EXTERNAL_ENGINE_POOL_ID = "external-inference-engine-0"
_EXTERNAL_MODEL_ID = "default"


# ================================== provider ==================================


def static_inference_engine_provider(
    args: Any, *, capability: BackendCapability
) -> "StaticInferenceEngineWorkerProvider":
    return StaticInferenceEngineWorkerProvider(args=args)


@dataclass(frozen=True)
class _ExternalCell:
    cell_info: CellInfo
    addrs: NamedHostAndPorts


class StaticInferenceEngineWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, args: Any) -> None:
        self._args = args
        self._cells: dict[str, _ExternalCell] | None = None
        self._cell_id_by_worker_name: dict[str, str] = {}

    async def init(self) -> None:
        urls = _compute_external_engine_urls(self._args)
        engines = await _discover_external_engines(urls)
        _assert_engines_match_args(self._args, engines=engines)

        self._cells = _compute_cells(args=self._args, engines=engines)
        self._cell_id_by_worker_name = {
            worker_name: cell_id
            for cell_id, cell in self._cells.items()
            for worker_name in cell.cell_info.worker_names
        }

        logger.info(f"Discovered external inference engines: {self.cell_infos}")

    @property
    def cell_infos(self) -> list[CellInfo]:
        return [cell.cell_info for cell in self._initialized_cells.values()]

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        assert worker_name in self._cell_id_by_worker_name, (
            f"{worker_name} is not one of the external engines this provider was given "
            f"({sorted(self._cell_id_by_worker_name)})"
        )
        return self._initialized_cells[self._cell_id_by_worker_name[worker_name]].addrs

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        cells = self._initialized_cells
        return [
            [
                WorkerInfo(
                    name=worker_name,
                    generation=0,
                    self_addrs=cells[cell_id].addrs,
                    gpu_ids=[],
                )
                for worker_name in cells[cell_id].cell_info.worker_names
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        for cell_id, cell in self._initialized_cells.items():
            await reconcile(cell_id, cell.cell_info)

        async def _stop() -> None:
            pass

        return _stop

    def expected_num_cells(self, *, model_id: str) -> int:
        return sum(1 for cell in self._initialized_cells.values() if cell.cell_info.meta["model_id"] == model_id)

    @property
    def _initialized_cells(self) -> dict[str, _ExternalCell]:
        assert self._cells is not None, "the provider discovers its engines in init(), which has not run"
        return self._cells


# ============================= cell construction ==============================


@dataclass(frozen=True)
class _ExternalEngineInfo:
    url: str
    host: str
    port: int
    worker_type: Literal["regular", "prefill", "decode"]
    num_gpus: int
    disaggregation_bootstrap_port: int | None


def _compute_cells(*, args: Any, engines: list[_ExternalEngineInfo]) -> dict[str, _ExternalCell]:
    cells: dict[str, _ExternalCell] = {}
    gpu_offset = 0
    for cell_index, engine in enumerate(engines):
        cell_id = compute_cell_id(pool_id=_EXTERNAL_ENGINE_POOL_ID, cell_index=cell_index)
        worker_name = compute_worker_name(pool_id=_EXTERNAL_ENGINE_POOL_ID, cell_index=cell_index)
        cells[cell_id] = _ExternalCell(
            cell_info=CellInfo(
                cell_id=cell_id,
                pool_id=_EXTERNAL_ENGINE_POOL_ID,
                alive=True,
                worker_names=[worker_name],
                workers_hash=engine.url,
                meta=dict(
                    model_id=_EXTERNAL_MODEL_ID,
                    worker_type=engine.worker_type,
                    num_gpus_per_engine=engine.num_gpus,
                    gpu_offset=gpu_offset,
                    sglang_api_key=args.sglang_api_key,
                    needs_offload=False,
                    update_weights=True,
                ),
            ),
            addrs=_compute_engine_addrs(engine),
        )
        gpu_offset += engine.num_gpus
    return cells


def _compute_engine_addrs(engine: _ExternalEngineInfo) -> NamedHostAndPorts:
    addrs: NamedHostAndPorts = dict(primary=HostAndPort(host=engine.host, port=engine.port))
    if engine.disaggregation_bootstrap_port is not None:
        addrs["disaggregation_bootstrap"] = HostAndPort(host=engine.host, port=engine.disaggregation_bootstrap_port)
    return addrs


# ============================ address book parsing ============================


def _compute_external_engine_urls(args: Any) -> list[str]:
    addrs: list[str] = args.rollout_external_engine_addrs
    assert addrs, "--rollout-external-engine-addrs must name at least one engine"

    urls = [_normalize_external_engine_url(addr) for addr in addrs]

    duplicates = sorted({url for url in urls if urls.count(url) > 1})
    assert not duplicates, (
        f"--rollout-external-engine-addrs lists {duplicates} more than once; one engine would be registered "
        f"with the router twice and would take two rank slots in the weight-update group"
    )
    return urls


def _normalize_external_engine_url(addr: str) -> str:
    url = addr if "://" in addr else f"http://{addr}"
    addr_and_port = _parse_external_engine_url(url.rstrip("/"), source=addr)
    return f"http://{addr_and_port.host}:{addr_and_port.port}"


def _parse_external_engine_url(url: str, *, source: str | None = None) -> HostAndPort:
    parsed = urlsplit(url)
    assert parsed.scheme == "http" and parsed.hostname is not None and parsed.port is not None, (
        f"invalid external engine address {source or url!r}: use host:port or http://host:port "
        f"(bracket IPv6 literals)"
    )
    return HostAndPort(host=_wrap_ipv6(_canonicalize_host(parsed.hostname)), port=parsed.port)


def _canonicalize_host(hostname: str) -> str:
    try:
        return ipaddress.ip_address(hostname).compressed
    except ValueError:
        return hostname


# ================================= discovery ==================================


async def _discover_external_engines(urls: list[str]) -> list[_ExternalEngineInfo]:
    return list(await asyncio.gather(*[_discover_external_engine(url) for url in urls]))


async def _discover_external_engine(url: str) -> _ExternalEngineInfo:
    addr = _parse_external_engine_url(url)
    server_info = await _fetch_server_info_with_retry(url)

    return _ExternalEngineInfo(
        url=url,
        host=addr.host,
        port=addr.port,
        worker_type=_infer_worker_type(server_info),
        num_gpus=int(server_info["internal_states"][0]["world_size"]),
        disaggregation_bootstrap_port=(
            int(x) if (x := server_info.get("disaggregation_bootstrap_port")) is not None else None
        ),
    )


async def _fetch_server_info_with_retry(
    url: str, *, timeout_seconds: float = DISCOVERY_TIMEOUT_SECONDS
) -> dict[str, Any]:
    async def _attempt(_remaining_seconds: float) -> dict[str, Any]:
        return await SGLangApiClient(server_url=url).get_server_info()

    try:
        return await retry_until_deadline(
            _attempt,
            total_seconds=timeout_seconds,
            retry_on=(httpx.HTTPError, OSError),
            log_fields=dict(op="discover_external_engine", url=url),
        )
    except (httpx.HTTPError, OSError) as e:
        raise TimeoutError(f"External engine {url} did not answer /server_info within {timeout_seconds}s") from e


def _infer_worker_type(server_info: dict[str, Any]) -> Literal["regular", "prefill", "decode"]:
    mode = server_info.get("disaggregation_mode")
    if mode in ("prefill", "decode"):
        return mode
    return "regular"


# ============================== args cross-check ==============================


def _assert_engines_match_args(args: Any, *, engines: list[_ExternalEngineInfo]) -> None:
    discovered_total = sum(engine.num_gpus for engine in engines)
    assert discovered_total == args.rollout_num_gpus, (
        f"the external engines report {discovered_total} gpus in total "
        f"({[(engine.url, engine.num_gpus) for engine in engines]}), but --rollout-num-gpus is "
        f"{args.rollout_num_gpus}. That argument sizes the placement group and the router, so let it "
        f"describe the fleet that is actually running"
    )

    pd_engine_urls = [engine.url for engine in engines if engine.worker_type != "regular"]
    assert bool(pd_engine_urls) == args.rollout_external_router_pd, (
        f"the router is launched in PD mode iff --rollout-external-router-pd is set "
        f"({args.rollout_external_router_pd}), but the engines reporting prefill/decode are "
        f"{pd_engine_urls or 'none'}"
    )
