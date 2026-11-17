"""Engines carved out of the training model for one eval (``--eval-pool-num-engines``).

For one eval, ``carve_out`` moves the least-loaded serving engines from the training router to an
eval router that starts empty: they keep the weights they hold (the version the eval
measures), and as cells pinned for eval they are no weight receivers, so training keeps
updating every other engine. ``give_back`` takes them off the eval router as cells pending
weights; the next weight update reaches them, and only then do they rejoin the training
router. The executor reads the pool through the same ``info`` / ``check_version`` surface
as the dedicated eval engines.
"""

import asyncio
import dataclasses
import logging
from argparse import Namespace

from miles.ray.rollout.eval_fleet import (
    EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
    EvalFleetInfo,
    EvalFleetPin,
    compute_probe_headers,
    wait_router_generate_ready,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class EvalPoolCarveOut:
    """The weight version the carved engines hold, or why no engine was carved out."""

    weight_version: str | None
    skip_reason: str | None


class InferenceControllerEvalPool:
    """The controller's side of the carved-out eval engines.

    ``carve_out`` and ``give_back`` run under the controller's lock, so they never interleave with a weight update.
    """

    def __init__(self, args: Namespace, *, srv: RolloutServer, router: HostAndPort):
        self._num_engines = args.eval_pool_num_engines
        self._num_gpus_per_engine = args.rollout_num_gpus_per_engine
        self._srv = srv
        self._router = router
        self._probe_headers = compute_probe_headers(args)

    @property
    def _router_url(self) -> str:
        return f"http://{self._router.host}:{self._router.port}"

    async def info(self) -> EvalFleetInfo:
        return EvalFleetInfo(router=self._router, engine_gpu_counts=[self._num_gpus_per_engine] * self._num_engines)

    async def carve_out(self) -> EvalPoolCarveOut:
        """Pin the ``num_engines`` least-loaded serving cells for one eval, all at one weight version; a failure skips the eval."""
        try:
            # cells a failed give-back left pinned get another try here, so they do not stay out of training
            await self._srv.give_back_eval_pinned_cells()
            return await self._pin_serving_cells()
        except Exception:
            logger.exception("Carving out engines for eval failed; skipping this eval")
            return EvalPoolCarveOut(weight_version=None, skip_reason="unhealthy")

    async def _pin_serving_cells(self) -> EvalPoolCarveOut:
        serving = [cell for cell in self._srv.all_server_cells.values() if cell.is_serving]
        # training keeps at least one engine
        if len(serving) <= self._num_engines:
            logger.warning(f"Only {len(serving)} serving engines, cannot carve out {self._num_engines} for eval")
            return EvalPoolCarveOut(weight_version=None, skip_reason="insufficient_engines")
        cells = await _least_loaded(serving, num_cells=self._num_engines)

        versions = await _weight_versions(cells)
        if len(set(versions)) != 1:
            logger.warning(f"Engines to carve out for eval hold different weight versions: {versions}")
            return EvalPoolCarveOut(weight_version=None, skip_reason="pin_violation")

        try:
            for cell in cells:
                await cell.pin_for_eval(eval_router_url=self._router_url)
            await wait_router_generate_ready(self._router_url, headers=self._probe_headers)
        except Exception:
            await self._srv.give_back_eval_pinned_cells()
            raise
        return EvalPoolCarveOut(weight_version=versions[0], skip_reason=None)

    async def give_back(self) -> None:
        await self._srv.give_back_eval_pinned_cells()

    async def check_version(self, weight_version: str) -> EvalFleetPin:
        """Every pinned cell still holds ``weight_version``; never reload or retry weights."""
        try:
            async with self._srv.context_lock:
                cells = list(self._srv.eval_pinned_cells.values())
            versions = await _weight_versions(cells)
            if not versions or any(version != weight_version for version in versions):
                return EvalFleetPin(skip_reason="pin_violation")
            await wait_router_generate_ready(self._router_url, headers=self._probe_headers)
        except Exception as e:
            logger.warning(f"Carved-out eval engines are not ready: {e}")
            return EvalFleetPin(skip_reason="unhealthy")
        return EvalFleetPin(skip_reason=None)


async def _least_loaded(cells: list[ServerCell], *, num_cells: int) -> list[ServerCell]:
    """The cells with the fewest unfinished requests.

    Training requests a carved engine still runs keep their slots, and eval requests beyond
    the free slots wait in the engine queue, so lending the least-loaded engines shortens that wait.
    """
    responses = await asyncio.gather(*[cell.api_client.get_loads() for cell in cells])
    num_requests = [
        sum(load["num_running_reqs"] + load["num_waiting_reqs"] for load in response["loads"])
        for response in responses
    ]
    logger.info(f"Unfinished requests per serving engine before carving out for eval: {num_requests}")
    ranked = sorted(zip(num_requests, cells, strict=True), key=lambda item: (item[0], -item[1].meta.gpu_offset))
    return [cell for _, cell in ranked[:num_cells]]


async def _weight_versions(cells: list[ServerCell]) -> list[str]:
    versions = await asyncio.wait_for(
        asyncio.gather(*[cell.api_client.get_weight_version() for cell in cells]),
        timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
    )
    return [str(version) for version in versions]
