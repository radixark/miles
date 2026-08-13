"""The dedicated in-job eval engines (``--eval-num-gpus``).

Weight delivery only: ``pin`` loads a snapshot onto every engine and verifies each
one reports the expected version. The fleet lives beside its engines, in the
inference controller; the executor holds an ``RolloutExecutorEvalFleet`` over the wire and
builds the state to generate against from the fleet's description. Who generates is
the eval fn's business, exactly as on the training engines.
"""

import asyncio
import dataclasses
import logging
from argparse import Namespace

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.specs.inference import inference_controller_worker_name
from miles.rollout.checkpoint_eval import EvalSkip, retarget_args
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.http_utils import wait_http_ok
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

EVAL_WEIGHT_LOAD_TIMEOUT_SECS = 600.0


@dataclasses.dataclass(frozen=True)
class EvalFleetInfo:
    router: HostAndPort
    num_gpus: int
    num_gpus_per_engine: int


@dataclasses.dataclass(frozen=True)
class EvalFleetPin:
    skip_reason: str | None


class RolloutExecutorEvalFleet:
    """The executor's side of the fleet: one state to generate against, pinned over rpc."""

    def __init__(
        self, args: Namespace, *, info: EvalFleetInfo, inference_controller_provider: BaseWorkerProvider
    ) -> None:
        self._inference_controller_provider = inference_controller_provider
        self._state = GenerateState(
            retarget_args(args, info.router.host, info.router.port, info.num_gpus, info.num_gpus_per_engine)
        )

    async def pin(self, checkpoint_dir: str, weight_version: str) -> GenerateState:
        inference_controller = self._inference_controller_provider.get_handle(inference_controller_worker_name())
        pin = await inference_controller.pin_eval_fleet(checkpoint_dir=checkpoint_dir, weight_version=weight_version)

        if (skip_reason := pin.skip_reason) is not None:
            raise EvalSkip(skip_reason)
        return self._state


class InferenceControllerEvalFleet:
    """The dedicated in-job eval engines (``--eval-num-gpus``)."""

    def __init__(self, args: Namespace, *, srv):
        self.args = args
        self._srv = srv

    @property
    def info(self) -> EvalFleetInfo:
        return EvalFleetInfo(
            router=HostAndPort(host=self._srv.router_ip, port=self._srv.router_port),
            num_gpus=self.args.eval_num_gpus,
            num_gpus_per_engine=self.args.eval_num_gpus_per_engine,
        )

    async def pin(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        """Load the snapshot onto every engine, then report whether it can be generated against.

        On the controller's event loop: keep everything here awaiting rather than blocking.
        """
        if not await self._pin_fleet(checkpoint_dir, weight_version):
            return EvalFleetPin(skip_reason="pin_violation")

        try:
            await self._wait_router_ready()
        except Exception as e:
            logger.warning(f"Eval router not ready: {e}")
            return EvalFleetPin(skip_reason="unhealthy")

        return EvalFleetPin(skip_reason=None)

    async def _pin_fleet(self, checkpoint_dir: str, weight_version: str, *, retries: int = 2) -> bool:
        """Load the snapshot into every fleet engine and confirm all report
        ``weight_version`` — the router load-balances across engines, so a single
        stale engine would mix versions. Never raises: transient failures and
        mismatches are retried, then ``False`` lets the caller skip the point."""
        versions: list = []
        for attempt in range(retries):
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[
                            client.update_weights_from_disk(model_path=checkpoint_dir, weight_version=weight_version)
                            for client in await self._fleet_api_clients()
                        ]
                    ),
                    timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                )
                versions = await asyncio.wait_for(
                    asyncio.gather(*[client.get_weight_version() for client in await self._fleet_api_clients()]),
                    timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                )
            except Exception as e:
                logger.warning(f"Weight pin to {checkpoint_dir} failed (attempt {attempt + 1}/{retries}): {e}")
                continue
            if versions and all(str(v) == weight_version for v in versions):
                return True
        logger.warning(f"Failed to pin weight_version={weight_version} to {checkpoint_dir} (got {versions})")
        return False

    async def _fleet_api_clients(self) -> list[SGLangApiClient]:
        async with self._srv.context_lock:
            return list(self._srv.api_clients)

    async def _wait_router_ready(self, timeout: float = 180.0) -> None:
        """After a revival the router 503s until its health cycle evicts the dead
        worker; a retried one-token probe proves the route is usable before dispatch."""
        await wait_http_ok(
            f"http://{self._srv.router_ip}:{self._srv.router_port}/generate",
            json_payload={"input_ids": [0], "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
            timeout=timeout,
        )
