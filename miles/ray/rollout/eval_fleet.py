"""The dedicated in-job eval engines (``--eval-num-gpus``).

Weight delivery only: ``pin`` loads a snapshot onto every engine, verifies each one
reports the expected version, and hands back the state to generate against. Who
generates is the eval fn's business, exactly as on the training engines.
"""

import asyncio
import logging
from argparse import Namespace

from miles.rollout.checkpoint_eval import EvalSkip, retarget_args
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.http_utils import wait_http_ok

logger = logging.getLogger(__name__)

EVAL_WEIGHT_LOAD_TIMEOUT_SECS = 600.0


class EvalFleet:
    """The dedicated in-job eval engines (``--eval-num-gpus``)."""

    def __init__(self, args: Namespace, *, srv):
        self.args = args
        self._srv = srv
        self._rollout_ft_enabled = args.use_fault_tolerance and "rollout" in args.ft_components
        self._state = GenerateState(self._fleet_args())

    async def pin(self, checkpoint_dir: str, weight_version: str) -> GenerateState:
        """Load the snapshot onto every engine, then return the state to generate against.

        On the manager's event loop: keep everything here awaiting rather than blocking.
        """
        try:
            if not self._rollout_ft_enabled:
                # Otherwise RolloutHealthMonitor owns the probing for these engines.
                await self._srv.probe_and_mark_dead()
            await self._srv.recover()
            await self._srv.wait_all_engines_alive()
        except Exception as e:
            logger.warning(f"Eval fleet unhealthy: {e}")
            raise EvalSkip("unhealthy") from e

        if not await self._pin_fleet(checkpoint_dir, weight_version):
            raise EvalSkip("pin_violation")

        try:
            await self._wait_router_ready()
        except Exception as e:
            logger.warning(f"Eval router not ready: {e}")
            raise EvalSkip("unhealthy") from e

        return self._state

    def _fleet_args(self) -> Namespace:
        router_ip, router_port = self.args.sglang_model_routers["eval"]
        return retarget_args(
            self.args, router_ip, router_port, self.args.eval_num_gpus, self.args.eval_num_gpus_per_engine
        )

    async def _pin_fleet(self, checkpoint_dir: str, weight_version: str, *, retries: int = 2) -> bool:
        """Load the snapshot into every fleet engine and confirm all report
        ``weight_version`` — the router load-balances across engines, so a single
        stale engine would mix versions. Never raises: transient failures and
        mismatches are retried, then ``False`` lets the caller skip the point."""
        actors = [e.actor_handle for e in self._srv.engines]
        versions: list = []
        for attempt in range(retries):
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[
                            a.update_weights_from_disk.remote(checkpoint_dir, weight_version=weight_version)
                            for a in actors
                        ]
                    ),
                    timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                )
                versions = await asyncio.wait_for(
                    asyncio.gather(*[a.get_weight_version.remote() for a in actors]),
                    timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                )
            except Exception as e:
                logger.warning(f"Weight pin to {checkpoint_dir} failed (attempt {attempt + 1}/{retries}): {e}")
                continue
            if versions and all(str(v) == weight_version for v in versions):
                return True
        logger.warning(f"Failed to pin weight_version={weight_version} to {checkpoint_dir} (got {versions})")
        return False

    async def _wait_router_ready(self, timeout: float = 180.0) -> None:
        """After a revival the router 503s until its health cycle evicts the dead
        worker; a retried one-token probe proves the route is usable before dispatch."""
        await wait_http_ok(
            f"http://{self._srv.router_ip}:{self._srv.router_port}/generate",
            json_payload={"input_ids": [0], "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
            timeout=timeout,
        )
