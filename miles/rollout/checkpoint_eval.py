"""Eval against a dedicated eval fleet pinned to HF checkpoint snapshots.

The eval fleet never joins training weight updates; weights reach it only through
``update_weights_from_disk`` on a snapshot exported for a specific rollout_id.
"""

import asyncio
import copy
import logging
from argparse import Namespace
from typing import Protocol

from miles.rollout.inference_rollout.inference_rollout_common import GenerateState

__all__ = [
    "retarget_args",
    "make_eval_args",
    "make_eval_generate_state",
    "EvalFleetSession",
    "WeightTarget",
    "RayEngineTarget",
    "HttpServerTarget",
    "pin_and_verify",
]

logger = logging.getLogger(__name__)


def retarget_args(args: Namespace, router_ip, router_port, num_gpus: int, num_gpus_per_engine: int) -> Namespace:
    """Shallow-copy ``args`` with the router address and GPU sizing swapped for eval.

    Generate functions read the router from ``args`` and ``GenerateState`` sizes its
    semaphore off the GPU counts, so a retargeted copy runs the standard eval path
    against a different set of engines unchanged.
    """
    eval_args = copy.copy(args)
    eval_args.sglang_router_ip = router_ip
    eval_args.sglang_router_port = router_port
    eval_args.rollout_num_gpus = num_gpus
    eval_args.rollout_num_gpus_per_engine = num_gpus_per_engine
    return eval_args


def make_eval_args(args: Namespace) -> Namespace:
    router_ip, router_port = args.sglang_model_routers["eval"]
    return retarget_args(args, router_ip, router_port, args.eval_num_gpus, args.eval_num_gpus_per_engine)


def make_eval_generate_state(args: Namespace) -> GenerateState:
    return GenerateState(make_eval_args(args))


class EvalFleetSession:
    """Eval-fleet ``GenerateState``, built lazily on first use and cached.

    Lazy because the eval router only exists once the servers are up. Owned by
    ``RolloutManager`` (not by individual ``RolloutFn`` instances) — it decides
    once whether a given eval targets the fleet or the shared engines, and
    passes the resulting state through ``RolloutFnEvalInput.generate_state``.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self._state: GenerateState | None = None

    def state(self) -> GenerateState:
        if self._state is None:
            self._state = make_eval_generate_state(self.args)
        return self._state


class WeightTarget(Protocol):
    """Something that can load an HF snapshot from disk and report the
    weight_version it currently has loaded."""

    async def load_from_disk(self, hf_dir: str, weight_version: str) -> None: ...

    async def read_version(self) -> str | None: ...


class RayEngineTarget:
    """Adapts a Ray sglang engine actor handle to ``WeightTarget``."""

    def __init__(self, actor_handle):
        self._actor = actor_handle

    async def load_from_disk(self, hf_dir: str, weight_version: str) -> None:
        await self._actor.update_weights_from_disk.remote(hf_dir, weight_version=weight_version)

    async def read_version(self) -> str | None:
        return await self._actor.get_weight_version.remote()


class HttpServerTarget:
    """Adapts a bare sglang HTTP server to ``WeightTarget``."""

    def __init__(self, base_url: str):
        self._url = base_url

    async def load_from_disk(self, hf_dir: str, weight_version: str) -> None:
        from miles.utils.http_utils import post

        await post(f"{self._url}/update_weights_from_disk", {"model_path": hf_dir, "weight_version": weight_version})

    async def read_version(self) -> str | None:
        from miles.utils.http_utils import get

        info = await get(f"{self._url}/model_info")
        return info.get("weight_version")


async def pin_and_verify(
    targets: list[WeightTarget],
    hf_dir: str,
    weight_version: str,
    *,
    timeout: float = 600.0,
    retries: int = 2,
) -> bool:
    """Load ``hf_dir`` into every target and confirm all report ``weight_version``.

    Never raises: transient failures (timeout, RPC error, mismatched version)
    are retried up to ``retries`` times, then this returns ``False`` so the
    caller can decide what a failed pin means for it (skip a point, raise, ...).
    """
    versions: list = []
    for attempt in range(retries):
        try:
            await asyncio.wait_for(
                asyncio.gather(*[t.load_from_disk(hf_dir, weight_version) for t in targets]), timeout=timeout
            )
            versions = await asyncio.wait_for(asyncio.gather(*[t.read_version() for t in targets]), timeout=timeout)
        except Exception as e:
            logger.warning(f"Weight pin to {hf_dir} failed (attempt {attempt + 1}/{retries}): {e}")
            continue
        if versions and all(str(v) == weight_version for v in versions):
            return True
    logger.warning(f"Failed to pin weight_version={weight_version} to {hf_dir} (got {versions})")
    return False
