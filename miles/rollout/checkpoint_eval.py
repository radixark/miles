"""Eval that consumes exported HF checkpoint snapshots.

Two eval postures exist: against the live training engines (shared, pinned by
blocking call order) or against a checkpoint file (pinned by the file itself).
``CheckpointEvalFn`` is the contract for the second posture; ``FleetEvalFn`` is
its in-job implementation (the dedicated fleet), external backends subclass it.
Backends never join training weight updates; weights reach them only through a
snapshot exported for a specific rollout_id.
"""

import abc
import asyncio
import copy
import inspect
import logging
from argparse import Namespace
from typing import Protocol

import ray

from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnInput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState

__all__ = [
    "retarget_args",
    "make_eval_args",
    "make_eval_generate_state",
    "EvalSkip",
    "CheckpointEvalFn",
    "FleetEvalFn",
    "resolve_checkpoint_eval_fn",
    "WeightTarget",
    "RayEngineTarget",
    "HttpServerTarget",
    "pin_and_verify",
]

logger = logging.getLogger(__name__)

EVAL_WEIGHT_LOAD_TIMEOUT_SECS = 600.0


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


class EvalSkip(Exception):
    """Raise from a ``CheckpointEvalFn`` to skip this eval point with an attributable
    reason (logged as ``eval/skipped_{reason}``) instead of counting as a crash."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CheckpointEvalFn(abc.ABC):
    """Contract for eval backends that consume HF checkpoint snapshots.

    ``__init__`` prepares everything (launch or attach to your backend); each call
    then receives a snapshot dir + eval info and returns the eval results. The
    trainer owns the rest: per-point snapshot export, async dispatch, overflow
    policy, logging at the snapshot's step, and snapshot GC.

    Subclass and implement ``evaluate_checkpoint``; raise ``EvalSkip(reason)`` to
    skip a point with proper accounting. Point ``--eval-function-path`` at the
    subclass (requires ``train_async.py`` and a snapshot source: ``--eval-hf-dir``
    or ``--save-hf``). See ``examples/fully_async/external_eval_fn.py`` for a full
    implementation against an external sglang server; ``FleetEvalFn`` below is the
    in-job flavor (constructed by ``RolloutManager``, not via the CLI flag).
    """

    @abc.abstractmethod
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput: ...

    async def __call__(self, input: RolloutFnInput) -> RolloutFnEvalOutput:
        assert input.evaluation, "CheckpointEvalFn only serves eval; keep the train fn on --rollout-function-path"
        assert input.hf_dir is not None, (
            "no snapshot was dispatched — checkpoint eval fns require train_async.py "
            "and a snapshot source (--eval-hf-dir or --save-hf)"
        )
        return await self.evaluate_checkpoint(input.hf_dir, input)

    def dispose(self) -> None:  # noqa: B027 — optional hook, deliberately a no-op default
        """Tear down anything launched in ``__init__``. Called by RolloutManager.dispose()."""


class FleetEvalFn(CheckpointEvalFn):
    """The dedicated in-job eval fleet (``--eval-num-gpus``) as a checkpoint backend.

    Pins every fleet engine to the snapshot (probing and reviving dead engines
    first), then delegates generation to the inner eval fn with the fleet's
    ``GenerateState`` — so custom eval fns work on the fleet unchanged. Privileged:
    constructed by ``RolloutManager`` with the fleet's server handle, not via
    ``--eval-function-path``.
    """

    def __init__(self, args: Namespace, srv, inner):
        self.args = args
        self._srv = srv
        self._inner = inner
        # Lazy: the eval router only exists once the servers are up.
        self._state: GenerateState | None = None

    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        try:
            await self._mark_unreachable_engines()
            await self._srv.recover()
            await self._srv.wait_all_engines_alive()
        except Exception as e:
            logger.warning(f"Eval fleet unhealthy at rollout {input.rollout_id}: {e}")
            raise EvalSkip("unhealthy") from e

        targets = [RayEngineTarget(e.actor_handle) for e in self._srv.engines]
        if not await pin_and_verify(
            targets, checkpoint_dir, input.weight_version, timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS
        ):
            raise EvalSkip("pin_violation")

        try:
            await self._wait_router_ready()
        except Exception as e:
            logger.warning(f"Eval router not ready at rollout {input.rollout_id}: {e}")
            raise EvalSkip("unhealthy") from e

        if self._state is None:
            self._state = make_eval_generate_state(self.args)
        output = self._inner(
            RolloutFnEvalInput(
                rollout_id=input.rollout_id,
                generate_state=self._state,
                weight_version=input.weight_version,
                hf_dir=checkpoint_dir,
            )
        )
        if inspect.iscoroutine(output):
            output = await output
        return output

    async def _wait_router_ready(self, timeout: float = 180.0) -> None:
        """After a revival the router 503s until its health cycle evicts the dead
        worker; a retried one-token probe proves the route is usable before dispatch."""
        from miles.utils.http_utils import wait_http_ok

        await wait_http_ok(
            f"http://{self._srv.router_ip}:{self._srv.router_port}/generate",
            json_payload={"input_ids": [0], "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
            timeout=timeout,
        )

    async def _mark_unreachable_engines(self) -> None:
        """Without fault tolerance nothing records an engine death (recover() only
        restarts engines already marked stopped), so the fleet probes itself."""
        for group in self._srv.server_groups:
            for engine in group.all_engines:
                if not engine.is_allocated:
                    continue
                try:
                    await asyncio.wait_for(engine.actor_handle.get_weight_version.remote(), timeout=60)
                except Exception as e:
                    logger.warning(f"Eval engine unreachable ({e!r}); marking stopped for recovery")
                    try:
                        ray.kill(engine.actor_handle)
                    except Exception:
                        pass
                    engine.mark_stopped()


def resolve_checkpoint_eval_fn(args: Namespace, eval_fn, servers) -> CheckpointEvalFn | None:
    """The single place that decides whether eval consumes snapshots, and with
    which backend. None = shared-engine eval (the fn runs on its own state)."""
    if args.eval_num_gpus > 0:
        return FleetEvalFn(args, servers["eval"], inner=eval_fn)
    if isinstance(eval_fn, CheckpointEvalFn):
        assert args.eval_hf_dir is not None or args.save_hf is not None, (
            "checkpoint eval fns need a snapshot source: set --eval-hf-dir (staging exports) "
            "or --save-hf (reuse periodic HF checkpoints)."
        )
        assert args.eval_keep_snapshots >= args.eval_max_in_flight, (
            f"--eval-keep-snapshots ({args.eval_keep_snapshots}) must be >= --eval-max-in-flight "
            f"({args.eval_max_in_flight}), otherwise a pending eval's snapshot could be GC'd."
        )
        return eval_fn
    assert not (
        inspect.isclass(eval_fn) and issubclass(eval_fn, CheckpointEvalFn)
    ), "checkpoint eval fns require the class-based rollout API (MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1)."
    return None


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
