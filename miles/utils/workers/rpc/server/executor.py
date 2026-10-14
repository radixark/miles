from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import DEFAULT_CONCURRENCY_GROUP, DedicatedRpcLoopWorker, RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server.worker_loop import _DedicatedWorkerLoop

logger = logging.getLogger(__name__)


class RpcCallExecutor:
    def __init__(self, *, worker: object, specs: dict[str, RpcMethodSpec]) -> None:
        self._worker = worker
        use_dedicated_loop = isinstance(worker, DedicatedRpcLoopWorker)
        if use_dedicated_loop:
            assert all(
                spec.concurrency_group == DEFAULT_CONCURRENCY_GROUP for spec in specs.values()
            ), "Dedicated-loop workers do not support thread concurrency groups"
        self._worker_loop = _DedicatedWorkerLoop() if use_dedicated_loop else None
        self._closed = False
        self._executors = {
            group: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rpc-{group}")
            for group in sorted(
                {spec.concurrency_group for spec in specs.values() if not spec.is_async and not use_dedicated_loop}
            )
        }
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def concurrency_groups(self) -> list[str]:
        return sorted(self._executors)

    def start(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]) -> None:
        if self._closed:
            raise RuntimeError("The RPC executor is closed")
        task = asyncio.create_task(self._run(spec=spec, kwargs=kwargs, call_id=call_id, finish=finish))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._worker_loop is not None:
            await self._worker_loop.close()
        for executor in self._executors.values():
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    async def _run(
        self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]
    ) -> None:
        started_at = time.monotonic()
        log_fields = {"tag": "rpc", "op": "execute", "method": spec.name, "call": call_id}
        log_structured(logger.debug, phase="start", **log_fields, group=spec.concurrency_group)

        try:
            result = await self._call_worker(spec=spec, kwargs=kwargs)
            outcome = CallStatusResponse(status="success", result=spec.serializer.encode_result(result))
            log_structured(
                logger.debug, phase="end", ok=True, **log_fields, elapsed_s=round(time.monotonic() - started_at, 3)
            )
        except asyncio.CancelledError as e:
            log_structured(logger.warning, phase="end", ok=False, cancelled=True, **log_fields)
            finish(outcome=CallStatusResponse(status="failed", error=repr(e)))
            raise
        except Exception as e:
            log_structured(logger.error, phase="end", ok=False, **log_fields, exc_info=True)
            outcome = CallStatusResponse(status="failed", error="".join(traceback.format_exception(e)))

        finish(outcome=outcome)

    async def _call_worker(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        method = getattr(self._worker, spec.name)
        if self._worker_loop is not None:
            return await self._worker_loop.call(method=method, kwargs=kwargs, is_async=spec.is_async)
        if spec.is_async:
            return await method(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executors[spec.concurrency_group], functools.partial(method, **kwargs)
        )
