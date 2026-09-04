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
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)


class RpcCallExecutor:
    def __init__(self, *, worker: object, specs: dict[str, RpcMethodSpec]) -> None:
        self._worker = worker
        self._executors = {
            group: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rpc-{group}")
            for group in sorted({spec.concurrency_group for spec in specs.values() if not spec.is_async})
        }
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def concurrency_groups(self) -> list[str]:
        return sorted(self._executors)

    def start(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]) -> None:
        task = asyncio.create_task(self._run(spec=spec, kwargs=kwargs, call_id=call_id, finish=finish))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
        if spec.is_async:
            return await method(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executors[spec.concurrency_group], functools.partial(method, **kwargs)
        )
