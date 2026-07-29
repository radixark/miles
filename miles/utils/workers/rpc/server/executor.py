from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)


class RpcCallExecutor:
    def __init__(self, *, worker: object) -> None:
        self._worker = worker
        self._background_tasks: set[asyncio.Task[None]] = set()

    def start(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]) -> None:
        task = asyncio.create_task(self._run(spec=spec, kwargs=kwargs, call_id=call_id, finish=finish))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run(
        self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]
    ) -> None:
        started_at = time.monotonic()
        log_fields = {"tag": "rpc", "op": "execute", "method": spec.name, "call": call_id}
        log_structured(logger.debug, phase="start", **log_fields)

        try:
            result = await self._call_worker(spec=spec, kwargs=kwargs)
            outcome = CallStatusResponse(status="success", result=spec.serializer.encode_result(result))
            log_structured(
                logger.debug, phase="end", ok=True, **log_fields, elapsed_s=round(time.monotonic() - started_at, 3)
            )
        except Exception as e:
            log_structured(logger.error, phase="end", ok=False, **log_fields, exc_info=True)
            outcome = CallStatusResponse(status="failed", error="".join(traceback.format_exception(e)))

        finish(outcome=outcome)

    async def _call_worker(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        return await getattr(self._worker, spec.name)(**kwargs)
