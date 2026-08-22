from __future__ import annotations

import asyncio
import functools
import json
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


class RpcOutcomeTooLargeError(Exception):
    pass


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
            self._validate_outcome_size(spec=spec, outcome=outcome)
            log_structured(
                logger.debug, phase="end", ok=True, **log_fields, elapsed_s=round(time.monotonic() - started_at, 3)
            )
        except asyncio.CancelledError as e:
            log_structured(logger.warning, phase="end", ok=False, cancelled=True, **log_fields)
            finish(outcome=self._failed_outcome(spec=spec, error=e))
            raise
        except Exception as e:
            log_structured(logger.error, phase="end", ok=False, **log_fields, exc_info=True)
            outcome = self._failed_outcome(spec=spec, error=e)

        finish(outcome=outcome)

    async def _call_worker(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        method = getattr(self._worker, spec.name)
        if spec.is_async:
            return await method(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executors[spec.concurrency_group], functools.partial(method, **kwargs)
        )

    def _validate_outcome_size(self, *, spec: RpcMethodSpec, outcome: CallStatusResponse) -> None:
        if spec.max_serialized_outcome_bytes is None:
            return
        if (
            _measure_json_bytes(
                outcome.model_dump(mode="json"),
                limit=spec.max_serialized_outcome_bytes,
            )
            is None
        ):
            raise RpcOutcomeTooLargeError(
                f"{spec.name} serialized outcome is above its "
                f"{spec.max_serialized_outcome_bytes}-byte RPC result limit"
            )

    def _failed_outcome(self, *, spec: RpcMethodSpec, error: BaseException) -> CallStatusResponse:
        limit = spec.max_serialized_outcome_bytes
        if limit is None:
            return CallStatusResponse(status="failed", error="".join(traceback.format_exception(error)))
        if _measure_json_bytes(str(error), limit=limit // 2) is None:
            return self._compact_failure(limit=limit, error=error)
        outcome = CallStatusResponse(status="failed", error="".join(traceback.format_exception(error)))
        if _measure_json_bytes(outcome.model_dump(mode="json"), limit=limit) is not None:
            return outcome

        return self._compact_failure(limit=limit, error=error)

    def _compact_failure(self, *, limit: int, error: BaseException) -> CallStatusResponse:
        compact = CallStatusResponse(
            status="failed",
            error=f"{type(error).__name__}: remote exception exceeded the {limit}-byte RPC error limit",
        )
        assert len(compact.model_dump_json().encode()) <= limit
        return compact


def _measure_json_bytes(value: Any, *, limit: int) -> int | None:
    measured = _measure_json_value(value=value, remaining=limit)
    return None if measured is None else limit - measured


def _measure_json_value(*, value: Any, remaining: int) -> int | None:
    if remaining < 0:
        return None
    if value is None:
        return _subtract_json_bytes(remaining=remaining, size=4)
    if value is True:
        return _subtract_json_bytes(remaining=remaining, size=4)
    if value is False:
        return _subtract_json_bytes(remaining=remaining, size=5)
    if isinstance(value, str):
        return _measure_json_string(value=value, remaining=remaining)
    if isinstance(value, (int, float)):
        return _subtract_json_bytes(remaining=remaining, size=len(json.dumps(value, allow_nan=False)))
    if isinstance(value, list):
        remaining = _subtract_json_bytes(remaining=remaining, size=2)
        if remaining is None:
            return None
        for index, item in enumerate(value):
            if index:
                remaining = _subtract_json_bytes(remaining=remaining, size=1)
                if remaining is None:
                    return None
            remaining = _measure_json_value(value=item, remaining=remaining)
            if remaining is None:
                return None
        return remaining
    if isinstance(value, dict):
        remaining = _subtract_json_bytes(remaining=remaining, size=2)
        if remaining is None:
            return None
        for index, (key, item) in enumerate(value.items()):
            if index:
                remaining = _subtract_json_bytes(remaining=remaining, size=1)
                if remaining is None:
                    return None
            remaining = _measure_json_string(value=key, remaining=remaining)
            if remaining is None:
                return None
            remaining = _subtract_json_bytes(remaining=remaining, size=1)
            if remaining is None:
                return None
            remaining = _measure_json_value(value=item, remaining=remaining)
            if remaining is None:
                return None
        return remaining
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _measure_json_string(*, value: str, remaining: int) -> int | None:
    remaining = _subtract_json_bytes(remaining=remaining, size=2)
    if remaining is None:
        return None
    if value.isascii() and all(character >= " " and character not in {'"', "\\"} for character in value):
        return _subtract_json_bytes(remaining=remaining, size=len(value))

    for character in value:
        if character in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
            size = 2
        elif ord(character) < 0x20:
            size = 6
        else:
            size = len(character.encode())
        remaining = _subtract_json_bytes(remaining=remaining, size=size)
        if remaining is None:
            return None
    return remaining


def _subtract_json_bytes(*, remaining: int, size: int) -> int | None:
    remaining -= size
    return None if remaining < 0 else remaining
