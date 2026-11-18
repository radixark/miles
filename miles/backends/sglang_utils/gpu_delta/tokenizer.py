"""Control plane for engines exclusively owned by the GPU-delta coordinator."""

import asyncio
import time
import uuid
from collections.abc import Callable

import zmq

from sglang.srt.runtime_context import get_serving
from miles.backends.sglang_utils.gpu_delta.io import (
    AbortGpuDeltaReqInput,
    ApplyGpuDeltaReqInput,
    ClearGpuDeltaStateReqInput,
    GetGpuDeltaInfoReqInput,
    GpuDeltaReqOutput,
    PrepareGpuDeltaReqInput,
    ReleaseGpuDeltaCacheReqInput,
    ResumeGpuDeltaReqInput,
    UpdateWeightsFromGpuDeltaReqInput,
)

# Bound on the all-rank preparation; a preparation still unsettled is aborted before any weight changes.
_PREPARE_TIMEOUT_S = 1800


class GpuDeltaCommunicator:
    """Serialize delta controls and correlate replies without blocking generation."""

    def __init__(self, send: Callable, fan_out: int):
        self._send = send
        self._fan_out = fan_out
        self._lock = asyncio.Lock()
        self._rid = self._event = self._results = self._expected = None
        self._pending_ranks = None
        self._prepare_ranks = None

    async def __call__(self, request):
        async with self._lock:
            request.rid = self._rid = uuid.uuid4().hex
            self._event = asyncio.Event()
            self._results = []
            self._expected = self._fan_out
            self._pending_ranks = (
                {rank["rank_id"] for rank in request.participants}
                if isinstance(request, PrepareGpuDeltaReqInput)
                else None
            )
            self._prepare_ranks = self._pending_ranks.copy() if self._pending_ranks is not None else None
            try:
                self._send(request)
                await self._event.wait()
                return self._results
            finally:
                self._rid = self._event = self._results = self._expected = None
                self._pending_ranks = None
                self._prepare_ranks = None

    def handle_recv(self, reply):
        # Canceled requests can still reply after the next control was sent.
        if self._event is None or reply.rid != self._rid:
            return
        if self._pending_ranks is not None:
            if not reply.success:
                self._results.append(reply)
                self._event.set()
                return
            ranks = {row["identity"]["rank_id"] for row in reply.participants}
            if not ranks or not ranks.issubset(self._prepare_ranks):
                reply.success = False
                reply.message = "delta prepare replied from another receiver incarnation"
                self._results.append(reply)
                self._event.set()
                return
            if not ranks.intersection(self._pending_ranks):
                return
            self._pending_ranks.difference_update(ranks)
        self._results.append(reply)
        complete = not self._pending_ranks if self._pending_ranks is not None else len(self._results) == self._expected
        if complete:
            self._event.set()

    def set_fan_out(self, fan_out: int):
        self._fan_out = fan_out


class GpuDeltaTokenizerControl:
    def __init__(self, manager, fan_out: int):
        from sglang.utils import TypeBasedDispatcher

        self.manager = manager
        self._control_lock = asyncio.Lock()
        self._last_activation = None
        self.reply_endpoint = manager.recv_from_detokenizer.getsockopt_string(zmq.LAST_ENDPOINT)
        self.communicator = GpuDeltaCommunicator(manager._dispatch_to_scheduler, fan_out)
        manager._result_dispatcher += TypeBasedDispatcher([(GpuDeltaReqOutput, self.communicator.handle_recv)])

    async def request(self, obj):
        try:
            self.manager.auto_create_handle_loop()
            if get_serving().tokenizer_worker_num != 1:
                return {
                    "success": False,
                    "message": "GPU delta requires one Python tokenizer worker",
                    "participants": [],
                }
            # Serialize whole transactions, including info reads that settle a lost HTTP reply.
            # This control-plane lock is independent of generation.
            async with self._control_lock:
                if isinstance(obj, UpdateWeightsFromGpuDeltaReqInput):
                    result = await self._update(obj)
                    if result["success"]:
                        self._last_activation = obj.session_id, result
                    return result
                elif isinstance(obj, ClearGpuDeltaStateReqInput):
                    return await self._clear()
                elif isinstance(obj, GetGpuDeltaInfoReqInput):
                    result = await self._request(obj)
                    if self._last_activation is not None and self._last_activation[0] == obj.session_id:
                        result["activation"] = self._last_activation[1]
                    return result
                else:
                    raise ValueError(f"unknown GPU delta request: {type(obj).__name__}")
        except Exception as exc:
            return {"success": False, "message": f"{type(exc).__name__}: {exc}", "participants": []}

    async def _resume(self, obj):
        async with self.manager.is_pause_cond:
            result = await self._request(obj)
            if result["success"]:
                self.manager._update_weight_version_if_provided(
                    str(result["participants"][0]["version"]) if obj.weight_version is None else obj.weight_version
                )
                if not obj.keep_pause:
                    self.manager.is_pause = False
                    self.manager.is_pause_cond.notify_all()
            return result

    async def _apply(self, obj):
        async with self.manager.is_pause_cond:
            was_paused = self.manager.is_pause
            self.manager.is_pause = True
            if obj.abort_all_requests:
                self.manager.abort_request(abort_all=True)
            result = await self._request(ApplyGpuDeltaReqInput(session_id=obj.session_id, pause_mode=obj.pause_mode))
            if result["success"] and obj.pause_mode == "retract" and self.manager.mm_processor is not None:
                self.manager.mm_processor.clear_preprocess_cache()
            return result, was_paused

    async def _update(self, obj):
        started = time.monotonic()
        prepare = PrepareGpuDeltaReqInput(http_worker_ipc=self.reply_endpoint, **obj.publication())
        # Preparation can fail safely while the engine still serves base weights.
        # After apply dispatch, neither cancellation nor failure may replay XOR
        # or reopen admission: a participant may already have modified weights.
        try:
            result = await asyncio.wait_for(self._request(prepare), timeout=_PREPARE_TIMEOUT_S)
        except BaseException:
            await self._request(AbortGpuDeltaReqInput(session_id=obj.session_id))
            raise
        if not result["success"]:
            await self._request(AbortGpuDeltaReqInput(session_id=obj.session_id))
            return result
        prepared_at = time.monotonic()
        applied, was_paused = await self._apply(obj)
        if not applied["success"]:
            return applied
        applied_at = time.monotonic()
        result = await self._resume(
            ResumeGpuDeltaReqInput(
                session_id=obj.session_id,
                keep_pause=was_paused,
                weight_version=obj.weight_version,
            )
        )
        if not result["success"]:
            return result
        resumed_at = time.monotonic()
        result["applied_participants"] = applied["participants"]
        result["timings"] = {
            "prepare_s": prepared_at - started,
            "apply_s": applied_at - prepared_at,
            "resume_s": resumed_at - applied_at,
            "activation_s": resumed_at - started,
        }
        if obj.release_state:
            cleared = await self._clear()
            if not cleared["success"]:
                return cleared
        return result

    async def _clear(self):
        result = await self._request(ClearGpuDeltaStateReqInput())
        if not result["success"]:
            return result
        # Each engine-host's arena owner removes its shared cache once every rank has closed.
        return await self._request(ReleaseGpuDeltaCacheReqInput())

    async def _request(self, obj):
        results = await self.communicator(obj)
        participants = [rank for result in results for rank in result.participants]
        success = all(result.success for result in results)
        return {
            "success": success,
            "message": " | ".join(result.message for result in results if result.message),
            "participants": participants,
        }
