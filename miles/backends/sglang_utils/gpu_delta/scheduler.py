"""SGLang scheduler adapter for the GPU-delta session: receiver admission, ordered controls and replies.

Every scheduler registers the controls; only a describe loads the receiver's dependencies, so ordinary
disk users never import them.
"""

from __future__ import annotations

import os
import socket
import uuid
from functools import partial

from sglang.srt.managers.io_struct import ContinueGenerationReqInput, PauseGenerationReqInput
from torch.distributed import gather_object

from miles.backends.sglang_utils.gpu_delta import io as delta_io
from miles.backends.sglang_utils.gpu_delta.session import DeltaSession

# Every receipt state, classified explicitly: an unknown state is an error, never a success.
_FAILED_STATES = frozenset({"FAILED", "POISONED"})
_SUCCEEDED_STATES = frozenset(
    {"IDLE", "CLEARED", "PREPARING", "PREPARED", "ABORTED", "APPLYING", "APPLIED", "RESUMING", "RESUMED"}
)


def _succeeded(state: str) -> bool:
    if state in _FAILED_STATES:
        return False
    elif state in _SUCCEEDED_STATES:
        return True
    else:
        raise ValueError(f"unknown delta receipt state: {state}")


def _dspark_lm_head_dtype(spec):
    """SGLang main's ``--speculative-dspark-lm-head-dtype``, read only where the spec namespace declares it.

    Main quantizes the target lm_head once into a cache that the DSPARK draft projects through
    (``models/dspark.py``), which a weight update would leave stale. The companion SGLang tree declares
    no such field; its only quantized DSPARK head is ``SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD``, gated to
    ROCm gfx95, which a CUDA receiver never serves. There this returns None and describe is unchanged.
    """
    from sglang.srt.arg_groups.fields.spec import Spec

    if "speculative_dspark_lm_head_dtype" in Spec.__struct_fields__:
        return spec.speculative_dspark_lm_head_dtype
    else:
        return None


def _admit_receiver(parallel, moe, spec, scheduler) -> None:
    """Reject a launch whose topology, features or weight caches an in-place delta cannot serve."""
    # DWDP serves routed experts from per-step copies of peer ranks' weights. Its launch also forces the
    # local control broadcast rejected next, so reject it first under its own name.
    if parallel.dwdp_size > 1:
        raise ValueError("GPU delta does not support DWDP expert weight prefetch")
    if parallel.enable_dp_attention_local_control_broadcast or moe.is_ep_scale_joiner:
        raise ValueError("GPU delta requires the global control broadcast without EP scale joiners")
    # PP relays controls only after this handler returns and retains several
    # in-flight microbatches; it needs a separate pause/drain protocol.
    if parallel.pp_size != 1:
        raise ValueError("GPU delta currently requires PP1")
    # Every rank replies to prepare on the tokenizer's reply socket. Only attention DP serves it over TCP;
    # otherwise it is an IPC path on the head node, which other nodes cannot reach.
    if parallel.nnodes > 1 and not parallel.attn_dp_enabled:
        raise ValueError("GPU delta requires attention DP when an engine spans several nodes")
    if parallel.attn_cp_size > 1 and parallel.enable_cp_decode_attn_tp:
        raise ValueError("GPU delta does not support CP decode attention TP caches")
    if parallel.attn_tp_size % parallel.attn_dcp_size:
        raise ValueError("GPU delta requires DCP groups inside attention TP groups")
    if scheduler.enable_lora or str(scheduler.disaggregation_mode.value) != "null":
        raise ValueError("GPU delta currently requires no LoRA and no disaggregation")
    if scheduler.rust_server is not None:
        raise ValueError("GPU delta endpoints currently require the Python server")
    if _dspark_lm_head_dtype(spec) is not None:
        raise ValueError(
            "GPU delta cannot refresh DSPARK's quantized lm_head cache; unset --speculative-dspark-lm-head-dtype"
        )
    # With a token map, the EAGLE worker gives the draft a gathered copy of the target lm_head's hot rows.
    if spec.speculative_token_map is not None:
        raise ValueError("GPU delta cannot refresh the draft's hot-token lm_head copy; unset --speculative-token-map")
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    runner = scheduler.tp_worker.model_runner
    # Reuse the ordinary updater's exclusions: shared IPC weights may belong
    # to other engines, and untracked derived caches would retain old values.
    runner.weight_updater._assert_weight_cache_inactive("apply_gpu_delta")
    error = _unsupported_derived_weight_cache_error(runner.model)
    if error is not None:
        raise ValueError(error)


def with_gpu_delta_controls(scheduler, dispatcher):
    """Register only delta requests; ordinary handlers remain unchanged."""
    from sglang.srt.runtime_context import get_parallel
    from sglang.utils import TypeBasedDispatcher

    parallel = get_parallel()
    reply_groups = []
    if parallel.attn_cp_size > 1:
        reply_groups.append(parallel.attn_cp_group)
    if parallel.attn_tp_size > 1:
        reply_groups.append(parallel.attn_tp_group)
    control = GpuDeltaSchedulerControl(scheduler, tuple(reply_groups))
    dispatcher += TypeBasedDispatcher(
        [
            (delta_io.GetGpuDeltaInfoReqInput, control.handle),
            (delta_io.ClearGpuDeltaStateReqInput, control.handle),
            (delta_io.ReleaseGpuDeltaCacheReqInput, control.handle),
            (delta_io.PrepareGpuDeltaReqInput, control.handle),
            (delta_io.ApplyGpuDeltaReqInput, control.handle),
            (delta_io.AbortGpuDeltaReqInput, control.handle),
            (delta_io.ResumeGpuDeltaReqInput, control.handle),
        ]
    )
    return dispatcher


class GpuDeltaSchedulerControl:
    """One scheduler rank's adapter: it owns the rank's delta session and answers its controls."""

    def __init__(self, scheduler, reply_groups=()):
        self.scheduler = scheduler
        self.reply_groups = reply_groups
        self.session: DeltaSession | None = None

    def _prepare_completed(self, request, future):
        try:
            receipt = future.result()
        except Exception as exc:
            self._reply_prepare(request, self.session.status(), False, f"{type(exc).__name__}: {exc}")
        else:
            self._reply_prepare(request, receipt, receipt["state"] == "PREPARED")

    def _reply_prepare(self, request, receipt, success, message=""):
        # Each rank replies from its preparation worker without entering any
        # inference collective or sharing the scheduler's thread-owned socket.
        import zmq
        from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
        from sglang.srt.utils.network import get_zmq_socket

        receipt["generation_paused"] = self.scheduler._engine_paused
        with get_zmq_socket(zmq.Context.instance(), zmq.PUSH, request.http_worker_ipc, False) as socket:
            SenderWrapper(socket).send_output(
                delta_io.GpuDeltaReqOutput(
                    rid=request.rid,
                    success=success,
                    message=message or receipt.get("message", ""),
                    participants=[receipt],
                ),
                request,
            )

    def _describe(self, engine_id: str) -> dict:
        from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

        from miles.backends.sglang_utils.gpu_delta.bindings import CanonicalShadows
        from miles.backends.sglang_utils.gpu_delta.receiver import GpuDeltaBackend
        from miles.utils.gpu_delta.host import host_cache_id
        from miles.utils.gpu_delta.host_worker import process_start_ticks

        parallel = get_parallel()
        _admit_receiver(parallel, get_exec().moe, get_spec(), self.scheduler)
        runner = self.scheduler.tp_worker.model_runner

        def make_backend(identity, shadow_store, create):
            return GpuDeltaBackend(runner, identity, CanonicalShadows(shadow_store, create=create))

        if self.session is None:
            identity = {
                "engine_id": engine_id,
                "rank_id": uuid.uuid4().hex,
                "hostname": socket.gethostname(),
                "host_cache_id": host_cache_id(engine_id),
                "pid": os.getpid(),
                "start_ticks": process_start_ticks(),
                "tp_rank": parallel.tp_rank,
                "dp_rank": 0 if parallel.dp_rank is None else parallel.dp_rank,
                "pp_rank": parallel.pp_rank,
            }
            # A fresh receiver serves the immutable checkpoint, so it may create shadows from it.
            shadow_store = {}
            self.session = DeltaSession(identity, make_backend(identity, shadow_store, True), shadow_store)
        elif self.session.identity["engine_id"] != engine_id:
            raise ValueError("receiver incarnation belongs to another engine")
        elif self.session.backend is None:
            self.session.reattach(make_backend)
        stream_id, version = self.session.committed
        return {
            "identity": self.session.identity,
            "state": self.session.status()["state"],
            "stream_id": stream_id,
            "version": version,
            "plan_digest": self.session.plan_digest,
        }

    def _pause_and_fence(self):
        # DeltaSession.apply calls this only for its PREPARED transaction. The pause flag
        # precedes the reader fence, generation pause and flush; SGLang's flush_cache relies on it.
        self.scheduler._engine_paused = True
        self.scheduler.device_module.synchronize()

    def _pause_generation(self, pause_mode):
        """SGLang's own generation pause in the requested mode, run after the reader fence."""
        if pause_mode == "retract":
            # Drains the overlap result, requeues running requests and frees their KV: a flush can follow.
            request = PauseGenerationReqInput(mode="retract")
        elif pause_mode == "in_place":
            # Running batches, their KV and the overlap result stay; the normal loop resumes them.
            request = PauseGenerationReqInput(mode="in_place")
        else:
            raise ValueError(f"unknown GPU delta pause mode: {pause_mode}")
        return partial(self.scheduler.pause_generation, request)

    def _resume_generation(self, request, version):
        # Serving model identity can differ from the delta stream sequence.
        self.scheduler.record_weight_version_change(
            str(version) if request.weight_version is None else request.weight_version
        )
        if not request.keep_pause:
            self.scheduler.continue_generation(ContinueGenerationReqInput(torch_empty_cache=False))

    def _control(self, request) -> dict:
        """Run one synchronous control on the scheduler thread and return this rank's receipt."""
        if isinstance(request, delta_io.GetGpuDeltaInfoReqInput):
            return self._describe(request.engine_id)
        elif isinstance(request, delta_io.ClearGpuDeltaStateReqInput):
            return self.session.clear() if self.session is not None else {"identity": None, "state": "CLEARED"}
        elif isinstance(request, delta_io.ReleaseGpuDeltaCacheReqInput):
            return self.session.release_cache() if self.session is not None else {"identity": None, "state": "CLEARED"}
        elif isinstance(request, delta_io.ApplyGpuDeltaReqInput):
            return self.session.apply(
                request.session_id,
                self._pause_and_fence,
                self._pause_generation(request.pause_mode),
                (lambda: self.scheduler.flush_cache(empty_cache=False)) if request.pause_mode == "retract" else None,
                pause_mode=request.pause_mode,
            )
        elif isinstance(request, delta_io.ResumeGpuDeltaReqInput):
            resume = partial(self._resume_generation, request)
            return self.session.resume(request.session_id, resume, keep_pause=request.keep_pause)
        elif isinstance(request, delta_io.AbortGpuDeltaReqInput):
            return self.session.abort(request.session_id)
        else:
            raise ValueError("unknown delta operation")

    def _gather_replies(self, request, replies):
        for group in self.reply_groups:
            gathered = [None] * group.world_size if group.rank_in_group == 0 else None
            # CP leaders gather along TP. Carry every accumulated receipt,
            # including failures, to the sole CP0/TP0 tokenizer socket per DP
            # route before advancing prepare/apply/resume to the next phase.
            gather_object(replies, gathered, dst=group.ranks[0], group=group.cpu_group)
            if gathered is None:
                return None
            replies = [reply for participants in gathered for reply in participants]
        return delta_io.GpuDeltaReqOutput(
            rid=request.rid,
            success=all(reply[0] for reply in replies),
            message=" | ".join(reply[1] for reply in replies if reply[1]),
            participants=[reply[2] for reply in replies],
        )

    def handle(self, request):
        try:
            if isinstance(request, delta_io.PrepareGpuDeltaReqInput):
                completion = self.session.prepare(request.publication())
                # The preparation worker replies once this transaction settles.
                completion.add_done_callback(lambda future: self._prepare_completed(request, future))
                return None
            receipt = self._control(request)
            success, message = _succeeded(receipt["state"]), receipt.get("message", "")
        except Exception as exc:
            receipt = self.session.status() if self.session is not None else {"identity": None, "state": "REJECTED"}
            if receipt["state"] == "POISONED":
                self.scheduler._engine_paused = True
            success, message = False, f"{type(exc).__name__}: {exc}"
        receipt["generation_paused"] = self.scheduler._engine_paused
        if isinstance(request, delta_io.PrepareGpuDeltaReqInput):
            self._reply_prepare(request, receipt, success, message)
            return None
        return self._gather_replies(request, [(success, message, receipt)])
