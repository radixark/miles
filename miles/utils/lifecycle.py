"""Trajectory-lifecycle seam callable from core rollout code.

A neutral observation point at the same layer as ``Timer.event_sinks``: core
generation paths (``sglang_rollout``, ``generate_hub``) read
``TrajectoryLifecycle().sink`` unconditionally, so this module must never
import anything from the dashboard extra. The sink object itself is constructed and installed by
``miles.dashboard.hooks`` when the dashboard is enabled; while it is ``None``
every probe site is a guarded no-op.
"""

from __future__ import annotations

from miles.utils.misc import SingletonMeta


class TrajectoryLifecycle(metaclass=SingletonMeta):
    def __init__(self):
        # the dashboard TrajectorySink, duck-typed; None = dashboard off
        self.sink = None


def attach_lifecycle_metadata(sample, record, prev_record, turn: int) -> None:
    """Fold one session-server chat record's timing onto the sample it became
    (design §18.3 hook 3): ``SessionRecord.timestamp`` is stamped after the
    backend returns (= segment END); the start comes from the engine-reported
    ``e2e_latency`` when present. Written to ``sample.metadata["lifecycle"]``
    (the ``tito_session_mismatch`` precedent) so it rides the Sample across
    the session boundary — records themselves never need to leave it.
    """
    meta_info = record.response["choices"][0].get("meta_info", {})
    latency = meta_info.get("e2e_latency")
    t1 = record.timestamp
    t0 = t1 - latency if latency is not None else None
    segment = dict(t0=t0, t1=t1, turn=turn)
    if prev_record is not None:
        # the gap between the previous chat call's end and this one's start is
        # agent-side work (tool execution, environment steps)
        segment["prev_t1"] = prev_record.timestamp
    sample.metadata["lifecycle"] = segment
