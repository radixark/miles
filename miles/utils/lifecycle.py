"""Trajectory-lifecycle seam callable from core rollout code.

Core generation paths read ``TrajectoryLifecycle().sink`` unconditionally, so
this module must never import from the dashboard extra. The dashboard installs
the sink when enabled; while it is ``None`` every probe site is a no-op.
"""

from __future__ import annotations

from miles.utils.misc import SingletonMeta


class TrajectoryLifecycle(metaclass=SingletonMeta):
    def __init__(self):
        self.sink = None  # the dashboard TrajectorySink (duck-typed); None = off


def attach_lifecycle_metadata(sample, record, prev_record, turn: int) -> None:
    """Fold one chat record's timing onto its sample via
    ``metadata["lifecycle"]`` so it rides the Sample across the session
    boundary. ``timestamp`` is the segment end; the start is the engine's
    ``e2e_latency`` when reported."""
    meta_info = record.response["choices"][0].get("meta_info", {})
    latency = meta_info.get("e2e_latency")
    t1 = record.timestamp
    t0 = t1 - latency if latency is not None else None
    segment = dict(t0=t0, t1=t1, turn=turn)
    if record.request_timestamp is not None:
        segment["req_ts"] = record.request_timestamp  # server-edge arrival bounds the agent gap
    if prev_record is not None:
        segment["prev_t1"] = prev_record.timestamp  # gap since the previous call end = agent-side work
    sample.metadata["lifecycle"] = segment
