"""Run observers: packages that watch a run from inside it, found through the ``miles.observers`` entry points.

miles reports what happens at a few fixed points, in plain data only (Python
builtins and numpy arrays, never miles objects), so an observer keeps working
across miles versions. Installing a package that registers an entry point in
``ENTRY_POINT_GROUP`` is all it takes; with none installed every report is a
no-op. An observer that raises is logged at most once per ``WARN_INTERVAL_S``
and never affects the run.

The order within one process: ``start_run`` (driver only) and
``start_process`` come first, then any events. The environment variables
``start_run`` returns are set in every worker process the driver launches.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal, TypedDict, get_args

import numpy as np

from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.misc import SingletonMeta, get_current_node_ip
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "miles.observers"


class RunStart(TypedDict):
    run_uuid: str
    start_ts: float
    args: dict[str, Any]  # every argument, JSON-safe


class ProcessStart(TypedDict):
    role: str  # "driver", "train", "rollout_manager", "inference_controller", "command"
    node: str  # the node's IP, as miles names nodes everywhere
    rank: int  # rank in its training group; -1 outside training
    gpus: list[int]  # physical GPU indices on the node held by this process


class TrajectoryEvent(TypedDict):
    ts: float
    kind: str  # attempt_start, gen_start, gen_end, tool_start, tool_end, attempt_end
    sample_index: int  # run-global sample index; -1 when the sample has none
    group_index: int  # -1 when the sample has none
    turn: int  # 1-based for gen/tool spans, -1 for attempt boundaries
    weight_version: str  # weight version of the newest span, "" when unknown
    detail: str  # completion tokens of a gen span, the cause of a tool span, the final status of an attempt


class EngineInfo(TypedDict):
    addr: str
    worker_type: str
    engine_rank: int
    gpus: list[list]  # [node IP, physical GPU index] pairs
    gpu_uuids: list[str | None]


SandboxStatus = Literal["healthy", "error", "finished"]
SANDBOX_STATUSES: tuple[str, ...] = get_args(SandboxStatus)


class SandboxReport(TypedDict):
    """One ``report_sandbox`` call from agent code: a sandbox's state when it started, ran a command, failed or ended."""

    ts: float
    sandbox_key: str
    status: SandboxStatus
    task: str | None
    provider: str | None
    sandbox_id: str | None
    error: str | None


class RolloutSample(TypedDict):
    """One generated sample of a rollout step, as the trainer will see it."""

    sample_index: int
    sample_occurrence: int  # 0, 1, ... among samples sharing ``sample_index`` (multi-leaf trajectories)
    group_index: int | None
    status: str
    remove_sample: bool
    reward: float | None  # None when the reward is not a number
    response_length: int
    tokens: list[int]  # prompt then response
    loss_mask: list[int] | None  # over the response
    rollout_log_probs: list[float] | None  # over the response
    weight_versions: list[str]  # versions of every generated span, oldest first
    turns: int | None  # generation calls
    prefill_weight_version_min: int | None
    prefill_lag: int | None  # largest lag between a call's prefill and its generation versions
    tool_calls: int | None  # tool messages in a chat prompt; None for a plain-string prompt
    non_generation_time: float
    spec_correct_drafts: int
    spec_proposed_drafts: int
    prefix_cached_tokens: int
    prefix_prompt_tokens: int
    prompt: str | list[dict]
    messages: list[dict] | None  # the raw conversation, when the generate path recorded one


class TrainTokens(TypedDict):
    """One sample's per-token training values held by one rank.

    Under context parallelism each rank holds part of the response;
    ``positions`` says which response positions the arrays cover, so readers
    need not know the split."""

    sample_index: int
    rank: int
    response_length: int
    total_length: int
    reward: float  # as trained on, after normalization
    truncated: bool | None
    loss_mask: np.ndarray  # over the whole response
    positions: np.ndarray  # response positions of the arrays below
    log_probs: np.ndarray | None
    rollout_log_probs: np.ndarray | None
    ref_log_probs: np.ndarray | None
    entropy: np.ndarray | None
    ref_entropy: np.ndarray | None
    advantages: np.ndarray | None
    returns: np.ndarray | None


class Observer:
    """What an observer can hear; every method defaults to doing nothing."""

    def enabled(self) -> bool:
        """Whether this observer takes part in this process. One that does not is dropped,
        and miles runs exactly as if its package were not installed."""
        return True

    def start_run(self, run: RunStart) -> dict[str, str]:
        """Driver only, before any worker starts. Returns environment variables for every worker."""
        return {}

    def start_process(self, process: ProcessStart) -> None: ...

    def phase_start(self, name: str, t0: float) -> None: ...

    def phase_end(self, name: str, t0: float, t1: float) -> None: ...

    def metrics(self, metrics: dict[str, Any], *, step_key: str, step: int | None) -> None: ...

    def trajectory_event(self, event: TrajectoryEvent) -> None: ...

    def engine_topology(self, engines: list[EngineInfo]) -> None: ...

    def data_buffer(self, length: int) -> None: ...

    def sandbox_report(self, report: SandboxReport) -> None: ...

    def rollout_samples(self, rollout_id: int, *, evaluation: bool, samples: list[RolloutSample]) -> None: ...

    def train_tokens(self, rollout_id: int, *, rows: list[TrainTokens]) -> None: ...

    def end_run(self, *, succeeded: bool) -> None:
        """Driver only, as the run ends either way."""


class Observers(metaclass=SingletonMeta):
    """The observers taking part in this process, each isolated from the run and from the others."""

    WARN_INTERVAL_S: ClassVar[float] = 300.0

    def __init__(self):
        self._observers = _load_observers()
        self._last_warning_of: dict[tuple[str, str], float] = {}

    @property
    def active(self) -> bool:
        return bool(self._observers)

    def hears(self, event: str) -> bool:
        """Whether any observer implements ``event``; a report that is costly to build skips it when none does."""
        default = getattr(Observer, event)  # config-access-exempt: event names an Observer method
        return any(
            getattr(type(observer), event, default) is not default for _, observer in self._observers
        )  # config-access-exempt: event names an Observer method

    def start_run(self, run: RunStart) -> dict[str, str]:
        env: dict[str, str] = {}
        for result in self._each("start_run", run):
            env.update(result or {})
        return env

    def start_process(self, process: ProcessStart) -> None:
        self._each("start_process", process)

    def phase_start(self, name: str, t0: float) -> None:
        self._each("phase_start", name, t0)

    def phase_end(self, name: str, t0: float, t1: float) -> None:
        self._each("phase_end", name, t0, t1)

    def metrics(self, metrics: dict[str, Any], *, step_key: str, step: int | None) -> None:
        self._each("metrics", metrics, step_key=step_key, step=step)

    def trajectory_event(self, event: TrajectoryEvent) -> None:
        self._each("trajectory_event", event)

    def engine_topology(self, engines: list[EngineInfo]) -> None:
        self._each("engine_topology", engines)

    def data_buffer(self, length: int) -> None:
        self._each("data_buffer", length)

    def sandbox_report(self, report: SandboxReport) -> None:
        self._each("sandbox_report", report)

    def rollout_samples(self, rollout_id: int, *, evaluation: bool, samples: list[RolloutSample]) -> None:
        self._each("rollout_samples", rollout_id, evaluation=evaluation, samples=samples)

    def train_tokens(self, rollout_id: int, *, rows: list[TrainTokens]) -> None:
        self._each("train_tokens", rollout_id, rows=rows)

    def end_run(self, *, succeeded: bool) -> None:
        self._each("end_run", succeeded=succeeded)

    @contextmanager
    def isolated(self, what: str) -> Iterator[None]:
        """Wrap miles code that only feeds observers: its failure is logged like an observer's and never reaches the run."""
        try:
            yield
        except Exception:
            self._warn("miles", what)

    def _each(self, method: str, *args, **kwargs) -> list:
        results = []
        for name, observer in self._observers:
            try:
                results.append(
                    getattr(observer, method)(*args, **kwargs)
                )  # config-access-exempt: attribute selected at runtime from method
            except Exception:
                self._warn(name, method)
        return results

    def _warn(self, name: str, method: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_of.get((name, method), -self.WARN_INTERVAL_S) >= self.WARN_INTERVAL_S:
            self._last_warning_of[(name, method)] = now
            logger.warning(
                f"observer {name!r} failed in {method}; the run continues "
                f"(further failures suppressed for {self.WARN_INTERVAL_S:.0f}s)",
                exc_info=True,
            )


def _load_observers() -> list[tuple[str, Observer]]:
    observers = []
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            observer = entry_point.load()()
            if observer.enabled():
                observers.append((entry_point.name, observer))
        except Exception:
            logger.exception(f"observer {entry_point.name!r} failed to load; the run continues without it")
    if observers:
        logger.info(f"run observers: {[name for name, _ in observers]}")
    return observers


def start_run(*, run_uuid: str, args) -> dict[str, str]:
    """Report the run's start from the driver; returns the environment its workers need."""
    observers = Observers()
    if not observers.active:
        return {}
    with observers.isolated("start_run"):
        return observers.start_run(RunStart(run_uuid=run_uuid, start_ts=time.time(), args=_plain_args(args)))
    return {}


def start_process(role: str, *, rank: int = -1, gpus: Callable[[], Iterable[int]] = tuple) -> None:
    """Report this process's start and route its timer phases (and, in the rollout
    manager, sample lifecycles) to the observers. ``gpus`` lists the GPUs the
    process holds; it is called only when an observer is installed."""
    observers = Observers()
    if not observers.active:
        return
    with observers.isolated("start_process"):
        process = ProcessStart(role=role, node=get_current_node_ip(), rank=rank, gpus=[int(gpu) for gpu in gpus()])
        observers.start_process(process)
    Timer().event_sinks.append(_PhaseReporter(observers))
    if role == "rollout_manager" and observers.hears("trajectory_event"):
        TrajectoryLifecycle().sink = _TrajectoryReporter(observers)


def _plain_args(args) -> dict[str, Any]:
    fields = {name: value for name, value in vars(args).items() if not name.startswith("_")}
    return json.loads(json.dumps(fields, default=str))


class _PhaseReporter:
    """Timer sink: phase starts and ends of this process."""

    def __init__(self, observers: Observers):
        self._observers = observers

    def begin(self, name: str, t0: float) -> None:
        self._observers.phase_start(name, t0)

    def __call__(self, name: str, t0: float, t1: float) -> None:
        self._observers.phase_end(name, t0, t1)


class _TrajectoryReporter:
    """TrajectoryLifecycle sink: turns the Samples that rollout code passes into plain lifecycle events."""

    def __init__(self, observers: Observers):
        self._observers = observers

    def attempt_start(self, sample) -> None:
        self._emit("attempt_start", sample)

    def gen_start(self, sample) -> None:
        self._emit("gen_start", sample)

    def attempt_end(self, result) -> None:
        """``result`` is the Sample or the list of per-turn Samples a generate
        function returned; per-turn samples share one identity and each carry
        their own ``metadata["lifecycle"]`` segment (agentic sessions)."""
        with self._observers.isolated("trajectory_event"):
            self._attempt_end(result if isinstance(result, list) else [result])

    def _attempt_end(self, samples: list) -> None:
        for sample in samples:
            value = (sample.metadata or {}).get("lifecycle")
            # one dict per turn-sample; a merged sample carries the list
            for segment in value if isinstance(value, list) else [value] if value else []:
                gap_end = segment.get("req_ts") or segment.get("t0")
                if segment.get("prev_t1") is not None and gap_end is not None:
                    # time between chat calls is agent-side work (tools, environment steps)
                    self.tool_span(sample, segment["prev_t1"], gap_end, turn=segment["turn"], detail="agent gap")
                if segment.get("t0") is not None:
                    self._emit("gen_start", sample, ts=segment["t0"], turn=segment["turn"])
                self._emit("gen_end", sample, ts=segment["t1"], turn=segment["turn"])
        last = samples[-1]
        self._emit("attempt_end", last, detail=last.status.value if last.status is not None else "")

    def gen_span(self, sample, t0: float, t1: float, turn: int, detail: str = "") -> None:
        self._emit("gen_start", sample, ts=t0, turn=turn)
        self._emit("gen_end", sample, ts=t1, turn=turn, detail=detail)

    def tool_span(self, sample, t0: float, t1: float, turn: int, detail: str = "") -> None:
        self._emit("tool_start", sample, ts=t0, turn=turn, detail=detail)
        self._emit("tool_end", sample, ts=t1, turn=turn, detail=detail)

    def _emit(self, kind: str, sample, *, ts: float | None = None, turn: int = -1, detail: str = "") -> None:
        with self._observers.isolated("trajectory_event"):
            self._observers.trajectory_event(self._event(kind, sample, ts=ts, turn=turn, detail=detail))

    @staticmethod
    def _event(kind: str, sample, *, ts: float | None, turn: int, detail: str) -> TrajectoryEvent:
        versions = [span.version for span in sample.all_weight_version_spans]
        return TrajectoryEvent(
            ts=time.time() if ts is None else ts,
            kind=kind,
            sample_index=sample.index if sample.index is not None else -1,
            group_index=sample.group_index if sample.group_index is not None else -1,
            turn=turn,
            weight_version=str(versions[-1]) if versions else "",
            detail=detail,
        )
