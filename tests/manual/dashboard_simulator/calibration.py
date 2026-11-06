"""A finished real ``--dump-details`` run, read as the source a widened replay copies."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from miles.utils.types import Sample
from miles_dashboard.dump_reader import DumpReader
from miles_dashboard.store import MetricStore, PhaseEvent, Role, Stream, TrajectoryEvent, TrajectoryEventKind

_LIFECYCLE_TS_KEYS = ("t0", "t1", "req_ts", "prev_t1")


@dataclass(frozen=True)
class CalibrationTrajectory:
    """What the trajectory stream needs of one calibration sample, without its tokens."""

    slot: int
    status: str
    lifecycle: list[dict]
    last_weight_version: int
    attempt_start_ts: float
    attempt_end_ts: float
    # the calibration's own trajectories-stream events for this sample, when it recorded them
    recorded_events: tuple[TrajectoryEvent, ...] = ()


class CalibrationRun:
    def __init__(self, dump_dir: Path, *, attempt_start_source: Path | None):
        self.dump_dir = dump_dir
        self.store = MetricStore.load(dump_dir / "dashboard")
        self.lanes = self.store.lane_index()
        self.train_gpus = sum(1 for lane in self.lanes if Role.TRAIN.value in lane["roles"])
        self.engines = self.store.records[Stream.TOPOLOGY][-1].engines
        self.metrics = sorted(self.store.records[Stream.METRICS], key=lambda record: record.ts)
        self.data_buffer = self.store.records[Stream.DATA_BUFFER]
        # the raw rows, open markers included: replay pushes each through the sink as written
        self.phases = _drop_never_closed(self.store.iter_records(Stream.PHASES))
        # a step logs several rollout/step records; its dump lands with the first
        self.consume_ts_by_rollout_id = {}
        for record in self.metrics:
            if record.step_key == "rollout/step":
                self.consume_ts_by_rollout_id.setdefault(record.step, record.ts)
        dumped = DumpReader(dump_dir).rollout_ids()
        self.rollout_ids = [i for i in dumped.train if i in self.consume_ts_by_rollout_id]
        self.eval_rollout_ids = [i for i in dumped.eval if i in self.consume_ts_by_rollout_id]
        self.start_ts = self.store.meta.start_ts
        self.end_ts = max(self.consume_ts_by_rollout_id.values())
        # after model load and engine startup: what a longer run repeats
        self.steady_start_ts = min(
            event.t0 for event in self.phases if event.role == Role.ROLLOUT_MANAGER and event.name == "rollout"
        )
        self.trajectories_by_rollout_id = self._read_trajectories(_attempt_start_offsets(attempt_start_source))

    def telemetry_hour_files(self, stream: Stream) -> list[Path]:
        return sorted((self.dump_dir / "dashboard" / stream.value).glob("*.jsonl"))

    def load_samples(self, rollout_id: int, *, evaluation: bool = False) -> list[dict]:
        return torch.load(self._rollout_path(rollout_id, evaluation), weights_only=False)["samples"]

    def load_messages(self, rollout_id: int, *, evaluation: bool = False) -> dict[tuple[int, int], list]:
        """Sidecar conversations by (sample_index, sample_occurrence); the real
        sidecar lacks a few samples per step."""
        stem = f"eval_{rollout_id}" if evaluation else str(rollout_id)
        path = self.dump_dir / "trajectory" / f"{stem}.jsonl"
        if not path.exists():
            return {}
        messages_by_key, occurrences = {}, defaultdict(int)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                occurrence = row.get("sample_occurrence", occurrences[row["sample_index"]])
                occurrences[row["sample_index"]] += 1
                messages_by_key[(row["sample_index"], occurrence)] = row["messages"]
        return messages_by_key

    def _rollout_path(self, rollout_id: int, evaluation: bool) -> Path:
        return self.dump_dir / "rollout_data" / (f"eval_{rollout_id}.pt" if evaluation else f"{rollout_id}.pt")

    def _read_trajectories(self, attempt_start_offsets: list[float]) -> dict[int, list[CalibrationTrajectory]]:
        """From the run's own trajectories stream when it recorded one, else
        rebuilt from each sample's ``metadata["lifecycle"]``."""
        recorded_events_by_index = defaultdict(list)
        for event in self.store.trajectory_events():
            recorded_events_by_index[event.sample_index].append(event)
        rng = random.Random(0)
        trajectories_by_rollout_id = {}
        for rollout_id in self.rollout_ids:
            trajectories = []
            samples = self.load_samples(rollout_id)
            # one trajectory per session, as the sink records it: the leaves of
            # a session share a slot, and attempt_end takes all of them at once
            leaves_by_slot = defaultdict(list)
            for slot, sample in zip(sample_slots(samples), samples, strict=True):
                leaves_by_slot[slot].append(sample)
            for slot, leaves in leaves_by_slot.items():
                last_leaf = leaves[-1]
                if events := sorted(recorded_events_by_index.get(last_leaf["index"], []), key=lambda e: e.ts):
                    trajectory = CalibrationTrajectory(
                        slot=slot,
                        status=last_leaf["status"],
                        lifecycle=[],
                        last_weight_version=_last_weight_version(last_leaf),
                        attempt_start_ts=events[0].ts,
                        attempt_end_ts=events[-1].ts,
                        recorded_events=tuple(events),
                    )
                else:
                    lifecycle = [
                        segment for leaf in leaves for segment in _lifecycle_segments(leaf["metadata"]["lifecycle"])
                    ]
                    first_request_ts = min(segment.get("req_ts") or segment["t0"] for segment in lifecycle)
                    trajectory = CalibrationTrajectory(
                        slot=slot,
                        status=last_leaf["status"],
                        lifecycle=lifecycle,
                        last_weight_version=_last_weight_version(last_leaf),
                        attempt_start_ts=first_request_ts - rng.choice(attempt_start_offsets),
                        attempt_end_ts=max(segment["t1"] for segment in lifecycle),
                    )
                trajectories.append(trajectory)
            trajectories_by_rollout_id[rollout_id] = trajectories
        return trajectories_by_rollout_id


def sample_slots(samples: list[dict]) -> list[int]:
    """Each sample's slot in its step: the order its ``index`` first appears.
    The leaves of one session share an index, so they share a slot."""
    slot_by_index: dict[int, int] = {}
    return [slot_by_index.setdefault(sample["index"], len(slot_by_index)) for sample in samples]


def _last_weight_version(sample: dict) -> int:
    """Per-call versions: legacy strings, or per-call span dicts."""
    versions = sample["weight_versions"]
    if not versions:
        return -1
    last = versions[-1]
    return int(last) if isinstance(last, str) else int(last["spans"][-1]["version"])


def _attempt_start_offsets(attempt_start_source: Path | None) -> list[float]:
    """Seconds from attempt_start to the first generation (sandbox setup):
    lifecycle metadata starts at the first request, so they come from a run
    that recorded the trajectories stream."""
    if attempt_start_source is None:
        return [0.0]
    store = MetricStore.load(attempt_start_source / "dashboard")
    attempt_start_ts, first_gen_ts = {}, {}
    for event in store.trajectory_events():
        if event.kind == TrajectoryEventKind.ATTEMPT_START:
            attempt_start_ts[event.sample_index] = event.ts
        elif event.kind == TrajectoryEventKind.GEN_START and event.turn >= 0:
            first_gen_ts[event.sample_index] = min(first_gen_ts.get(event.sample_index, event.ts), event.ts)
    return sorted(
        first_gen_ts[i] - attempt_start_ts[i]
        for i in first_gen_ts
        if i in attempt_start_ts and first_gen_ts[i] >= attempt_start_ts[i]
    )


def _drop_never_closed(events: list[PhaseEvent]) -> list[PhaseEvent]:
    """Open markers the calibration never closed: its run was killed mid-phase,
    which a longer replay is not."""
    closed = {(e.node, e.rank, e.name, e.t0) for e in events if not e.open}
    return [e for e in events if not e.open or (e.node, e.rank, e.name, e.t0) in closed]


def _lifecycle_segments(lifecycle: dict | list[dict]) -> list[dict]:
    return lifecycle if isinstance(lifecycle, list) else [lifecycle]


def shift_lifecycle(lifecycle: list[dict], shift_s: float) -> list[dict]:
    return [
        {
            key: value + shift_s if key in _LIFECYCLE_TS_KEYS and value is not None else value
            for key, value in segment.items()
        }
        for segment in lifecycle
    ]


def build_sample(
    calibration_sample: dict,
    *,
    index: int,
    group_index: int,
    rollout_id: int | None,
    shift_s: float,
    weight_version_shift: int,
) -> Sample:
    """A calibration sample dict as a current-schema ``Sample``, re-indexed and time-shifted.

    The calibration dumps predate per-call version spans: each legacy per-call
    version is placed on that call's output, the loss-mask run it generated."""
    fields = dict(calibration_sample)
    if "lifecycle" in fields["metadata"]:  # only agentic sessions record it
        fields["metadata"] = dict(
            fields["metadata"],
            lifecycle=shift_lifecycle(_lifecycle_segments(fields["metadata"]["lifecycle"]), shift_s),
        )
    legacy_versions = fields["weight_versions"]
    if legacy_versions and isinstance(legacy_versions[0], dict):
        fields["weight_versions"] = [
            call
            | dict(
                spans=[_shift_span(span, weight_version_shift) for span in call["spans"]],
                prefill_spans=[_shift_span(span, weight_version_shift) for span in call["prefill_spans"]],
            )
            for call in legacy_versions
        ]
    elif legacy_versions and isinstance(legacy_versions[0], str):
        prompt_length = len(fields["tokens"]) - fields["response_length"]
        fields["weight_versions"] = [
            dict(
                spans=[dict(version=str(int(version) + weight_version_shift), abs_start=start, abs_end=end)],
                prefill_spans=[],
                output_start=start,
                prompt_tokens=start,
            )
            for version, (start, end) in zip(
                legacy_versions, _output_spans(fields["loss_mask"], prompt_length), strict=False
            )
        ]
    spec_info = fields.get("spec_info") or {}
    if "spec_accept_token_num" in spec_info:
        fields["spec_info"] = dict(
            spec_num_correct_drafts=spec_info["spec_accept_token_num"],
            spec_num_proposed_drafts=spec_info["spec_draft_token_num"],
            spec_verify_ct=spec_info["spec_verify_ct"],
            completion_tokens=spec_info["completion_token_num"],
        )
    fields.update(index=index, group_index=group_index, rollout_id=rollout_id)
    return Sample.from_dict(fields)


def _shift_span(span: dict, weight_version_shift: int) -> dict:
    return span | dict(version=str(int(span["version"]) + weight_version_shift))


def _output_spans(loss_mask: list[int], prompt_length: int) -> list[tuple[int, int]]:
    """Absolute [start, end) of each run of generated tokens."""
    spans, start = [], None
    for i, bit in enumerate(loss_mask + [0]):
        if bit and start is None:
            start = i
        elif not bit and start is not None:
            spans.append((prompt_length + start, prompt_length + i))
            start = None
    return spans
