"""Replay a calibration run widened to N GPUs through the dashboard's real write path.

Every telemetry row of the calibration run is mapped onto ``replicas`` copies
of its lane / engine / train rank and pushed through the real ``PhaseSink`` /
``TrajectorySink`` batching into ``DashboardCollector``, which flushes hourly
partitions; a virtual clock stands in for wall time. Each rollout step carries
``replicas`` times the samples: replica j of step k replays calibration step
k+j, shifted onto step k's consume time. Runs longer than the calibration
repeat it.
"""

from __future__ import annotations

import calendar
import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from tests.fast.dashboard.dummy_dump import _make_args
from tests.manual.dashboard_simulator.calibration import (
    CalibrationRun,
    CalibrationTrajectory,
    build_sample,
    sample_slots,
    shift_lifecycle,
)
from tests.manual.dashboard_simulator.dumps import ActorLayout, TrainDumpWriter

from miles.dashboard import hooks
from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.gpu_sampler import GpuSampler
from miles.dashboard.store import (
    DataBufferSample,
    EngineInfo,
    EngineSample,
    GpuProcessSample,
    GpuSample,
    MetricsRecord,
    PhaseEvent,
    Role,
    Stream,
    TopologySnapshot,
    TrajectoryEvent,
)
from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.utils.types import Sample

_CYCLE_GAP_S = 60.0


class VirtualClock:
    """Stands in for the ``time`` module inside ``hooks``, whose sink batching reads it."""

    def __init__(self, now: float):
        self.now = now

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now


class _InProcessCollectorHandle:
    """The collector's Ray handle, in process: ``handle.m.remote(*a)`` calls ``collector.m(*a)``."""

    def __init__(self, collector: DashboardCollector):
        self._collector = collector

    def __getattr__(self, method_name: str):
        return SimpleNamespace(remote=getattr(self._collector, method_name))


class ReplicaLayout:
    """Calibration node / engine / train rank -> its copy in replica ``replica``."""

    def __init__(self, calibration: CalibrationRun, replicas: int):
        calibration_nodes = sorted({lane["node"] for lane in calibration.lanes})
        self.node_by_calibration_node_and_replica = {
            (node, replica): _node_ip(replica * len(calibration_nodes) + i)
            for replica in range(replicas)
            for i, node in enumerate(calibration_nodes)
        }
        self.train_ranks_per_replica = sum(len(lane["ranks"]) for lane in calibration.lanes)

    def node(self, calibration_node: str, replica: int) -> str:
        return self.node_by_calibration_node_and_replica[(calibration_node, replica)]

    def engine_addr(self, calibration_addr: str, replica: int) -> str:
        host, port = calibration_addr.removeprefix("http://").rsplit(":", 1)
        return f"http://{self.node(host, replica)}:{port}"

    def train_rank(self, calibration_rank: int, replica: int) -> int:
        return replica * self.train_ranks_per_replica + calibration_rank


def _node_ip(n: int) -> str:
    return f"10.{n // 65536 % 256}.{n // 256 % 256}.{n % 256 + 1}"


class Replay:
    def __init__(
        self,
        calibration: CalibrationRun,
        *,
        replicas: int,
        hours: float,
        out: Path,
        actor_layout: ActorLayout | None,
        train_dumps_last_hours: float | None,
        rollout_dumps_last_hours: float | None,
    ):
        self.calibration = calibration
        self.replicas = replicas
        self.layout = ReplicaLayout(calibration, replicas)
        # the first cycle replays the whole calibration, later ones only its steady state
        self.steady_cycle_s = calibration.end_ts - calibration.steady_start_ts + _CYCLE_GAP_S
        self.end_ts = calibration.start_ts + hours * 3600
        # the reader opens one step's train shards at a time: older steps only cost disk
        self.train_dumps_from_ts = (
            None if train_dumps_last_hours is None else self.end_ts - train_dumps_last_hours * 3600
        )
        self.rollout_dumps_from_ts = (
            None if rollout_dumps_last_hours is None else self.end_ts - rollout_dumps_last_hours * 3600
        )
        self.clock = VirtualClock(calibration.start_ts)
        hooks.time = self.clock
        run_args = calibration.store.meta.args
        self.config = CollectorConfig(
            dashboard_dir=str(out / "dashboard"),
            run_name=f"{calibration.store.meta.run_name}-x{replicas}",
            start_ts=calibration.start_ts,
            args_snapshot=run_args
            | {key: run_args[key] * replicas for key in ("actor_num_nodes", "rollout_num_gpus", "rollout_batch_size")},
        )
        self.collector = DashboardCollector(config=self.config)
        handle = _InProcessCollectorHandle(self.collector)
        self.train_sinks_by_rank = self._build_train_sinks(handle)
        self.rollout_manager_sink = self._build_sink(
            handle, Role.ROLLOUT_MANAGER, node=_rollout_manager_node(calibration), gpus=[], rank=-1
        )
        self.trajectory_sink = hooks.TrajectorySink(handle)
        self.dump_args = _make_args(
            out,
            num_prompts=run_args["rollout_batch_size"] * replicas,
            n_samples_per_prompt=run_args["n_samples_per_prompt"],
        )
        if actor_layout is not None:
            # one optimizer step per rollout step, as in both calibration runs
            self.dump_args.global_batch_size = self.dump_args.rollout_batch_size * self.dump_args.n_samples_per_prompt
            self.dump_args.use_dynamic_batch_size = True
            self.dump_args.max_tokens_per_gpu = actor_layout.max_tokens_per_gpu
        train_gpus = calibration.train_gpus * replicas
        self.train_dump_writer = (
            None
            if actor_layout is None
            else TrainDumpWriter(self.dump_args, out=out, train_gpus=train_gpus, layout=actor_layout)
        )
        self.queue: list = []
        self.queued = 0
        self.telemetry_rows = 0

    def _build_sink(self, handle, role: Role, *, node: str, gpus: list[int], rank: int) -> hooks.PhaseSink:
        sink = hooks.PhaseSink(handle, role)
        sink._identity = hooks._Identity(node=node, gpus=gpus, rank=rank)
        sink._flusher = True  # the per-process flusher thread is _flush_idle_sinks
        return sink

    def _build_train_sinks(self, handle) -> dict[int, hooks.PhaseSink]:
        sinks_by_rank = {}
        for lane in self.calibration.lanes:
            for calibration_rank in lane["ranks"]:
                for replica in range(self.replicas):
                    rank = self.layout.train_rank(calibration_rank, replica)
                    sinks_by_rank[rank] = self._build_sink(
                        handle, Role.TRAIN, node=self.layout.node(lane["node"], replica), gpus=[lane["gpu"]], rank=rank
                    )
        return sinks_by_rank

    # ------------------------------ event loop ------------------------------

    def schedule_at(self, ts: float, action, *payload) -> None:
        self.queued += 1
        heapq.heappush(self.queue, (ts, self.queued, action, payload))

    def run(self) -> None:
        start = self.calibration.start_ts
        self.collector.update_topology(self._widened_topology(start))
        cycle = 0
        while (cycle_start_ts := self._cycle_from_ts(cycle) + self._cycle_shift_s(cycle)) < self.end_ts:
            self.schedule_at(cycle_start_ts, self._schedule_cycle, cycle)
            cycle += 1
        for interval, action in (
            (self.config.flush_interval_seconds, self._flush_collector),
            (hooks.BATCH_MAX_SECONDS, self._flush_idle_sinks),
        ):
            for k in range(1, int((self.end_ts - start) // interval) + 2):
                self.schedule_at(start + k * interval, action)

        started = time.monotonic()
        next_report = started + 30
        while self.queue:
            ts, _, action, payload = heapq.heappop(self.queue)
            if ts > self.end_ts:
                break
            self.clock.now = ts
            action(ts, *payload)
            if time.monotonic() >= next_report:
                next_report += 30
                simulated = (ts - start) / (self.end_ts - start)
                print(
                    f"  {simulated:6.1%} simulated, {self.telemetry_rows:,} telemetry rows, {time.monotonic() - started:.0f}s",
                    flush=True,
                )
        self.clock.now = self.end_ts
        for sink in [*self.train_sinks_by_rank.values(), self.rollout_manager_sink]:
            sink.flush()
        self.trajectory_sink.flush()
        self.collector.flush()

    def _cycle_from_ts(self, cycle: int) -> float:
        return self.calibration.start_ts if cycle == 0 else self.calibration.steady_start_ts

    def _cycle_shift_s(self, cycle: int) -> float:
        return cycle * self.steady_cycle_s

    def _schedule_cycle(self, ts: float, cycle: int) -> None:
        """One pass over the calibration run: telemetry an hour at a time
        (bounds memory), the sparse per-step events up front."""
        from_ts, shift_s = self._cycle_from_ts(cycle), self._cycle_shift_s(cycle)
        rollout_id_base = cycle * (max(self.calibration.rollout_ids) + 1)
        for stream in (Stream.GPU_UTIL, Stream.GPU_PROCESSES, Stream.ENGINE_SERIES):
            for path in self.calibration.telemetry_hour_files(stream):
                if _hour_start_ts(path.stem) + 3600 > from_ts:
                    hour_ts = max(ts, _hour_start_ts(path.stem) + shift_s - 1.0)
                    self.schedule_at(hour_ts, self._schedule_telemetry_hour, stream, path, from_ts, shift_s)
        for event in self.calibration.phases:
            if event.t0 >= from_ts:
                self.schedule_at((event.t0 if event.open else event.t1) + shift_s, self._push_phase, event, shift_s)
        for record in self.calibration.metrics:
            if record.ts >= from_ts:
                step = None if record.step is None else record.step + rollout_id_base
                self.schedule_at(record.ts + shift_s, self._push_metrics, record, step, shift_s)
        for record in self.calibration.data_buffer:
            if record.ts >= from_ts:
                self.schedule_at(record.ts + shift_s, self._push_data_buffer, record.length)
        consume_ts_by_rollout_id = self.calibration.consume_ts_by_rollout_id
        for position, calibration_rollout_id in enumerate(self.calibration.rollout_ids):
            if consume_ts_by_rollout_id[calibration_rollout_id] >= from_ts:
                self._schedule_rollout_step(position, calibration_rollout_id, rollout_id_base, shift_s)
        for calibration_rollout_id in self.calibration.eval_rollout_ids:
            if consume_ts_by_rollout_id[calibration_rollout_id] >= from_ts:
                self.schedule_at(
                    consume_ts_by_rollout_id[calibration_rollout_id] + shift_s,
                    self._dump_eval,
                    calibration_rollout_id,
                    calibration_rollout_id + rollout_id_base,
                    shift_s,
                )

    # --------------------------- telemetry replay ---------------------------

    def _schedule_telemetry_hour(self, ts: float, stream: Stream, path: Path, from_ts: float, shift_s: float) -> None:
        with open(path, "rb") as f:
            rows = sorted((row for row in map(json.loads, f) if row["ts"] >= from_ts), key=lambda row: row["ts"])
        if stream is Stream.ENGINE_SERIES:
            rows_by_scrape_ts = defaultdict(list)
            for row in rows:
                rows_by_scrape_ts[row["ts"]].append(row)
            for scrape_ts, scrape_rows in rows_by_scrape_ts.items():
                self.schedule_at(scrape_ts + shift_s, self._push_engine_scrape, scrape_rows, shift_s)
            return
        # one GpuSampler batch per node per flush
        interval = GpuSampler.FLUSH_INTERVAL_SECONDS
        batches_by_node_and_flush = defaultdict(list)
        for row in rows:
            batches_by_node_and_flush[(row["node"], int(row["ts"] // interval))].append(row)
        for (node, flush_index), batch in batches_by_node_and_flush.items():
            self.schedule_at(
                (flush_index + 1) * interval + shift_s, self._push_gpu_batch, stream, node, batch, shift_s
            )

    def _push_gpu_batch(
        self, ts: float, stream: Stream, calibration_node: str, batch: list[dict], shift_s: float
    ) -> None:
        record_type, push = (
            (GpuSample, self.collector.push_gpu_samples)
            if stream is Stream.GPU_UTIL
            else (GpuProcessSample, self.collector.push_gpu_processes)
        )
        for replica in range(self.replicas):
            node = self.layout.node(calibration_node, replica)
            push(node, [record_type(**(row | dict(ts=row["ts"] + shift_s, node=node))) for row in batch])
        self.telemetry_rows += len(batch) * self.replicas

    def _push_engine_scrape(self, ts: float, rows: list[dict], shift_s: float) -> None:
        # the GLM calibration's sglang predates the dp_rank label: its dp ranks
        # arrive as same-label rows in dp_rank order
        def label_key(row: dict) -> tuple:
            return row["addr"], row["metric"], tuple(sorted(row["labels"].items()))

        rows_per_key = Counter(label_key(row) for row in rows)
        dp_ranks = defaultdict(int)
        for row in rows:
            labels = dict(row["labels"])
            if rows_per_key[label_key(row)] > 1:
                labels["dp_rank"] = str(dp_ranks[label_key(row)])
                dp_ranks[label_key(row)] += 1
            for replica in range(self.replicas):
                self.collector._append(
                    EngineSample(
                        ts=row["ts"] + shift_s,
                        addr=self.layout.engine_addr(row["addr"], replica),
                        metric=row["metric"],
                        labels=labels,
                        value=row["value"],
                    )
                )
        self.telemetry_rows += len(rows) * self.replicas

    def _push_phase(self, ts: float, event: PhaseEvent, shift_s: float) -> None:
        if event.role == Role.ROLLOUT_MANAGER:
            sinks = [self.rollout_manager_sink]
        else:
            sinks = [
                self.train_sinks_by_rank[self.layout.train_rank(event.rank, replica)]
                for replica in range(self.replicas)
            ]
        for sink in sinks:
            if event.open:
                sink.begin(event.name, event.t0 + shift_s)
            else:
                sink(event.name, event.t0 + shift_s, event.t1 + shift_s)
        self.telemetry_rows += len(sinks)

    def _push_metrics(self, ts: float, record: MetricsRecord, step: int | None, shift_s: float) -> None:
        self.collector.push_metrics(
            MetricsRecord(ts=record.ts + shift_s, step_key=record.step_key, step=step, metrics=record.metrics)
        )

    def _push_data_buffer(self, ts: float, length: int) -> None:
        self.collector.push_data_buffer(DataBufferSample(ts=ts, length=length * self.replicas))

    def _flush_collector(self, ts: float) -> None:
        self.collector._sync_external_topology()
        self.collector.flush()

    def _flush_idle_sinks(self, ts: float) -> None:
        for sink in [*self.train_sinks_by_rank.values(), self.rollout_manager_sink]:
            if sink._buffer:
                sink.flush()

    def _widened_topology(self, ts: float) -> TopologySnapshot:
        engines = [
            EngineInfo(
                addr=self.layout.engine_addr(engine.addr, replica),
                worker_type=engine.worker_type,
                engine_rank=replica * len(self.calibration.engines) + engine.engine_rank,
                gpus=[[self.layout.node(node, replica), gpu] for node, gpu in engine.gpus],
                gpu_uuids=[f"{uuid}-r{replica}" for uuid in engine.gpu_uuids],
            )
            for replica in range(self.replicas)
            for engine in self.calibration.engines
        ]
        return TopologySnapshot(ts=ts, engines=engines)

    # ----------------------------- rollout steps -----------------------------

    def _schedule_rollout_step(
        self, position: int, calibration_rollout_id: int, rollout_id_base: int, shift_s: float
    ) -> None:
        rollout_ids = self.calibration.rollout_ids
        consume_ts_by_rollout_id = self.calibration.consume_ts_by_rollout_id
        rollout_id = calibration_rollout_id + rollout_id_base
        consume_ts = consume_ts_by_rollout_id[calibration_rollout_id] + shift_s
        sources = []
        for replica in range(self.replicas):
            source_rollout_id = rollout_ids[(position + replica) % len(rollout_ids)]
            source_shift_s = consume_ts - consume_ts_by_rollout_id[source_rollout_id]
            sources.append((source_rollout_id, source_shift_s))
            for trajectory in self.calibration.trajectories_by_rollout_id[source_rollout_id]:
                lifecycle_sample = self._lifecycle_sample(
                    trajectory,
                    rollout_id=rollout_id,
                    replica=replica,
                    weight_version_shift=rollout_id - source_rollout_id,
                    shift_s=source_shift_s,
                )
                if trajectory.recorded_events:
                    for event in trajectory.recorded_events:
                        self.schedule_at(
                            event.ts + source_shift_s,
                            self._replay_trajectory_event,
                            lifecycle_sample,
                            event,
                            rollout_id - source_rollout_id,
                        )
                    continue
                self.schedule_at(trajectory.attempt_start_ts + source_shift_s, self._attempt_start, lifecycle_sample)
                self.schedule_at(trajectory.attempt_end_ts + source_shift_s, self._attempt_end, lifecycle_sample)
        self.schedule_at(consume_ts, self._dump_rollout_step, rollout_id, sources)

    def _sample_index(self, rollout_id: int, replica: int, slot: int) -> int:
        samples_per_replica = self.dump_args.rollout_batch_size // self.replicas * self.dump_args.n_samples_per_prompt
        assert slot < samples_per_replica, (
            f"a calibration step holds more than {samples_per_replica} distinct sample indices: "
            "its widened indices would run into the next replica's"
        )
        return (rollout_id * self.replicas + replica) * samples_per_replica + slot

    def _lifecycle_sample(
        self,
        trajectory: CalibrationTrajectory,
        *,
        rollout_id: int,
        replica: int,
        weight_version_shift: int,
        shift_s: float,
    ) -> SimpleNamespace:
        """The fields ``TrajectorySink`` reads off a sample, without its tokens."""
        index = self._sample_index(rollout_id, replica, trajectory.slot)
        return SimpleNamespace(
            index=index,
            group_index=index // self.dump_args.n_samples_per_prompt,
            all_weight_version_spans=[
                SimpleNamespace(version=str(trajectory.last_weight_version + weight_version_shift))
            ],
            metadata=dict(lifecycle=shift_lifecycle(trajectory.lifecycle, shift_s)),
            status=Sample.Status(trajectory.status),
        )

    def _attempt_start(self, ts: float, lifecycle_sample: SimpleNamespace) -> None:
        # no weight version before the first turn finishes
        starting = SimpleNamespace(**(vars(lifecycle_sample) | dict(all_weight_version_spans=[])))
        self.trajectory_sink.attempt_start(starting)
        self.trajectory_sink.gen_start(starting)
        self.telemetry_rows += 2

    def _replay_trajectory_event(
        self, ts: float, lifecycle_sample: SimpleNamespace, event: TrajectoryEvent, weight_version_shift: int
    ) -> None:
        version = event.weight_version
        if version.isdigit():
            version = str(int(version) + weight_version_shift)
        sample = SimpleNamespace(
            **(
                vars(lifecycle_sample)
                | dict(all_weight_version_spans=[SimpleNamespace(version=version)] if version else [])
            )
        )
        self.trajectory_sink._emit(event.kind, sample, ts=ts, turn=event.turn, detail=event.detail)
        self.telemetry_rows += 1

    def _attempt_end(self, ts: float, lifecycle_sample: SimpleNamespace) -> None:
        self.trajectory_sink.attempt_end(lifecycle_sample)
        self.telemetry_rows += 4 * len(lifecycle_sample.metadata["lifecycle"])

    def _dump_rollout_step(self, ts: float, rollout_id: int, sources: list[tuple[int, float]]) -> None:
        if self.rollout_dumps_from_ts is not None and ts < self.rollout_dumps_from_ts:
            return
        samples = []
        for replica, (source_rollout_id, source_shift_s) in enumerate(sources):
            messages_by_key = self.calibration.load_messages(source_rollout_id)
            occurrences = defaultdict(int)
            # siblings of one rollout share its id: the new index of its first sample
            new_rollout_id_by_calibration_rollout_id: dict[int, int] = {}
            calibration_samples = self.calibration.load_samples(source_rollout_id)
            for slot, calibration_sample in zip(sample_slots(calibration_samples), calibration_samples, strict=True):
                index = self._sample_index(rollout_id, replica, slot)
                # dumps from before Sample.rollout_id existed carry no such key
                calibration_rollout_id = calibration_sample.get("rollout_id")
                sample = build_sample(
                    calibration_sample,
                    index=index,
                    group_index=index // self.dump_args.n_samples_per_prompt,
                    rollout_id=(
                        None
                        if calibration_rollout_id is None
                        else new_rollout_id_by_calibration_rollout_id.setdefault(calibration_rollout_id, index)
                    ),
                    shift_s=source_shift_s,
                    weight_version_shift=rollout_id - source_rollout_id,
                )
                key = (calibration_sample["index"], occurrences[calibration_sample["index"]])
                occurrences[calibration_sample["index"]] += 1
                if key in messages_by_key:
                    sample.metadata["messages"] = messages_by_key[key]
                samples.append(sample)
        save_debug_rollout_data(self.dump_args, samples, rollout_id=rollout_id, evaluation=False)
        if self.train_dump_writer is not None and (self.train_dumps_from_ts is None or ts >= self.train_dumps_from_ts):
            self.train_dump_writer.write(rollout_id, samples)
        print(f"  rollout {rollout_id}: {len(samples)} samples dumped", flush=True)

    def _dump_eval(self, ts: float, calibration_rollout_id: int, rollout_id: int, shift_s: float) -> None:
        """Eval batches do not grow with the cluster: replayed once, unwidened."""
        messages_by_key = self.calibration.load_messages(calibration_rollout_id, evaluation=True)
        samples = []
        for calibration_sample in self.calibration.load_samples(calibration_rollout_id, evaluation=True):
            sample = build_sample(
                calibration_sample,
                index=calibration_sample["index"],
                group_index=calibration_sample["group_index"],
                rollout_id=calibration_sample.get("rollout_id"),
                shift_s=shift_s,
                weight_version_shift=rollout_id - calibration_rollout_id,
            )
            if (messages := messages_by_key.get((calibration_sample["index"], 0))) is not None:
                sample.metadata["messages"] = messages
            samples.append(sample)
        save_debug_rollout_data(self.dump_args, {"eval": {"samples": samples}}, rollout_id=rollout_id, evaluation=True)


def _rollout_manager_node(calibration: CalibrationRun) -> str:
    return next(event.node for event in calibration.phases if event.role == Role.ROLLOUT_MANAGER)


def _hour_start_ts(partition_stem: str) -> float:
    return float(calendar.timegm(time.strptime(partition_stem, "%Y%m%d_%H")))
