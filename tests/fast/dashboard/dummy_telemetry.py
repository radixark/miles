"""Generate DUMMY live-collection telemetry through miles' REAL store write path.

Companion of ``dummy_dump.py`` (which fabricates the ``.pt`` dump side): only
the *content* here is fake — every record goes through the actual
``MetricStore`` append/flush pipeline into ``{dump_details}/dashboard/``, so
schema or query changes break the timeline tests instead of drifting
silently.

This file doubles as the OUTPUT CONTRACT for the live collector (PR-10 in
the implementation plan): the collector must produce files that
``MetricStore.load`` plus the timeline queries interpret exactly like this
fixture.

The fabricated story deliberately contains the shapes the efficiency view is
designed to surface:

1. colocate cadence per 100 s step: rollout [T, T+40) then per-rank
   ``train_wait`` / ``actor_train`` / ``update_weights`` with cross-rank skew;
2. a rollout long tail: engine A stays busy until T+38 while engine B drains
   at T+25 (util drops, ``num_running_reqs`` hits 0 — a bubble);
3. a fault-tolerance engine restart MID-ROLLOUT of step 1: engine B moves
   from port 15004 to 15006, so topology windows split and the rollout-lane
   expansion must clip intervals at the boundary;
4. one util spike (100%) at a known instant for downsampling assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from miles.dashboard.store import (
    EngineInfo,
    EngineSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Role,
    TopologySnapshot,
    TrajectoryEvent,
    TrajectoryEventKind,
)

BASE_TS = 1_000_000.0
GPU_NODE = "10.0.0.1"
MANAGER_NODE = "10.0.0.2"  # rollout manager runs on a different (GPU-less) node

ENGINE_A = "http://10.0.0.1:15000"
ENGINE_B_OLD = "http://10.0.0.1:15004"
ENGINE_B_NEW = "http://10.0.0.1:15006"

STEP_SECONDS = 100
INIT_SECONDS = 30  # gap between collector start (meta.start_ts) and step 0
ROLLOUT_SECONDS = 40
DRAIN_A = 38  # engine A busy almost to the end of the rollout window
DRAIN_B = 25  # engine B drains early -> bubble on its lanes
RESTART_OFFSET = 20  # engine B restarts at step-1 rollout start + 20s (mid-rollout)

SPIKE = dict(step=0, offset=50, gpu=0, util=100)


@dataclass
class DummyTelemetryTruth:
    """Ground truth returned by :func:`dump_dummy_telemetry` for test assertions."""

    steps: int
    gpus: int
    restart_ts: float
    spike_ts: float

    def step_start(self, step: int) -> float:
        return BASE_TS + INIT_SECONDS + step * STEP_SECONDS

    def rollout_interval(self, step: int) -> tuple[float, float]:
        return self.step_start(step), self.step_start(step) + ROLLOUT_SECONDS


SAMPLES_PER_STEP = 8  # dummy_dump's default num_prompts * n_samples_per_prompt


def _trajectory_events(store: MetricStore, truth: DummyTelemetryTruth, samples_per_step: int) -> None:
    """Lifecycle events for each step's batch, aligned with dummy_dump's
    global sample indices (step * n + i) and its agentic pattern (every third
    sample: two turns, one tool call, mixed weight versions)."""
    for step in range(truth.steps):
        t0, t1 = truth.rollout_interval(step)
        for i in range(samples_per_step):
            index = step * samples_per_step + i
            agentic = index % 3 == 0
            # stagger submits so every lifecycle fits inside the rollout window
            start = t0 + i * (t1 - t0 - 16.0) / samples_per_step
            version = str(index % 3)

            def emit(kind, ts, turn=-1, version=version, detail="", index=index, group=i // 2):
                store.append(
                    TrajectoryEvent(
                        ts=ts,
                        kind=kind,
                        sample_index=index,
                        group_index=group,
                        turn=turn,
                        weight_version=version,
                        detail=detail,
                    )
                )

            emit(TrajectoryEventKind.ATTEMPT_START, start)
            emit(TrajectoryEventKind.GEN_START, start + 1.0, turn=1)
            emit(TrajectoryEventKind.GEN_END, start + 6.0, turn=1)
            if agentic:
                emit(TrajectoryEventKind.TOOL_START, start + 6.0, turn=1, detail="1 calls")
                emit(TrajectoryEventKind.TOOL_END, start + 9.0, turn=1, detail="1 calls")
                bumped = str(index % 3 + 1)
                emit(TrajectoryEventKind.GEN_START, start + 9.0, turn=2, version=bumped)
                emit(TrajectoryEventKind.GEN_END, start + 14.0, turn=2, version=bumped)
            emit(TrajectoryEventKind.ATTEMPT_END, start + (15.0 if agentic else 7.0), detail="completed")


def dump_dummy_telemetry(
    dump_dir: Path, *, steps: int = 3, gpus: int = 4, samples_per_step: int = SAMPLES_PER_STEP
) -> DummyTelemetryTruth:
    truth = DummyTelemetryTruth(
        steps=steps,
        gpus=gpus,
        restart_ts=BASE_TS + INIT_SECONDS + STEP_SECONDS + RESTART_OFFSET,
        spike_ts=BASE_TS + INIT_SECONDS + SPIKE["step"] * STEP_SECONDS + SPIKE["offset"],
    )
    store = MetricStore(dump_dir / "dashboard")
    store.write_meta(Meta(run_name="dummy-telemetry", start_ts=BASE_TS, args={"colocate": True}))

    def engine(addr: str, engine_rank: int, gpu_ids: list[int]) -> EngineInfo:
        return EngineInfo(
            addr=addr,
            worker_type="regular",
            engine_rank=engine_rank,
            gpus=[[GPU_NODE, g] for g in gpu_ids],
            gpu_uuids=[None] * len(gpu_ids),
        )

    store.append(TopologySnapshot(ts=BASE_TS, engines=[engine(ENGINE_A, 0, [0, 1]), engine(ENGINE_B_OLD, 1, [2, 3])]))
    store.append(
        TopologySnapshot(ts=truth.restart_ts, engines=[engine(ENGINE_A, 0, [0, 1]), engine(ENGINE_B_NEW, 1, [2, 3])])
    )

    def engine_b_addr(ts: float) -> str:
        return ENGINE_B_OLD if ts < truth.restart_ts else ENGINE_B_NEW

    for offset in range(INIT_SECONDS):
        for gpu in range(gpus):
            store.append(GpuSample(ts=BASE_TS + offset, node=GPU_NODE, gpu=gpu, util=20, mem_mb=30_000, power_w=300))

    for step in range(steps):
        t = truth.step_start(step)
        rollout_end = t + ROLLOUT_SECONDS
        store.append(
            PhaseEvent(
                name="rollout", t0=t, t1=rollout_end, node=MANAGER_NODE, gpus=[], rank=-1, role=Role.ROLLOUT_MANAGER
            )
        )
        for rank in range(gpus):
            wait_end = rollout_end + 1 + 0.2 * rank
            train_end = t + 80 + 0.5 * rank
            store.append(
                PhaseEvent(
                    name="train_wait",
                    t0=rollout_end,
                    t1=wait_end,
                    node=GPU_NODE,
                    gpus=[rank],
                    rank=rank,
                    role=Role.TRAIN,
                )
            )
            store.append(
                PhaseEvent(
                    name="actor_train",
                    t0=wait_end,
                    t1=train_end,
                    node=GPU_NODE,
                    gpus=[rank],
                    rank=rank,
                    role=Role.TRAIN,
                )
            )
            store.append(
                PhaseEvent(
                    name="update_weights",
                    t0=train_end,
                    t1=train_end + 4 + 0.3 * rank,
                    node=GPU_NODE,
                    gpus=[rank],
                    rank=rank,
                    role=Role.TRAIN,
                )
            )
        store.append(
            MetricsRecord(
                ts=t + STEP_SECONDS - 5,
                step_key="rollout/step",
                step=step,
                metrics={
                    "perf/step_time": 95.0 - step,
                    "perf/wait_time_ratio": 0.4 - 0.1 * step,
                    "rollout/rewards_mean": 0.4 + 0.05 * step,
                },
            )
        )

        for offset in range(STEP_SECONDS):
            ts = t + offset
            for gpu in range(gpus):
                drain = DRAIN_A if gpu < 2 else DRAIN_B
                if offset < ROLLOUT_SECONDS:
                    util = 90 if offset < drain else 5
                elif offset < 80:
                    util = 95
                else:
                    util = 30
                if step == SPIKE["step"] and gpu == SPIKE["gpu"] and offset == SPIKE["offset"]:
                    util = SPIKE["util"]
                store.append(
                    GpuSample(ts=ts, node=GPU_NODE, gpu=gpu, util=util, mem_mb=60_000 + 100 * gpu, power_w=600)
                )

        for offset in range(0, STEP_SECONDS, 2):
            ts = t + offset
            for addr, drain in ((ENGINE_A, DRAIN_A), (engine_b_addr(ts), DRAIN_B)):
                running = 16.0 if offset < min(drain, ROLLOUT_SECONDS) else 0.0
                store.append(
                    EngineSample(ts=ts, addr=addr, metric="sglang_num_running_reqs", labels={}, value=running)
                )
                store.append(
                    EngineSample(ts=ts, addr=addr, metric="sglang_gen_throughput", labels={}, value=running * 50.0)
                )
                store.append(
                    EngineSample(
                        ts=ts, addr=addr, metric="sglang_generation_tokens_total", labels={}, value=offset * 100.0
                    )
                )
                store.append(
                    EngineSample(
                        ts=ts,
                        addr=addr,
                        metric="sglang_time_to_first_token_seconds_sum",
                        labels={},
                        value=offset * 0.4,
                    )
                )
                store.append(
                    EngineSample(
                        ts=ts,
                        addr=addr,
                        metric="sglang_time_to_first_token_seconds_count",
                        labels={},
                        value=offset * 2.0,
                    )
                )

    _trajectory_events(store, truth, samples_per_step)
    store.flush()
    return truth
