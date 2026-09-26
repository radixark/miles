"""Executable integration probes using real Sample serialization and Miles admission."""

import asyncio
import os
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import continuous_collect as cc
import numpy as np
from frozen_collect import persist

from miles.rollout.base_types import RolloutFnTrainInput
from miles.utils.types import Sample


def sample(slot: int) -> Sample:
    return Sample(index=slot, group_index=slot // 2, metadata={"instance_id": f"task-{slot // 2}"})


def complete(original: Sample) -> Sample:
    s = deepcopy(original)
    s.tokens = [1, 2]
    s.response_length = 1
    s.response = "answer"
    s.loss_mask = [1]
    s.rollout_log_probs = [-0.5]
    s.rollout_topk_token_ids = np.array([[2]], dtype=np.int32)
    s.rollout_topk_log_probs = np.array([[-0.5]], dtype=np.float32)
    s.status = Sample.Status.COMPLETED
    s.metadata.update({"exit_status": "Submitted", "eval_report": {"ok": True}, "reward": 0.0, "frozen_sampling_seed": 20260924 + s.index})
    s.reward = 0.0
    return s


def collector(groups: int) -> cc.RolloutFn:
    c = object.__new__(cc.RolloutFn)
    c.state = SimpleNamespace(args=SimpleNamespace(debug_rollout_only=True, start_rollout_id=0, num_rollout=groups, rollout_batch_size=1, n_samples_per_prompt=2), sampling_params={})
    sequence = iter([[sample(g * 2), sample(g * 2 + 1)] for g in range(groups)])
    c.data_source = SimpleNamespace(get_samples=lambda count: [next(sequence) for _ in range(count)])
    return c


async def backfill(root: Path) -> None:
    os.environ.update(PILOT_ROOT=str(root), SANDBOX_CONCURRENCY="2", FROZEN_RESUME_ROOTS="")
    c = collector(4)
    release = asyncio.Event()
    later = asyncio.Event()
    active = peak = 0
    calls = Counter()

    async def generate(state: object, s: Sample, params: dict) -> list[Sample]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls[s.index] += 1
        try:
            if s.index == 0:
                await release.wait()
            if s.index == 2:
                later.set()
            await asyncio.sleep(0)
            return [complete(s)]
        finally:
            active -= 1

    with patch.object(cc, "generate_and_rm", generate):
        first = asyncio.create_task(c._call_train(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.wait_for(later.wait(), 5)
        assert not first.done(), "Slow first batch must not block later submission"
        release.set()
        outputs = [await first]
        for i in range(1, 4):
            outputs.append(await c._call_train(RolloutFnTrainInput(rollout_id=i)))
        await c._producer
    assert calls == Counter(range(8)) and peak <= 2
    assert [[s.index for s in out.samples[0]] for out in outputs] == [[i, i + 1] for i in range(0, 8, 2)]
    assert len((root / "attempts.jsonl").read_text().splitlines()) == 8
    assert all(isinstance(e, dict) for e in c._results.values())


async def resume(root: Path) -> None:
    parent = root / "parent"
    parent.mkdir()
    for slot, attempts, valid in [(0, 1, True), (1, 1, False), (2, 2, False), (3, 1, True)]:
        for attempt in range(attempts):
            persist(parent, slot, attempt, [complete(sample(slot))], valid)
    target = root / "resumed"
    os.environ.update(PILOT_ROOT=str(target), SANDBOX_CONCURRENCY="2", FROZEN_RESUME_ROOTS=str(parent))
    c = collector(2)
    calls = []

    async def generate(state: object, s: Sample, params: dict) -> list[Sample]:
        calls.append((s.index, s.metadata["frozen_attempt"]))
        return [complete(s)]

    with patch.object(cc, "generate_and_rm", generate):
        first = await c._call_train(RolloutFnTrainInput(rollout_id=0))
        second = await c._call_train(RolloutFnTrainInput(rollout_id=1))
        await c._producer
    assert calls == [(1, 1)], calls
    assert len(first.samples) == 1 and second.samples == []
    history = cc.read_history([parent, target])
    assert [e["attempt"] for e in history[1]] == [0, 1]
    os.environ["FROZEN_RESUME_ROOTS"] = str(parent) + ":" + str(target)
    os.environ["PILOT_ROOT"] = str(root / "again")
    c = collector(2)
    with patch.object(cc, "generate_and_rm", side_effect=AssertionError("No regeneration")):
        await c._call_train(RolloutFnTrainInput(rollout_id=0))
        await c._call_train(RolloutFnTrainInput(rollout_id=1))
        await c._producer
    path = Path(history[0][0]["path"])
    path.write_bytes(path.read_bytes() + b"corrupt")
    os.environ["PILOT_ROOT"] = str(root / "corrupt")
    c = collector(2)
    try:
        await asyncio.wait_for(c._call_train(RolloutFnTrainInput(rollout_id=0)), 5)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("Corruption must fail closed")


async def failure(root: Path) -> None:
    os.environ.update(PILOT_ROOT=str(root), SANDBOX_CONCURRENCY="2", FROZEN_RESUME_ROOTS="")
    c = collector(2)
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def generate(state: object, s: Sample, params: dict) -> list[Sample]:
        if s.index == 0:
            await started.wait()
            raise RuntimeError("synthetic failure")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with patch.object(cc, "generate_and_rm", generate):
        try:
            await asyncio.wait_for(c._call_train(RolloutFnTrainInput(rollout_id=0)), 5)
        except RuntimeError as error:
            assert str(error) == "synthetic failure"
        else:
            raise AssertionError("Worker failure was swallowed")
    assert cancelled.is_set()


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        await backfill(root / "backfill")
        await resume(root)
        await failure(root / "failure")
    print("PASS: cross-batch trajectory backfill, bounded concurrency, deterministic groups, disk references, partial/exhausted/valid resume, repeated resume, corruption rejection, failure cleanup")


if __name__ == "__main__":
    asyncio.run(main())
