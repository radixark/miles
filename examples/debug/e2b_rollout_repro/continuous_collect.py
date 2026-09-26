"""Continuously fill frozen trajectory slots while exporting deterministic batches.

The fixed corpus is persisted per slot. Background tasks hold artifact references,
not completed token arrays, so a slow group cannot stall submission or retain the
whole corpus in RAM. This collector must only run with debug_rollout_only.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from frozen_collect import eligible, persist

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, generate_and_rm
from miles.rollout.submission_scheduler import SampleBackfillSubmission
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def read_history(roots: list[Path]) -> dict[int, list[dict]]:
    """Merge immutable parent manifests without resetting or duplicating attempts."""
    by_slot = defaultdict(dict)
    for root in roots:
        manifest = root / "attempts.jsonl"
        if not manifest.exists():
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            path = Path(event["path"])
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Artifact outside parent root: {path}")
            key = event["attempt"]
            prior = by_slot[event["slot"]].get(key)
            if prior is not None and prior["sha256"] != event["sha256"]:
                raise ValueError(f"Conflicting attempt: {event['slot']}/{key}")
            by_slot[event["slot"]][key] = event
    result = {}
    for slot, events in by_slot.items():
        ordered = [events[k] for k in sorted(events)]
        if sorted(events) != list(range(len(events))) or len(events) > 2:
            raise ValueError(f"Invalid retry history for slot {slot}")
        if any(e["valid"] for e in ordered[:-1]):
            raise ValueError(f"Attempt after valid result for slot {slot}")
        result[slot] = ordered
    return result


def load_saved(event: dict, original: Sample) -> list[Sample]:
    """Verify checksum and identity before allowing an archived trace into training."""
    path = Path(event["path"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != event["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    with gzip.open(path, "rb") as stream:
        saved = torch.load(stream, weights_only=False)
    samples = [Sample.from_dict(s) for s in saved["samples"]]
    if not event["valid"] or not saved["valid"] or len(samples) != 1:
        raise ValueError(f"Invalid training artifact: {path}")
    sample = samples[0]
    if sample.index != original.index or sample.group_index != original.group_index or sample.metadata.get("instance_id") != original.metadata.get("instance_id") or sample.metadata.get("frozen_sampling_seed") != 20260924 + original.index or not eligible(sample):
        raise ValueError(f"Resume identity or trace validation failed: {path}")
    return samples


class RolloutFn(InferenceRolloutFn):
    async def _collect_slot(self, original: Sample) -> dict | None:
        events = self._history.get(original.index, [])
        for event in events:
            if event["task"] != original.metadata.get("instance_id"):
                raise ValueError(f"Task mismatch for slot {original.index}")
            # Audit every parent attempt, including terminal rejected attempts.
            async with self._archive_io:
                await asyncio.to_thread(self._verify_hash, event)
        if events and events[-1]["valid"]:
            async with self._archive_io:
                await asyncio.to_thread(load_saved, events[-1], original)
            return events[-1]
        for attempt in range(len(events), 2):
            sample = deepcopy(original)
            sample.metadata.update({"frozen_slot": original.index, "frozen_attempt": attempt, "frozen_sampling_seed": 20260924 + original.index})
            params = dict(self.state.sampling_params, sampling_seed=20260924 + original.index)
            output = await generate_and_rm(self.state, sample, params)
            samples = output if isinstance(output, list) else [output]
            valid = len(samples) == 1 and all(eligible(s) for s in samples)
            async with self._archive_io:
                await asyncio.to_thread(persist, self._root, original.index, attempt, samples, valid)
            path = self._root / "slots" / f"{original.index:05d}-{attempt}.pt.gz"
            # persist() completed before freeing capacity. Its manifest is the source of truth.
            event = {"slot": original.index, "attempt": attempt, "path": str(path), "valid": valid}
            async with self._archive_io:
                event["sha256"] = await asyncio.to_thread(self._digest, path)
            logger.info("continuous frozen slot=%s attempt=%s valid=%s", original.index, attempt, valid)
            if valid:
                return event
        return None

    @staticmethod
    def _digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @classmethod
    def _verify_hash(cls, event: dict) -> None:
        if cls._digest(Path(event["path"])) != event["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {event['path']}")

    async def _produce(self) -> None:
        # Reuse Miles' accounting with singleton admission units: a completed
        # trajectory (including its bounded retry) admits one new trajectory.
        scheduler = SampleBackfillSubmission(1)
        pending = {}
        originals = iter(s for group in self._groups for s in group)
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and scheduler.has_capacity(pending_groups=len(pending), group_budget=self._concurrency):
                    original = next(originals, None)
                    if original is None:
                        exhausted = True
                        break
                    scheduler.on_submit([[original]])
                    task = asyncio.create_task(self._collect_slot(original))
                    pending[task] = original.index
                if not pending:
                    break
                done, _ = await scheduler.wait_for_progress(set(pending))
                for task in done:
                    slot = pending.pop(task)
                    scheduler.sample_done_callback()
                    self._results[slot] = task.result()
                    self._settled[slot].set()
                self._write_progress(len(pending))
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _write_progress(self, active: int) -> None:
        report = {"planned_slots": len(self._settled), "settled_slots": len(self._results), "usable_slots": sum(e is not None for e in self._results.values()), "active_slots": active, "concurrency_limit": self._concurrency}
        temporary = self._root / "progress.tmp"
        temporary.write_text(json.dumps(report))
        temporary.replace(self._root / "progress.json")

    def _start(self) -> None:
        args = self.state.args
        if not args.debug_rollout_only or args.start_rollout_id != 0:
            raise ValueError("Continuous frozen collection requires rollout-only mode and start ID zero")
        self._root = Path(os.environ["PILOT_ROOT"])
        self._root.mkdir(parents=True, exist_ok=True)
        parents = os.environ.get("FROZEN_RESUME_ROOTS", os.environ.get("FROZEN_RESUME_ROOT", ""))
        self._history = read_history([Path(p) for p in parents.split(":") if p] + [self._root])
        self._concurrency = int(os.environ.get("SANDBOX_CONCURRENCY", "128"))
        if self._concurrency < 1:
            raise ValueError("Concurrency must be positive")
        # Freeze the same data-source sequence that the old collector consumed.
        self._groups = []
        for _ in range(args.num_rollout):
            groups = self.data_source.get_samples(args.rollout_batch_size)
            if len(groups) != args.rollout_batch_size or any(len(g) != args.n_samples_per_prompt for g in groups):
                raise ValueError("Unexpected corpus group shape")
            self._groups.extend(groups)
        originals = [s for group in self._groups for s in group]
        slots = [s.index for s in originals]
        if len(set(slots)) != len(slots) or not set(self._history).issubset(slots):
            raise ValueError("Duplicate slots or parent outside requested corpus")
        self._settled = {slot: asyncio.Event() for slot in slots}
        self._results = {}
        self._archive_io = asyncio.Semaphore(2)
        self._producer = asyncio.create_task(self._produce())

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        if not hasattr(self, "_producer"):
            self._start()
        size = self.state.args.rollout_batch_size
        groups = self._groups[input.rollout_id * size : (input.rollout_id + 1) * size]
        if len(groups) != size:
            raise ValueError("Collection batch outside fixed corpus")
        waiter = asyncio.ensure_future(asyncio.gather(*(self._settled[s.index].wait() for g in groups for s in g)))
        try:
            await asyncio.wait({waiter, self._producer}, return_when=asyncio.FIRST_COMPLETED)
            if self._producer.done():
                self._producer.result()  # Propagate producer failure instead of hanging forever.
            await waiter
        except BaseException:
            self._producer.cancel()
            await asyncio.gather(self._producer, return_exceptions=True)
            raise
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        retained = []
        for group in groups:
            if all(self._results[s.index] is not None for s in group):
                loaded = []
                for original in group:
                    async with self._archive_io:
                        samples = await asyncio.to_thread(load_saved, self._results[original.index], original)
                    loaded.append(samples[0])
                retained.append(loaded)
        report = {"rollout_id": input.rollout_id, "planned_slots": sum(map(len, groups)), "usable_slots": sum(self._results[s.index] is not None for g in groups for s in g), "complete_groups": len(retained)}
        (self._root / f"batch-{input.rollout_id:02d}.json").write_text(json.dumps(report))
        return RolloutFnTrainOutput(samples=retained, metrics={"collection/" + k: v for k, v in report.items()})
