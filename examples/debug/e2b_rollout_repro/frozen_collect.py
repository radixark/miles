"""Collect fixed task slots without success filtering or unbounded replacement groups."""
import asyncio
import gzip
import hashlib
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import torch

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, generate_and_rm
from miles.utils.types import Sample
from reward_policy import apply_reward_policy

logger = logging.getLogger(__name__)


def eligible(sample: Sample) -> bool:
    """Keep budget failures; reject missing traces and infrastructure failures."""
    sample.metadata = apply_reward_policy(sample.metadata or {})
    if not sample.metadata.get("pilot_valid_for_training"):
        return False
    sample.reward = float(sample.metadata["reward"])
    n = sample.response_length
    return (
        sample.status != Sample.Status.ABORTED
        and 0 < n <= len(sample.tokens) <= 65536
        and sample.loss_mask is not None
        and len(sample.loss_mask) == n
        and any(sample.loss_mask)
        and sample.rollout_log_probs is not None
        and len(sample.rollout_log_probs) == n
        and sample.rollout_topk_token_ids is not None
        and len(sample.rollout_topk_token_ids) == n
        and sample.rollout_topk_log_probs is not None
        and len(sample.rollout_topk_log_probs) == n
    )


def persist(root: Path, slot: int, attempt: int, samples: list[Sample], valid: bool) -> None:
    """Preserve every attempt before retrying; compression is lossless."""
    path = root / "slots" / f"{slot:05d}-{attempt}.pt.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        torch.save({"samples": [s.to_dict() for s in samples], "valid": valid}, stream)
    temporary.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    event = {"slot": slot, "attempt": attempt, "valid": valid, "sha256": digest,
             "path": str(path), "rewards": [s.reward for s in samples],
             "statuses": [s.metadata.get("exit_status") for s in samples],
             "task": samples[0].metadata.get("instance_id") if samples else None}
    with (root / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def restore_slot(root: Path, original: Sample) -> tuple[list[Sample] | None, bool]:
    """Reuse audited parent attempts without resetting their retry budget."""
    events = [
        json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    events = sorted((e for e in events if e["slot"] == original.index), key=lambda e: e["attempt"])
    if not events:
        return None, False
    assert [e["attempt"] for e in events] == list(range(len(events)))
    for event in events:
        path = Path(event["path"])
        assert path.resolve().is_relative_to(root.resolve())
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == event["sha256"]
        if event["valid"]:
            with gzip.open(path, "rb") as stream:
                saved = torch.load(stream, weights_only=False)
            samples = [Sample.from_dict(s) for s in saved["samples"]]
            assert saved["valid"] and len(samples) == 1
            sample = samples[0]
            assert sample.index == original.index and sample.group_index == original.group_index
            assert sample.metadata.get("instance_id") == original.metadata.get("instance_id")
            assert sample.metadata.get("frozen_sampling_seed") == 20260924 + original.index
            assert eligible(sample)
            return samples, False
    assert len(events) == 2, "Incomplete parent retry needs explicit recovery; do not reset its budget"
    return None, True


class RolloutFn(InferenceRolloutFn):
    async def _slot(self, original: Sample, rollout_id: int) -> list[Sample] | None:
        root = Path(os.environ["PILOT_ROOT"])
        parents = os.environ.get("FROZEN_RESUME_ROOTS", os.environ.get("FROZEN_RESUME_ROOT", ""))
        for parent in filter(None, parents.split(":")):
            replayed, exhausted = await asyncio.to_thread(restore_slot, Path(parent), original)
            if replayed is not None or exhausted:
                logger.info("frozen resume slot=%s valid=%s", original.index, replayed is not None)
                return replayed
        # One replacement at most, only for unusable/infrastructure outcomes.
        for attempt in range(2):
            sample = deepcopy(original)
            sample.metadata.update({"frozen_slot": original.index, "frozen_attempt": attempt,
                                    "frozen_sampling_seed": 20260924 + original.index})
            params = dict(self.state.sampling_params, sampling_seed=20260924 + original.index)
            output = await generate_and_rm(self.state, sample, params)
            samples = output if isinstance(output, list) else [output]
            checks = [eligible(s) for s in samples]
            valid = bool(samples) and all(checks)
            # Linear history was frozen for this corpus; unexpected segmentation must be audited.
            valid = valid and len(samples) == 1
            await asyncio.to_thread(persist, root, original.index, attempt, samples, valid)
            logger.info("frozen slot=%s attempt=%s valid=%s", original.index, attempt, valid)
            if valid:
                return samples
        return None

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        assert self.state.args.debug_rollout_only, "Frozen collection must never train online"
        groups = self.data_source.get_samples(self.state.args.rollout_batch_size)
        assert len(groups) == 16 and all(len(group) == 8 for group in groups)
        # Native generation semaphore covers collection as well as the agent call.
        results = await asyncio.gather(*(self._slot(s, input.rollout_id) for g in groups for s in g))
        retained = []
        for i in range(0, len(results), 8):
            group = results[i:i + 8]
            if all(s is not None for s in group):
                retained.append([s[0] for s in group])
        report = {"rollout_id": input.rollout_id, "planned_slots": len(results),
                  "usable_slots": sum(s is not None for s in results), "complete_groups": len(retained)}
        (Path(os.environ["PILOT_ROOT"]) / f"batch-{input.rollout_id:02d}.json").write_text(json.dumps(report))
        if not retained:
            logger.warning("No complete groups in wave %s; artifacts preserved, collection continues", input.rollout_id)
        return RolloutFnTrainOutput(samples=retained, metrics={"collection/" + k: v for k, v in report.items()})
