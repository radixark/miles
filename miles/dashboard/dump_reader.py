"""Lazy reader for ``--dump-details`` directories: discovery, loading, join.

``rollout_data/{rid}.pt`` holds the full sample batch of one rollout step;
``train_data/{rid}_{rank}.pt`` holds that rank's DP shard with per-token
tensors and ``sample_indices`` mapping each row back to ``Sample.index``.
``load_joined()`` reunites the two: every rollout sample plus (for train
dumps) its per-token training-side row, deduplicated across TP-duplicate
rank files.

Files being written concurrently by a live run are handled in two layers:
``rollout_ids()`` hides files younger than ``MIN_AGE_SECONDS`` unless their
train companion already exists, and a ``torch.load`` failure on a fresh file
raises :class:`DumpStillWriting` (the server maps it to HTTP 503) instead of
surfacing as corruption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from miles.utils.types import Sample


class DumpStillWriting(Exception):
    """A dump file exists but cannot be used yet (``torch.save`` in progress)."""


@dataclass
class RolloutIds:
    train: list[int]
    eval: list[int]


@dataclass
class TrainRow:
    """Per-sample slice of one rank's train dump.

    Required columns fail loudly when absent; columns that legitimately depend
    on run configuration (entropy needs ``--use-rollout-entropy``,
    ``ref_log_probs`` needs a KL term, ...) are ``None`` when not dumped.

    ``raw_reward`` is passed in separately by the caller: unlike every other
    column it is stored batch-global (full batch, rollout order — see the
    "splited at train side" block in ``split_train_data_by_dp``), so indexing
    it by shard row would silently misattribute rewards.
    """

    sample_index: int
    rank: int
    tokens: torch.Tensor
    response_length: int
    total_length: int
    reward: float
    loss_mask: torch.Tensor
    log_probs: torch.Tensor
    rollout_log_probs: torch.Tensor | None
    ref_log_probs: torch.Tensor | None
    entropy: torch.Tensor | None
    ref_entropy: torch.Tensor | None
    advantages: torch.Tensor | None
    returns: torch.Tensor | None
    raw_reward: Any
    truncated: int | None
    weight_versions: list[str] | None

    @classmethod
    def from_columns(cls, columns: dict, row: int, *, rank: int, raw_reward) -> TrainRow:
        def optional(key: str):
            values = columns.get(key)
            return None if values is None else values[row]

        return cls(
            sample_index=columns["sample_indices"][row],
            rank=rank,
            tokens=columns["tokens"][row],
            response_length=columns["response_lengths"][row],
            total_length=columns["total_lengths"][row],
            reward=columns["rewards"][row],
            loss_mask=columns["loss_masks"][row],
            log_probs=columns["log_probs"][row],
            rollout_log_probs=optional("rollout_log_probs"),
            ref_log_probs=optional("ref_log_probs"),
            entropy=optional("entropy"),
            ref_entropy=optional("ref_entropy"),
            advantages=optional("advantages"),
            returns=optional("returns"),
            raw_reward=raw_reward,
            truncated=optional("truncated"),
            weight_versions=optional("weight_versions"),
        )


@dataclass
class JoinedRollout:
    rollout_id: int
    evaluation: bool
    samples: list[Sample]
    train_rows: dict[int, TrainRow]  # keyed by Sample.index; empty for eval dumps

    @property
    def train_coverage(self) -> float:
        return len(self.train_rows) / len(self.samples) if self.samples else 0.0


class DumpReader:
    # A fresh rollout file is only trusted once its train companion exists
    # (written strictly after it) or it has stopped changing for this long.
    MIN_AGE_SECONDS: ClassVar[float] = 10.0
    # torch.load failures on files younger than this are "still being written";
    # on older files they are real corruption and propagate.
    FRESH_SECONDS: ClassVar[float] = 60.0

    def __init__(self, dump_dir: Path | str):
        self.dump_dir = Path(dump_dir)
        self.rollout_dir = self.dump_dir / "rollout_data"
        self.train_dir = self.dump_dir / "train_data"

    def rollout_ids(self) -> RolloutIds:
        ids = RolloutIds(train=[], eval=[])
        if not self.rollout_dir.is_dir():
            return ids
        now = time.time()
        for path in self.rollout_dir.glob("*.pt"):
            evaluation = path.stem.startswith("eval_")
            rollout_id = int(path.stem.removeprefix("eval_"))
            if self._visible(path, rollout_id, evaluation=evaluation, now=now):
                (ids.eval if evaluation else ids.train).append(rollout_id)
        ids.train.sort()
        ids.eval.sort()
        return ids

    def load_joined(self, rollout_id: int, *, evaluation: bool = False) -> JoinedRollout:
        name = f"eval_{rollout_id}.pt" if evaluation else f"{rollout_id}.pt"
        pack = self._torch_load(self.rollout_dir / name)
        assert pack["rollout_id"] == rollout_id, f"{pack['rollout_id']=} != {rollout_id=} in {name}"
        samples = [Sample.from_dict(data) for data in pack["samples"]]
        # raw_reward is stored batch-global, so it is indexed by the sample's
        # position in the rollout dump, not by shard row.
        batch_position = {s.index: i for i, s in enumerate(samples)}

        train_rows: dict[int, TrainRow] = {}
        if not evaluation:
            for path in self._train_paths(rollout_id):
                rank_pack = self._torch_load(path)
                columns = rank_pack["rollout_data"]
                raw_rewards = columns.get("raw_reward")
                if raw_rewards is not None:
                    assert len(raw_rewards) == len(samples), (
                        f"{path}: raw_reward must be batch-global "
                        f"(expected {len(samples)} entries, got {len(raw_rewards)})"
                    )
                for row, sample_index in enumerate(columns["sample_indices"]):
                    assert (
                        sample_index in batch_position
                    ), f"{path} references sample_index {sample_index} absent from the rollout dump"
                    if (existing := train_rows.get(sample_index)) is not None:
                        # TP-duplicate rank carrying the same DP shard.
                        assert existing.response_length == columns["response_lengths"][row], (
                            f"rank {rank_pack['rank']} disagrees with rank {existing.rank} "
                            f"on sample {sample_index} in rollout {rollout_id}"
                        )
                        continue
                    train_rows[sample_index] = TrainRow.from_columns(
                        columns,
                        row,
                        rank=rank_pack["rank"],
                        raw_reward=None if raw_rewards is None else raw_rewards[batch_position[sample_index]],
                    )

        return JoinedRollout(rollout_id=rollout_id, evaluation=evaluation, samples=samples, train_rows=train_rows)

    # ------------------------------- internals ------------------------------

    def _train_paths(self, rollout_id: int) -> list[Path]:
        return sorted(self.train_dir.glob(f"{rollout_id}_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))

    def _visible(self, path: Path, rollout_id: int, *, evaluation: bool, now: float) -> bool:
        if now - path.stat().st_mtime > self.MIN_AGE_SECONDS:
            return True
        return not evaluation and (self.train_dir / f"{rollout_id}_0.pt").exists()

    def _torch_load(self, path: Path):
        try:
            return torch.load(path, weights_only=False, map_location="cpu")
        except FileNotFoundError:
            raise
        except Exception as e:
            if time.time() - path.stat().st_mtime < self.FRESH_SECONDS:
                raise DumpStillWriting(str(path)) from e
            raise
