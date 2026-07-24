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

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import torch

from miles.utils.types import Sample


class DumpStillWriting(Exception):
    """A dump file exists but cannot be used yet (``torch.save`` in progress)."""


# metric columns of DumpReader.step_aggregates(), i.e. the dump-derived L0
# series (consumed by the server's metric catalog as "dump/<column>")
STEP_AGGREGATE_METRICS = (
    "reward_mean",
    "reward_std",
    "response_length_mean",
    "truncated_frac",
    "zero_std_group_frac",
    "mean_abs_lp_diff",
    "mean_entropy",
    "mixed_version_frac",
)


def _min_numeric_version(versions: list[str] | None) -> int | None:
    numeric = [int(v) for v in versions or [] if str(v).isdigit()]
    return min(numeric) if numeric else None


def _tool_call_count(sample: Sample) -> int | None:
    """Tool messages in a chat-style prompt; None for plain-string prompts
    (single-turn math runs have no message structure to count)."""
    if not isinstance(sample.prompt, list):
        return None
    return sum(1 for message in sample.prompt if isinstance(message, dict) and message.get("role") == "tool")


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
    log_probs: torch.Tensor | None  # absent when the run does not dump them
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
            log_probs=optional("log_probs"),
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

    # bump to invalidate summary parquet caches when their columns change
    SUMMARY_VERSION: ClassVar[int] = 2  # v2: staleness/turns/tool columns

    def __init__(self, dump_dir: Path | str, *, cache_dir: Path | str | None = None, tensor_lru: int = 2):
        self.dump_dir = Path(dump_dir)
        self.rollout_dir = self.dump_dir / "rollout_data"
        self.train_dir = self.dump_dir / "train_data"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.dump_dir / "dashboard" / "cache"
        self.tensor_lru = tensor_lru
        self._joined_cache: OrderedDict[tuple[int, bool], JoinedRollout] = OrderedDict()
        # token-view point reads: mmap'd train shards + {sample -> (shard, row)}
        self._shard_cache: OrderedDict[int, tuple[list[dict], dict[int, tuple[int, int]]]] = OrderedDict()
        self._trajectory_cache: OrderedDict[tuple[int, bool], dict[int, dict]] = OrderedDict()
        self._tokenizer = None
        self._tokenizer_loaded = False

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

    def joined(self, rollout_id: int, *, evaluation: bool = False) -> JoinedRollout:
        """LRU-cached :meth:`load_joined`. A completed rollout's dumps never
        change, so entries stay valid; the LRU (``tensor_lru`` ids resident)
        bounds memory since one id holds every per-token tensor of its step."""
        key = (rollout_id, evaluation)
        if key in self._joined_cache:
            self._joined_cache.move_to_end(key)
            return self._joined_cache[key]
        result = self.load_joined(rollout_id, evaluation=evaluation)
        self._joined_cache[key] = result
        while len(self._joined_cache) > self.tensor_lru:
            self._joined_cache.popitem(last=False)
        return result

    @property
    def tokenizer(self):
        """Tokenizer persisted by the run's data source, or None if absent."""
        if not self._tokenizer_loaded:
            self._tokenizer_loaded = True
            tokenizer_dir = self.dump_dir / "tokenizer"
            if tokenizer_dir.is_dir():
                from miles.utils.processing_utils import load_tokenizer

                self._tokenizer = load_tokenizer(str(tokenizer_dir))
        return self._tokenizer

    # ------------------------------- L1 views -------------------------------

    def summary(self, rollout_id: int, *, evaluation: bool = False) -> pl.DataFrame:
        """Per-sample summary table (one row per Sample), parquet-cached under
        ``cache_dir`` and invalidated on source mtime or SUMMARY_VERSION change."""
        stem = f"rollout_{'eval_' if evaluation else ''}{rollout_id}"
        cache_path = self.cache_dir / f"{stem}.parquet"
        sources_path = self.cache_dir / f"{stem}.sources.json"
        sources = self._source_stamps(rollout_id, evaluation=evaluation)
        if cache_path.exists() and sources_path.exists() and json.loads(sources_path.read_text()) == sources:
            return pl.read_parquet(cache_path)

        joined = self.joined(rollout_id, evaluation=evaluation)
        df = pl.DataFrame([self._summary_row(s, joined.train_rows.get(s.index)) for s in joined.samples], strict=False)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        sources_path.write_text(json.dumps(sources))
        return df

    def groups(self, rollout_id: int, *, evaluation: bool = False) -> pl.DataFrame:
        """Per-GRPO-group aggregates; ``zero_std`` flags degenerate groups
        (all samples got the same reward, so advantages vanish)."""
        reward_column = "reward" if evaluation else "raw_reward"
        return (
            self.summary(rollout_id, evaluation=evaluation)
            .group_by("group_index")
            .agg(
                n=pl.len(),
                reward_mean=pl.col(reward_column).mean(),
                reward_std=pl.col(reward_column).std(),
                response_length_mean=pl.col("response_length").mean(),
                truncated_frac=pl.col("truncated").cast(pl.Float64).mean(),
            )
            .with_columns(zero_std=pl.col("reward_std").fill_null(0.0) <= 1e-12)
            .sort("group_index")
        )

    def step_aggregates(self) -> pl.DataFrame:
        """Dump-derived per-step series: the L0 fallback when no metrics.jsonl
        exists. First call computes (and parquet-caches) every step's summary."""
        rows = []
        for rollout_id in self.rollout_ids().train:
            df = self.summary(rollout_id)
            groups = self.groups(rollout_id)
            rows.append(
                dict(
                    rollout_id=rollout_id,
                    n_samples=df.height,
                    reward_mean=df["raw_reward"].mean(),
                    reward_std=df["raw_reward"].std(),
                    response_length_mean=df["response_length"].mean(),
                    truncated_frac=df["truncated"].cast(pl.Float64).mean(),
                    zero_std_group_frac=groups["zero_std"].cast(pl.Float64).mean(),
                    mean_abs_lp_diff=df["mean_abs_lp_diff"].mean(),
                    mean_entropy=df["mean_entropy"].mean(),
                    mixed_version_frac=df["mixed_version"].cast(pl.Float64).mean(),
                )
            )
        return pl.DataFrame(rows, strict=False)

    def trajectory_messages(self, rollout_id: int, sample_index: int, *, evaluation: bool = False) -> dict:
        """Sidecar row for one sample; missing file or sample raises (-> 404),
        which is how the frontend learns the run recorded no conversation."""
        key = (rollout_id, evaluation)
        if key not in self._trajectory_cache:
            name = f"eval_{rollout_id}.jsonl" if evaluation else f"{rollout_id}.jsonl"
            with open(self.dump_dir / "trajectory" / name) as f:
                rows = {row["sample_index"]: row for row in map(json.loads, f)}
            self._trajectory_cache[key] = rows
            while len(self._trajectory_cache) > 4:
                self._trajectory_cache.popitem(last=False)
        self._trajectory_cache.move_to_end(key)
        rows = self._trajectory_cache[key]
        if sample_index not in rows:
            raise KeyError(f"sample {sample_index} has no recorded conversation in rollout {rollout_id}")
        return rows[sample_index]

    # -------------------------- token-view point reads ----------------------

    # rollout-side per-token columns; the parquet mirror is written by
    # save_dashboard_columns at dump time and lazily rebuilt here for runs
    # that predate it (a schema mismatch also triggers the rebuild)
    ROLLOUT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "sample_index",
        "response_length",
        "total_length",
        "tokens",
        "loss_mask",
        "rollout_log_probs",
    )

    def _rollout_columns(self, rollout_id: int, sample_index: int, *, evaluation: bool) -> dict:
        stem = ("eval_" if evaluation else "") + str(rollout_id)
        path = self.dump_dir / "dashboard_columns" / f"rollout_{stem}.parquet"
        if not path.exists() or set(pl.read_parquet_schema(path)) != set(self.ROLLOUT_COLUMNS):
            from miles.ray.rollout.debug_data import save_dashboard_columns

            name = f"eval_{rollout_id}.pt" if evaluation else f"{rollout_id}.pt"
            pack = self._torch_load(self.rollout_dir / name)
            save_dashboard_columns([Sample.from_dict(data) for data in pack["samples"]], path)
        frame = pl.scan_parquet(path).filter(pl.col("sample_index") == sample_index).collect()
        if not len(frame):
            raise KeyError(f"unknown sample_index {sample_index} in rollout {rollout_id}")
        return frame.row(0, named=True)

    def _train_row_lazy(self, rollout_id: int, sample_index: int) -> TrainRow | None:
        """One sample's train columns via mmap'd shards: opening a shard reads
        only its pickle graph; slicing a row faults in ~contiguous KBs."""
        if rollout_id not in self._shard_cache:
            handles: list[dict] = []
            index: dict[int, tuple[int, int]] = {}
            for shard_no, path in enumerate(self._train_paths(rollout_id)):
                columns = self._torch_load(path, mmap=True)["rollout_data"]
                for row_no, si in enumerate(columns["sample_indices"]):
                    index[int(si)] = (shard_no, row_no)
                handles.append(columns)
            self._shard_cache[rollout_id] = (handles, index)
            while len(self._shard_cache) > 4:
                self._shard_cache.popitem(last=False)
        self._shard_cache.move_to_end(rollout_id)
        handles, index = self._shard_cache[rollout_id]
        location = index.get(sample_index)
        if location is None:
            return None
        shard_no, row_no = location
        return TrainRow.from_columns(handles[shard_no], row_no, rank=shard_no, raw_reward=None)

    # ------------------------------- L2 view --------------------------------

    def tokens(
        self, rollout_id: int, sample_index: int, *, start: int = 0, end: int | None = None, evaluation: bool = False
    ) -> dict:
        """Per-token payload for one sample over token positions [start, end).

        Token ids/text cover the whole requested slice; per-token stat arrays
        cover only its overlap with the response region (stat ``i`` maps to
        token position ``prompt_len + a + i``). ``response_offset`` is the
        index within the returned token slice where the response begins.
        """
        columns = self._rollout_columns(rollout_id, sample_index, evaluation=evaluation)
        row = None if evaluation else self._train_row_lazy(rollout_id, sample_index)

        total = columns["total_length"]
        prompt_len = total - columns["response_length"]
        start = max(0, start)
        end = total if end is None else min(end, total)
        if start >= end:
            raise ValueError(f"empty token range [{start}, {end}) for total_len={total}")
        a = max(0, start - prompt_len)
        b = max(0, end - prompt_len)

        def response_slice(values) -> list[float] | None:
            return None if values is None else [float(v) for v in values[a:b]]

        token_ids = [int(t) for t in columns["tokens"][start:end]]
        lp_diff = (
            row.log_probs - row.rollout_log_probs
            if row is not None and row.log_probs is not None and row.rollout_log_probs is not None
            else None
        )
        return dict(
            rollout_id=rollout_id,
            sample_index=sample_index,
            evaluation=evaluation,
            total_len=total,
            prompt_len=prompt_len,
            start=start,
            end=end,
            response_offset=min(len(token_ids), max(0, prompt_len - start)),
            token_ids=token_ids,
            token_text=self._decode_tokens(token_ids),
            rollout_log_probs=(
                response_slice(columns["rollout_log_probs"])
                if columns["rollout_log_probs"] is not None
                else response_slice(row.rollout_log_probs) if row is not None else None
            ),
            loss_mask=None if row is None else [int(v) for v in row.loss_mask[a:b]],
            train_log_probs=None if row is None or row.log_probs is None else response_slice(row.log_probs),
            ref_log_probs=None if row is None else response_slice(row.ref_log_probs),
            lp_diff=response_slice(lp_diff),
            imp_ratio=None if lp_diff is None else response_slice(lp_diff.exp()),
            entropy=None if row is None else response_slice(row.entropy),
            ref_entropy=None if row is None else response_slice(row.ref_entropy),
            advantages=None if row is None else response_slice(row.advantages),
            returns=None if row is None else response_slice(row.returns),
        )

    # ------------------------------- internals ------------------------------

    def _decode_tokens(self, token_ids: list[int]) -> list[str] | None:
        if self.tokenizer is None:
            return None
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]

    def _source_stamps(self, rollout_id: int, *, evaluation: bool) -> dict:
        rollout_path = self.rollout_dir / (f"eval_{rollout_id}.pt" if evaluation else f"{rollout_id}.pt")
        paths = [rollout_path] + ([] if evaluation else self._train_paths(rollout_id))
        return {"_summary_version": self.SUMMARY_VERSION, **{p.name: p.stat().st_mtime for p in paths}}

    def _summary_row(self, sample: Sample, row: TrainRow | None) -> dict:
        spec = sample.spec_info
        cache_info = sample.prefix_cache_info
        entry = dict(
            sample_index=sample.index,
            group_index=sample.group_index,
            status=sample.status.value,
            remove_sample=sample.remove_sample,
            response_length=sample.response_length,
            total_length=len(sample.tokens),
            reward=float(sample.reward) if isinstance(sample.reward, (int, float)) else None,
            weight_version=sample.weight_versions[-1] if sample.weight_versions else None,
            weight_version_min=_min_numeric_version(sample.weight_versions),
            mixed_version=len(set(sample.weight_versions)) > 1 if sample.weight_versions else None,
            turns=len(sample.weight_versions) if sample.weight_versions else None,
            tool_calls=_tool_call_count(sample),
            non_generation_time=sample.non_generation_time,
            spec_accept_rate=(
                spec.spec_accept_token_num / spec.spec_draft_token_num if spec.spec_draft_token_num else None
            ),
            prefix_cache_hit_rate=(
                cache_info.cached_tokens / cache_info.total_prompt_tokens if cache_info.total_prompt_tokens else None
            ),
        )
        if row is None:
            train_columns = dict.fromkeys(
                (
                    "raw_reward",
                    "normalized_reward",
                    "dumped_rank",
                    "mean_entropy",
                    "max_entropy",
                    "ref_entropy_mean",
                    "mean_abs_lp_diff",
                    "max_abs_lp_diff",
                    "mean_imp_ratio",
                    "adv_mean",
                    "adv_std",
                    "return_mean",
                )
            )
            train_columns["truncated"] = sample.status == Sample.Status.TRUNCATED
            return entry | train_columns

        mask = row.loss_mask > 0
        lp_diff = (
            None if row.log_probs is None or row.rollout_log_probs is None else row.log_probs - row.rollout_log_probs
        )
        entropy = _masked(row.entropy, mask)
        abs_diff = _masked(None if lp_diff is None else lp_diff.abs(), mask)
        advantages = _masked(row.advantages, mask)
        return entry | dict(
            raw_reward=None if row.raw_reward is None else float(row.raw_reward),
            normalized_reward=float(row.reward),
            truncated=bool(row.truncated) if row.truncated is not None else sample.status == Sample.Status.TRUNCATED,
            dumped_rank=row.rank,
            mean_entropy=_mean(entropy),
            max_entropy=_max(entropy),
            ref_entropy_mean=_mean(_masked(row.ref_entropy, mask)),
            mean_abs_lp_diff=_mean(abs_diff),
            max_abs_lp_diff=_max(abs_diff),
            mean_imp_ratio=_mean(_masked(None if lp_diff is None else lp_diff.exp(), mask)),
            adv_mean=_mean(advantages),
            adv_std=_std(advantages),
            return_mean=_mean(_masked(row.returns, mask)),
        )

    def _train_paths(self, rollout_id: int) -> list[Path]:
        return sorted(self.train_dir.glob(f"{rollout_id}_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))

    def _visible(self, path: Path, rollout_id: int, *, evaluation: bool, now: float) -> bool:
        if now - path.stat().st_mtime > self.MIN_AGE_SECONDS:
            return True
        return not evaluation and (self.train_dir / f"{rollout_id}_0.pt").exists()

    def _torch_load(self, path: Path, *, mmap: bool = False):
        try:
            return torch.load(path, weights_only=False, map_location="cpu", mmap=mmap)
        except FileNotFoundError:
            raise
        except Exception as e:
            if time.time() - path.stat().st_mtime < self.FRESH_SECONDS:
                raise DumpStillWriting(str(path)) from e
            raise


# ---------------------- masked per-token statistics -------------------------


def _masked(values: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    """Loss-masked positions (tool outputs, removed samples) are excluded from
    all summary statistics; an empty selection yields None, not NaN."""
    if values is None:
        return None
    selected = values[mask]
    return selected.float() if selected.numel() else None


def _mean(values: torch.Tensor | None) -> float | None:
    return None if values is None else float(values.mean())


def _max(values: torch.Tensor | None) -> float | None:
    return None if values is None else float(values.max())


def _std(values: torch.Tensor | None) -> float | None:
    return None if values is None else float(values.std())
