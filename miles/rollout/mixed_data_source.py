"""Deterministic prompt mixing with independent dataset epochs and exact source resume."""

import copy
import hashlib
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from miles.rollout.data_source import DataSource, create_sample_groups, load_rollout_processing
from miles.utils.data import Dataset, _parse_generalized_path
from miles.utils.rollout_data_config import RolloutDatasetConfig
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_CHECKPOINT_VERSION = 1


@dataclass
class _DatasetCursor:
    dataset: Dataset
    shuffle: bool
    offset: int = 0
    epoch: int = 0

    def take(self, count: int) -> list[Sample]:
        prompts = []
        while count:
            if self.offset == len(self.dataset):
                self.offset = 0
                self.epoch += 1
                if self.shuffle:
                    self.dataset.shuffle(self.epoch)
            end = min(self.offset + count, len(self.dataset))
            prompts.extend(self.dataset.samples[self.offset : end])
            count -= end - self.offset
            self.offset = end
        return prompts


class MixedRolloutDataSource(DataSource):
    """Weight prompt draws, then shuffle and replicate them into homogeneous GRPO groups.

    Ratios describe draws, including oversampling requests. Downstream filtering,
    retries and async completion order can change the accepted training-batch ratio.
    """

    def __init__(self, args):
        self.args = args
        self.configs: tuple[RolloutDatasetConfig, ...] = args.rollout_datasets
        self._hashes = {config.name: _dataset_hash(config, warn_missing_pin=True) for config in self.configs}
        tokenizer, processor = load_rollout_processing(args)
        self._cursors = [
            _make_cursor(config, args=args, tokenizer=tokenizer, processor=processor, seed=args.rollout_seed + index)
            for index, config in enumerate(self.configs)
        ]
        self._weights = [config.samples_per_step for config in self.configs]
        self._total_weight = sum(self._weights)
        self._credits = [0] * len(self.configs)
        self._rng = random.Random(args.rollout_seed)
        self.sample_group_index = 0
        self.sample_index = 0
        self._signature = _source_signature(args, self.configs)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("num_samples must be a non-negative integer")
        counts = self._allocate_counts(num_samples)
        prompts = [
            prompt for cursor, count in zip(self._cursors, counts, strict=True) for prompt in cursor.take(count)
        ]
        # Mix selected prompts even when per-dataset epoch shuffling is disabled.
        self._rng.shuffle(prompts)
        groups = create_sample_groups(
            prompts,
            n_samples_per_prompt=self.args.n_samples_per_prompt,
            start_group_index=self.sample_group_index,
            start_sample_index=self.sample_index,
        )
        self.sample_group_index += len(groups)
        self.sample_index += len(groups) * self.args.n_samples_per_prompt
        return groups

    def _allocate_counts(self, count: int) -> list[int]:
        # Smooth weighted round robin carries integer quota debt across requests.
        # Counting individual draws keeps allocation independent of request boundaries.
        counts = [0] * len(self.configs)
        for _ in range(count):
            self._credits = [credit + weight for credit, weight in zip(self._credits, self._weights, strict=True)]
            selected = max(range(len(counts)), key=self._credits.__getitem__)
            counts[selected] += 1
            self._credits[selected] -= self._total_weight
        return counts

    def save(self, directory: Path) -> None:
        save_simple_checkpoint(
            directory=directory,
            data={
                "format_version": _CHECKPOINT_VERSION,
                "signature": self._signature,
                "sha256": self._hashes,
                "datasets": [
                    {"offset": cursor.offset, "epoch": cursor.epoch, "size": len(cursor.dataset)}
                    for cursor in self._cursors
                ],
                "credits": self._credits,
                "rng_state": self._rng.getstate(),
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
            },
        )

    def load(self, directory: Path) -> None:
        state = load_simple_checkpoint(directory=directory)
        if state.get("format_version") != _CHECKPOINT_VERSION:
            raise ValueError("Unsupported mixed rollout data checkpoint version")
        if state["signature"] != self._signature:
            raise ValueError("Mixed rollout data configuration changed since the checkpoint")
        for config in self.configs:
            actual = _dataset_hash(config, warn_missing_pin=False)
            expected = state["sha256"].get(config.name)
            if actual != expected or actual != self._hashes[config.name]:
                raise ValueError(
                    f"Dataset {config.name!r} changed on resume: expected SHA-256 {expected}, got {actual}"
                )
        _validate_checkpoint(state, cursors=self._cursors, n_samples_per_prompt=self.args.n_samples_per_prompt)
        # Validate everything, including RNG state, before changing live source state.
        restored_rng = random.Random()
        restored_rng.setstate(state["rng_state"])
        for cursor, saved in zip(self._cursors, state["datasets"], strict=True):
            cursor.offset = saved["offset"]
            cursor.epoch = saved["epoch"]
            if cursor.shuffle:
                cursor.dataset.shuffle(cursor.epoch)
        self._credits = list(state["credits"])
        self._rng = restored_rng
        self.sample_group_index = state["sample_group_index"]
        self.sample_index = state["sample_index"]


def _make_cursor(config: RolloutDatasetConfig, *, args, tokenizer, processor, seed: int) -> _DatasetCursor:
    dataset = Dataset(
        config.path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=config.max_prompt_len,
        prompt_key=config.input_key,
        multimodal_keys=args.multimodal_keys,
        label_key=config.label_key,
        metadata_key=config.metadata_key,
        tool_key=config.tool_key,
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        seed=seed,
    )
    if not len(dataset):
        raise ValueError(f"Dataset {config.name!r} has no prompts after loading and prompt-length filtering")
    if config.max_prompt_len is not None and any(not isinstance(sample.prompt, str) for sample in dataset.samples):
        raise ValueError(
            f"Dataset {config.name!r}: prompt-length filtering of conversations requires --apply-chat-template"
        )
    for sample in dataset.origin_samples:
        if not isinstance(sample.metadata, dict):
            raise ValueError(f"Dataset {config.name!r}: row metadata must be a mapping")
        # The configured source name owns this tag, even if a row supplies another value.
        sample.metadata = {**sample.metadata, "dataset_name": config.name}
    if args.rollout_shuffle:
        dataset.shuffle(0)
    return _DatasetCursor(dataset=dataset, shuffle=args.rollout_shuffle)


def _dataset_hash(config: RolloutDatasetConfig, *, warn_missing_pin: bool) -> str:
    path, _ = _parse_generalized_path(config.path)
    if not Path(path).is_file():
        raise ValueError(f"Dataset {config.name!r}: path must name a regular JSONL or parquet file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if config.sha256 is not None and config.sha256 != actual:
        raise ValueError(f"Dataset {config.name!r}: SHA-256 mismatch: expected {config.sha256}, got {actual}")
    if config.sha256 is None and warn_missing_pin:
        logger.warning("Dataset %r has no sha256 pin; computed sha256: %s", config.name, actual)
    return actual


def _source_signature(args, configs: tuple[RolloutDatasetConfig, ...]) -> dict:
    return {
        # Adding a matching pin does not alter the sample sequence; actual hashes are saved separately.
        "datasets": [{key: value for key, value in asdict(config).items() if key != "sha256"} for config in configs],
        "preprocessing": copy.deepcopy(
            {
                key: getattr(args, key)
                for key in (
                    "hf_checkpoint",
                    "chat_template_path",
                    "apply_chat_template",
                    "apply_chat_template_kwargs",
                    "multimodal_keys",
                    "rollout_seed",
                    "rollout_shuffle",
                    "n_samples_per_prompt",
                )
            }
        ),
    }


def _validate_checkpoint(state: dict, *, cursors: list[_DatasetCursor], n_samples_per_prompt: int) -> None:
    saved_datasets = state["datasets"]
    credits = state["credits"]
    if len(saved_datasets) != len(cursors) or len(credits) != len(cursors):
        raise ValueError("Mixed rollout checkpoint dataset count does not match")
    for cursor, saved in zip(cursors, saved_datasets, strict=True):
        if saved["size"] != len(cursor.dataset):
            raise ValueError("Mixed rollout checkpoint dataset size changed")
        if type(saved["offset"]) is not int or not 0 <= saved["offset"] <= len(cursor.dataset):
            raise ValueError("Invalid mixed rollout checkpoint dataset offset")
        if type(saved["epoch"]) is not int or saved["epoch"] < 0:
            raise ValueError("Invalid mixed rollout checkpoint dataset epoch")
    if any(type(credit) is not int for credit in credits) or sum(credits) != 0:
        raise ValueError("Invalid mixed rollout checkpoint allocation credits")
    group_index, sample_index = state["sample_group_index"], state["sample_index"]
    if type(group_index) is not int or group_index < 0 or type(sample_index) is not int:
        raise ValueError("Invalid mixed rollout checkpoint sample identity")
    if sample_index != group_index * n_samples_per_prompt:
        raise ValueError("Mixed rollout checkpoint sample and group identities do not match")
