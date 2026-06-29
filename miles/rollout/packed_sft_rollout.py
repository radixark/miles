import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class PackedSFTRecord:
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    metadata: dict[str, Any]


class PackedSFTDataset:
    def __init__(self, args):
        self.args = args
        self.seed = args.rollout_seed
        self.epoch_id = -1
        self.origin_samples = [
            self._record_to_sample(row_id, row) for row_id, row in enumerate(_read_jsonl(args.prompt_data))
        ]
        if not self.origin_samples:
            raise ValueError(f"Packed SFT dataset is empty: {args.prompt_data}")
        self.samples = self.origin_samples

    def _record_to_sample(self, row_id: int, row: dict[str, Any]) -> Sample:
        record = _parse_record(
            row,
            row_id=row_id,
            tokens_key=self.args.packed_sft_tokens_key,
            loss_mask_key=self.args.packed_sft_loss_mask_key,
            response_length_key=self.args.packed_sft_response_length_key,
            metadata_key=self.args.metadata_key,
            max_length=self.args.rollout_max_context_len,
            allow_empty_loss=self.args.packed_sft_allow_empty_loss,
        )
        return Sample(
            group_index=row_id,
            index=row_id,
            prompt="",
            tokens=record.tokens,
            response_length=record.response_length,
            reward=0,
            loss_mask=record.loss_mask,
            status=Sample.Status.COMPLETED,
            metadata=record.metadata,
        )

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)


class PackedSFTDataSource:
    """Data source for already-tokenized packed SFT blocks.

    Each JSONL row is one training sample. The row must contain token ids and a
    loss mask. The mask may be full-length and sparse, which is required when a
    block contains multiple packed conversations, or a tail response-window mask
    when an explicit response_length is supplied.
    """

    def __init__(self, args):
        self.args = args
        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata = {}
        self.dataset = PackedSFTDataset(args)
        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if self.sample_offset + num_samples <= len(self.dataset):
            prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            prompt_samples = self.dataset.samples[self.sample_offset :]
            num_samples -= len(prompt_samples)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            prompt_samples += self.dataset.samples[:num_samples]
            self.sample_offset = num_samples

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = _clone_packed_sample(
                    prompt_sample,
                    group_index=self.sample_group_index,
                    index=self.sample_index,
                )
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id: int) -> None:
        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import torch

        torch.save(state_dict, path)

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        import torch

        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    groups = data_buffer.get_samples(args.rollout_batch_size)
    flat = [sample for group in groups for sample in group]
    total_tokens = sum(len(sample.tokens) for sample in flat)
    loss_tokens = sum(sum(sample.loss_mask or []) for sample in flat)
    logger.info(
        "packed_sft_rollout::generate_rollout rollout_id=%s samples=%s total_tokens=%s loss_tokens=%s",
        rollout_id,
        len(flat),
        total_tokens,
        loss_tokens,
    )
    metrics = {
        "packed_sft/total_tokens": total_tokens,
        "packed_sft/loss_tokens": loss_tokens,
        "packed_sft/avg_tokens_per_sample": total_tokens / max(len(flat), 1),
        "packed_sft/avg_loss_tokens_per_sample": loss_tokens / max(len(flat), 1),
    }
    try:
        from miles.rollout.base_types import RolloutFnTrainOutput
    except ModuleNotFoundError:
        return groups
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


def _add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--packed-sft-tokens-key", type=str, default="tokens")
    parser.add_argument("--packed-sft-loss-mask-key", type=str, default="loss_mask")
    parser.add_argument("--packed-sft-response-length-key", type=str, default="response_length")
    parser.add_argument(
        "--packed-sft-allow-empty-loss",
        action="store_true",
        default=False,
        help="Allow packed blocks whose loss mask has no positive token. Disabled by default to catch bad packing.",
    )
    return parser


generate_rollout.add_arguments = _add_arguments


def _parse_record(
    row: dict[str, Any],
    *,
    row_id: int,
    tokens_key: str,
    loss_mask_key: str,
    response_length_key: str,
    metadata_key: str,
    max_length: int,
    allow_empty_loss: bool,
) -> PackedSFTRecord:
    tokens = _parse_int_list(row.get(tokens_key), key=tokens_key, row_id=row_id)
    if not tokens:
        raise ValueError(f"row {row_id}: {tokens_key} must not be empty")
    if len(tokens) > max_length:
        raise ValueError(f"row {row_id}: packed token length {len(tokens)} exceeds max_length={max_length}")

    loss_mask = _parse_int_list(row.get(loss_mask_key), key=loss_mask_key, row_id=row_id)
    if any(x not in (0, 1) for x in loss_mask):
        raise ValueError(f"row {row_id}: {loss_mask_key} must contain only 0/1 values")
    if not allow_empty_loss and 1 not in loss_mask:
        raise ValueError(f"row {row_id}: {loss_mask_key} contains no trainable token")

    response_length = row.get(response_length_key)
    if response_length is None:
        if len(loss_mask) != len(tokens):
            raise ValueError(
                f"row {row_id}: full-length {loss_mask_key} must match tokens length "
                f"when {response_length_key} is absent "
                f"({len(loss_mask)} != {len(tokens)})"
            )
        if loss_mask[0]:
            raise ValueError(
                f"row {row_id}: first trainable token is at position 0; packed SFT blocks need at least "
                "one prefix token before the first loss token"
            )
        response_length = len(tokens) - 1
        loss_mask = loss_mask[1:]
    else:
        response_length = _parse_non_negative_int(response_length, key=response_length_key, row_id=row_id)
        if response_length > len(tokens):
            raise ValueError(
                f"row {row_id}: {response_length_key}={response_length} exceeds tokens length={len(tokens)}"
            )
        if response_length == len(tokens):
            raise ValueError(
                f"row {row_id}: {response_length_key} equals tokens length; packed SFT blocks need at least "
                "one prefix token before the response window"
            )
        elif len(loss_mask) == len(tokens):
            prefix_mask = loss_mask[:-response_length] if response_length > 0 else loss_mask
            if any(prefix_mask):
                raise ValueError(
                    f"row {row_id}: full-length {loss_mask_key} has trainable tokens outside the explicit "
                    f"{response_length_key} tail window"
                )
            loss_mask = loss_mask[-response_length:] if response_length > 0 else []
        elif len(loss_mask) != response_length:
            raise ValueError(
                f"row {row_id}: {loss_mask_key} length must equal tokens length or {response_length_key} "
                f"({len(loss_mask)} not in {{{len(tokens)}, {response_length}}})"
            )

    if len(loss_mask) != response_length:
        raise ValueError(
            f"row {row_id}: normalized {loss_mask_key} length {len(loss_mask)} "
            f"!= {response_length_key} {response_length}"
        )
    if not allow_empty_loss and 1 not in loss_mask:
        raise ValueError(f"row {row_id}: normalized {loss_mask_key} contains no trainable token")

    metadata = row.get(metadata_key) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"row {row_id}: {metadata_key} must be an object when provided")
    metadata = dict(metadata)
    metadata.setdefault("packed_sft_row_id", row_id)
    metadata["packed_sft_total_length"] = len(tokens)
    metadata["packed_sft_loss_tokens"] = sum(loss_mask)

    return PackedSFTRecord(tokens=tokens, response_length=response_length, loss_mask=loss_mask, metadata=metadata)


def _parse_int_list(value: Any, *, key: str, row_id: int) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row_id}: {key} is not valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"row {row_id}: {key} must be a list of integers")
    if not all(isinstance(x, int) for x in value):
        raise ValueError(f"row {row_id}: {key} must be a list of integers")
    return value


def _parse_non_negative_int(value: Any, *, key: str, row_id: int) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"row {row_id}: {key} must be a non-negative integer")
    return value


def _read_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Packed SFT dataset path '{path}' does not exist.")
    if not path.endswith(".jsonl"):
        raise ValueError(f"Packed SFT dataset path '{path}' must be a .jsonl file.")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode error at line {line_num} in {path}: {exc}") from exc


def _clone_packed_sample(sample: Sample, *, group_index: int, index: int) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        prompt=sample.prompt,
        tokens=list(sample.tokens),
        response=sample.response,
        response_length=sample.response_length,
        reward=sample.reward,
        loss_mask=list(sample.loss_mask) if sample.loss_mask is not None else None,
        status=sample.status,
        metadata=dict(sample.metadata),
    )
