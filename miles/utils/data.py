import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import ray

from miles.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path, chunk_size: int = 10000):
    """Read a dataset file and yield rows as dictionaries.

    This function supports memory-efficient streaming for large files by using
    chunked reading for JSONL files and iterative row group reading for Parquet files.

    Args:
        path: Path to the dataset file. Supports .jsonl and .parquet formats.
              Can include optional slice notation: "path.jsonl@[start:end]"
        chunk_size: Number of rows to read at a time for JSONL files. Ignored for
                   Parquet files which use row groups. Default: 10000.

    Yields:
        dict: Each row as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):
        yield from _read_jsonl_file(path, row_slice, chunk_size)
    elif path.endswith(".parquet"):
        yield from _read_parquet_file(path, row_slice)
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")


def _read_jsonl_file(path: str, row_slice: slice | None, chunk_size: int):
    """Read JSONL file with memory-efficient chunked reading.

    Args:
        path: Path to the JSONL file.
        row_slice: Optional slice to select specific rows.
        chunk_size: Number of rows to read per chunk.

    Yields:
        dict: Each row as a dictionary.
    """
    if row_slice is not None:
        # For sliced reads, we need to track row indices
        # Load only the required slice for efficiency
        start = row_slice.start or 0
        stop = row_slice.stop

        current_idx = 0
        for chunk in pd.read_json(path, lines=True, dtype={"label": str}, chunksize=chunk_size):
            chunk_end = current_idx + len(chunk)

            # Skip chunks entirely before the slice start
            if stop is not None and current_idx >= stop:
                break
            if chunk_end <= start:
                current_idx = chunk_end
                continue

            # Calculate which rows in this chunk to yield
            chunk_start_offset = max(0, start - current_idx)
            chunk_end_offset = len(chunk) if stop is None else min(len(chunk), stop - current_idx)

            for idx in range(chunk_start_offset, chunk_end_offset):
                yield chunk.iloc[idx].to_dict()

            current_idx = chunk_end
    else:
        # No slice: stream through all chunks
        for chunk in pd.read_json(path, lines=True, dtype={"label": str}, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                yield row.to_dict()


def _read_parquet_file(path: str, row_slice: slice | None):
    """Read Parquet file with memory-efficient row group iteration.

    Uses PyArrow to read row groups iteratively instead of loading the entire
    file into memory at once.

    Args:
        path: Path to the Parquet file.
        row_slice: Optional slice to select specific rows.

    Yields:
        dict: Each row as a dictionary.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)

    if row_slice is not None:
        start = row_slice.start or 0
        stop = row_slice.stop

        current_idx = 0
        for i in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.read_row_group(i)
            chunk = row_group.to_pandas(types_mapper=pd.ArrowDtype)
            chunk_end = current_idx + len(chunk)

            # Skip row groups entirely before the slice start
            if stop is not None and current_idx >= stop:
                break
            if chunk_end <= start:
                current_idx = chunk_end
                continue

            # Calculate which rows in this row group to yield
            chunk_start_offset = max(0, start - current_idx)
            chunk_end_offset = len(chunk) if stop is None else min(len(chunk), stop - current_idx)

            for idx in range(chunk_start_offset, chunk_end_offset):
                yield chunk.iloc[idx].to_dict()

            current_idx = chunk_end
    else:
        # No slice: stream through all row groups
        for i in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.read_row_group(i)
            chunk = row_group.to_pandas(types_mapper=pd.ArrowDtype)
            for _, row in chunk.iterrows():
                yield row.to_dict()


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
    if max_length is None:
        return False

    from miles.utils.processing_utils import prepare_model_inputs

    input_ids, _ = prepare_model_inputs(prompt, tokenizer, processor, None, apply_chat_template_kwargs)
    return len(input_ids) > max_length


def _build_messages(data: dict, prompt_key: str, multimodal_keys: dict = None):
    messages = data.get(prompt_key)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in messages:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return messages


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        self.origin_samples = []
        for data in read_file(path):
            prompt = _build_messages(data, prompt_key, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            # TODO: this is slow.
            if _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
                continue

            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                )
            )

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
