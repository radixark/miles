import json
import logging
import os
import random
import re
from itertools import islice

import numpy as np
import pyarrow.parquet as pq
import ray

from miles.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def _read_jsonl_lazy(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _read_parquet_lazy(path):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            yield row


# TODO: don't read the whole file into memory.
def read_file(path):
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):
        reader = _read_jsonl_lazy(path)
    elif path.endswith(".parquet"):
        reader = _read_parquet_lazy(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info(f"read_file path={path} slice rows with {row_slice=}")
        reader = islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    return reader


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
        self.epoch_id = -1
        self._cache = {}
        self._pq_file = None

        self._file_type = self._get_file_type()
        self._build_index()

        self._shuffled_indices = list(range(len(self.valid_indices)))
        logger.info(f"Created a dataset with {len(self)} samples.")

    def _get_file_type(self):
        if self.raw_file_path.endswith(".jsonl"):
            return "jsonl"
        if self.raw_file_path.endswith(".parquet"):
            return "parquet"
        raise ValueError(f"Unsupported file format: {self.raw_file_path}")

    def _get_prompt_from_data(self, data):
        if self.multimodal_keys:
            prompt_content = []
            if self.prompt_key in data:
                prompt_content.append({"type": "text", "text": data[self.prompt_key]})
            for media_type, data_key in self.multimodal_keys.items():
                if data_key in data:
                    media_path = data[data_key]
                    prompt_content.append({"type": media_type, "path": media_path})
        else:
            prompt_content = data.get(self.prompt_key)

        if self.apply_chat_template:
            if self.tool_key is not None:
                tools = data[self.tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
            else:
                tools = None
            template_input = [{"role": "user", "content": prompt_content}] if self.multimodal_keys else prompt_content
            prompt = self.tokenizer.apply_chat_template(
                template_input,
                tools,
                tokenize=False,
                add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            )
        else:
            prompt = prompt_content
        return prompt

    def _should_include(self, data):
        prompt = self._get_prompt_from_data(data)
        if self.max_length is not None:
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not self.multimodal_keys:
                if len(raw_prompt_ids) > self.max_length:
                    return False
        return True

    def _build_index(self):
        logger.info(f"Building index for {self.path_with_slice}...")
        self.valid_indices = [] # Stores original_idx of valid samples
        
        if self._file_type == "jsonl":
            self.jsonl_offset_map = {} # Maps original_idx -> file_offset
            current_original_line_idx = 0
            with open(self.raw_file_path, "r", encoding="utf-8") as f:
                while True:
                    current_offset = f.tell()
                    line = f.readline()
                    if not line: break # EOF

                    # Apply row_slice filtering during index building
                    if self.row_slice is not None:
                        start = self.row_slice.start if self.row_slice.start is not None else 0
                        stop = self.row_slice.stop if self.row_slice.stop is not None else float('inf')
                        if not (start <= current_original_line_idx < stop):
                            current_original_line_idx += 1
                            continue # Skip this line if it's outside the slice
                    
                    data = json.loads(line)
                    if self._should_include(data):
                        self.valid_indices.append(current_original_line_idx)
                        self.jsonl_offset_map[current_original_line_idx] = current_offset
                    current_original_line_idx += 1

        elif self._file_type == "parquet":
            self._pq_file = pq.ParquetFile(self.raw_file_path)
            
            all_raw_parquet_indices = range(self._pq_file.metadata.num_rows) 

            if self.row_slice is not None:
                start = self.row_slice.start if self.row_slice.start is not None else 0
                stop = self.row_slice.stop if self.row_slice.stop is not None else self._pq_file.metadata.num_rows
                row_indices_for_pass = [i for i in all_raw_parquet_indices if start <= i < stop]
            else:
                row_indices_for_pass = list(all_raw_parquet_indices)

            for i in row_indices_for_pass:
                data = self._read_parquet_row(i)
                if self._should_include(data):
                    self.valid_indices.append(i)
        
        logger.info(f"Found {len(self.valid_indices)} valid samples in {self.path_with_slice}")

    def _read_parquet_row(self, idx):
        if self._pq_file is None: # Should be initialized in _build_index
            self._pq_file = pq.ParquetFile(self.raw_file_path)

        row_offset = 0
        for i in range(self._pq_file.num_row_groups):
            rg_rows = self._pq_file.metadata.row_group(i).num_rows
            if idx < row_offset + rg_rows:
                table = self._pq_file.read_row_group(i)
                row_in_group = idx - row_offset
                return {col.name: table.column(col.name)[row_in_group].as_py() for col in table.schema}
            row_offset += rg_rows
        raise IndexError(f"Index {idx} out of range for parquet file {self.raw_file_path}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # idx is an index into the shuffled self.samples list (0 to len(valid_indices)-1)
        shuffled_idx = self._shuffled_indices[idx] 
        # original_idx is the index of the sample in the full raw file 
        original_idx = self.valid_indices[shuffled_idx]

        if original_idx in self._cache:
            return self._cache[original_idx]

        data = None
        if self._file_type == "jsonl":
            offset = self.jsonl_offset_map[original_idx]
            with open(self.raw_file_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                data = json.loads(f.readline())
        elif self._file_type == "parquet":
            data = self._read_parquet_row(original_idx)
        else:
            raise ValueError(f"Unsupported file type: {self._file_type}")

        prompt = self._get_prompt_from_data(data)
        sample = Sample(
            prompt=prompt,
            label=data.get(self.label_key) if self.label_key is not None else None,
            metadata=data.get(self.metadata_key) or {},
        )

        # Basic cache management 
        if len(self._cache) >= 10000: # Limit cache size
            # Remove oldest item 
            del self._cache[next(iter(self._cache))]
        self._cache[original_idx] = sample
        
        return sample

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        random.shuffle(self._shuffled_indices)
        self.epoch_id = new_epoch_id
        self._cache.clear() # Clear cache on shuffle to avoid stale data


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
