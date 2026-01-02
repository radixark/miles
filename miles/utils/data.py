import itertools
import json
import logging
import os
import re
from functools import partial

import datasets
import numpy as np
import ray

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from miles.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)

_FILE_TYPE_MAP = {
    ".jsonl": "json",
    ".parquet": "parquet",
}


def _filter_func(example, tokenizer, processor, max_length, prompt_key, multimodal_keys, apply_chat_template, apply_chat_template_kwargs, tool_key):
    as_conversation = apply_chat_template
    prompt = _build_messages(example, prompt_key, as_conversation, multimodal_keys)

    tools = None
    if tool_key is not None and tool_key in example:
        tools = example[tool_key]
        if isinstance(tools, str):
            tools = json.loads(tools)
        elif isinstance(tools, np.ndarray):
            tools = tools.tolist()
        assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"

    if apply_chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
    else:
        formatted_prompt = prompt

    if processor:
        from miles.utils.processing_utils import process_vision_info
        multimodal_inputs = process_vision_info(prompt, processor)
    else:
        multimodal_inputs = None

    return not _should_skip_prompt(formatted_prompt, tokenizer, processor, max_length, multimodal_inputs)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _should_skip_prompt(formatted_prompt: str, tokenizer, processor, max_length, multimodal_inputs=None):
    if max_length is None:
        return False

    if processor:
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        input_ids = processor_output["input_ids"][0]
    else:
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    return len(input_ids) > max_length


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        assert as_conversation, "as_conversation must be True when multimodal_keys is not None"
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
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

    return prompt


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
        dataset_num_proc=8,
    ):
        # 1. Store basic config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.multimodal_keys = multimodal_keys
        self.label_key = label_key
        self.tool_key = tool_key
        self.metadata_key = metadata_key
        self.apply_chat_template = apply_chat_template
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.seed = seed
        self.epoch_id = -1

        # 2. Load and process dataset
        self.hf_dataset = self._load_and_filter_dataset(path, dataset_num_proc)
        self.origin_hf_dataset = self.hf_dataset

    def _get_file_type(self, path: str) -> str:
        _, ext = os.path.splitext(path)

        try:
            return _FILE_TYPE_MAP[ext]
        except KeyError:
            raise ValueError(f"Unsupported format: {ext}. Supported: {list(_FILE_TYPE_MAP.keys())}") from None

    def _load_and_filter_dataset(self, path, dataset_num_proc):
        raw_file_path, row_slice = _parse_generalized_path(path)

        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"Prompt dataset path '{raw_file_path}' does not exist.")

        logger.info(f"Loading dataset from {raw_file_path} using Hugging Face datasets.")

        # Determine file type and load using datasets library for memory-mapped access
        file_type = self._get_file_type(raw_file_path)
        ds = datasets.load_dataset(file_type, data_files=raw_file_path, split="train")

        # Apply row slicing if specified
        if row_slice:
            num_rows = len(ds)
            indices = range(num_rows)[row_slice]
            ds = ds.select(indices)
            logger.info(f"Applied slice {row_slice}, dataset size: {len(ds)}")

        filter_kwargs = {
            "tokenizer": self.tokenizer,
            "processor": self.processor,
            "max_length": self.max_length,
            "prompt_key": self.prompt_key,
            "multimodal_keys": self.multimodal_keys,
            "apply_chat_template": self.apply_chat_template,
            "apply_chat_template_kwargs": self.apply_chat_template_kwargs,
            "tool_key": self.tool_key,
        }

        original_size = len(ds)

        ds = ds.filter(
            partial(_filter_func, **filter_kwargs), num_proc=dataset_num_proc, desc="Filtering invalid samples"
        )

        new_size = len(ds)
        logger.info(f"Filtered dataset from {original_size} to {new_size} samples.")

        return ds

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # The underlying HF dataset handles lazy fetching
        data = self.hf_dataset[idx]

        # Process the data using existing logic
        as_conversation = self.apply_chat_template
        prompt = _build_messages(data, self.prompt_key, as_conversation, self.multimodal_keys)

        metadata = data.get(self.metadata_key) or {}
        tools = None
        if self.tool_key is not None and self.tool_key in data:
            tools = data[self.tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            # TODO (chenyang): If the JSON parsing is heavy, we might need
            #  to use hf_dataset.map() during init to pre-process these
            #  fields into a more efficient format (Arrow-native), rather
            #  than parsing raw strings on the fly.
            elif isinstance(tools, np.ndarray):
                tools = tools.tolist()
            assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
            metadata["tools"] = tools
        
        if self.apply_chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt, tools=tools, tokenize=False, add_generation_prompt=True, **self.apply_chat_template_kwargs
        )
        else:
            formatted_prompt = prompt

        multimodal_inputs = None
        if self.processor:
            from miles.utils.processing_utils import process_vision_info
            multimodal_inputs = process_vision_info(prompt, self.processor)

        sample = Sample(
            prompt=formatted_prompt,
            label=data.get(self.label_key) if self.label_key is not None else None,
            metadata=metadata,
            multimodal_inputs=multimodal_inputs,
        )

        return sample

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        logger.info(f"Shuffling dataset for epoch {new_epoch_id} with seed {self.seed + new_epoch_id}")
        self.hf_dataset = self.origin_hf_dataset.shuffle(seed=self.seed + new_epoch_id)
        self.epoch_id = new_epoch_id


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
