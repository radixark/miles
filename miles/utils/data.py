import json
import logging
import os
import re

import datasets
import numpy as np
import pandas as pd
import ray

from miles.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


# TODO: don't read the whole file into memory.
def read_file(path):
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, dtype={"label": str})
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info(f"read_file path={path} slice {len(df)=} rows into {row_slice=}")
        df = df.iloc[row_slice]

    for _, row in df.iterrows():
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
        self.raw_file_path, self.row_slice = _parse_generalized_path(path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.multimodal_keys = multimodal_keys
        self.label_key = label_key
        self.tool_key = tool_key
        self.metadata_key = metadata_key
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.seed = seed
        self.epoch_id = -1

        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Prompt dataset path '{self.raw_file_path}' does not exist.")

        logger.info(f"Loading dataset from {self.raw_file_path} using Hugging Face datasets.")

        # Determine file type and load using datasets library for memory-mapped access
        if self.raw_file_path.endswith(".jsonl"):
            file_type = "json"
        elif self.raw_file_path.endswith(".parquet"):
            file_type = "parquet"
        else:
            raise ValueError(
                f"Unsupported file format: {self.raw_file_path}. Supported formats are .jsonl and .parquet."
            )

        self.hf_dataset = datasets.load_dataset(file_type, data_files=self.raw_file_path, split="train")

        # Apply row slicing if specified
        if self.row_slice:
            num_rows = len(self.hf_dataset)
            indices = range(num_rows)[self.row_slice]
            self.hf_dataset = self.hf_dataset.select(indices)
            logger.info(f"Applied slice {self.row_slice}, dataset size: {len(self.hf_dataset)}")

        # Apply filtering using the existing helper functions
        def filter_func(example):
            prompt = _build_messages(example, self.prompt_key, self.multimodal_keys)
            return not _should_skip_prompt(
                prompt, self.tokenizer, self.processor, self.max_length, self.apply_chat_template_kwargs
            )

        original_size = len(self.hf_dataset)
        self.hf_dataset = self.hf_dataset.filter(filter_func, num_proc=os.cpu_count())
        new_size = len(self.hf_dataset)
        logger.info(f"Filtered dataset from {original_size} to {new_size} samples.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # The underlying HF dataset handles lazy fetching
        data = self.hf_dataset[idx]

        # Process the data using existing logic
        prompt = _build_messages(data, self.prompt_key, self.multimodal_keys)

        metadata = data.get(self.metadata_key) or {}
        if self.tool_key is not None and self.tool_key in data:
            tools = data[self.tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            elif isinstance(tools, np.ndarray):
                tools = tools.tolist()
            assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
            metadata["tools"] = tools

        sample = Sample(
            prompt=prompt,
            label=data.get(self.label_key) if self.label_key is not None else None,
            metadata=metadata,
        )

        return sample

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        logger.info(f"Shuffling dataset for epoch {new_epoch_id} with seed {self.seed + new_epoch_id}")
        self.hf_dataset = self.hf_dataset.shuffle(seed=self.seed + new_epoch_id)
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
