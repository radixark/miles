import abc
import copy
import hashlib
import logging
import os
import random
from pathlib import Path

from miles.utils.data import Dataset, _parse_generalized_path
from miles.utils.function_registry import load_function
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.rollout_data_config import RolloutDatasetConfig, load_rollout_data_config, lower_limit
from miles.utils.simple_checkpointer import has_simple_checkpoint, load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import RewardSpec, Sample

logger = logging.getLogger(__name__)

# a prime stride keeps (rollout_seed, request number) pairs from mapping to the same in-request shuffle seed
_MIX_SEED_STRIDE = 1_000_003


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def save(self, directory: Path) -> None:
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, directory: Path) -> None:
        """
        Load the state of the data source
        """

    def get_buffer_length(self) -> int | None:
        """Pending-sample backlog, or None for sources without a buffer."""
        return None


def _load_prompt_tokenizer_and_processor(args):
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

    # TODO move (during the refactor)
    if (d := args.dump_details) is not None:
        tokenizer.save_pretrained(Path(d) / "tokenizer")
        # Bespoke processors (e.g. Inkling's) are not ProcessorMixin and cannot serialise.
        if hasattr(
            processor, "save_pretrained"
        ):  # config-access-exempt: custom processors may omit serialization support
            processor.save_pretrained(Path(d) / "processor")
    return tokenizer, processor


def _load_prompt_dataset(args, path: str, *, tokenizer, processor, max_length: int | None) -> Dataset:
    return Dataset(
        path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        prompt_key=args.input_key,
        multimodal_keys=args.multimodal_keys,
        label_key=args.label_key,
        metadata_key=args.metadata_key,
        tool_key=args.tool_key,
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        seed=args.rollout_seed,
    )


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer, processor = _load_prompt_tokenizer_and_processor(args)
            self.dataset = _load_prompt_dataset(
                args,
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
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
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def save(self, directory: Path) -> None:
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        save_simple_checkpoint(directory=directory, data=state_dict)

    def load(self, directory: Path) -> None:
        if not self.args.rollout_global_dataset:
            logger.warning("--disable-rollout-global-dataset: the dataset starts where a fresh run's would")
            return

        state_dict = load_simple_checkpoint(directory=directory)
        if "offsets" in state_dict:
            raise RolloutDataResumeMismatch(
                "the checkpoint holds a --rollout-data-yaml-config mix, so the --prompt-data dataset starts from "
                "its first prompt",
                needs_flag=False,
            )
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def load_sample_ids(self, directory: Path) -> None:
        """Instead of ``load``, for a reset: the dataset starts over, but sample ids keep counting past the run's."""
        state_dict = load_simple_checkpoint(directory=directory)
        self.sample_group_index = state_dict["sample_group_index"]
        self.sample_index = state_dict["sample_index"]


class LegacyRolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)

    def load(self, directory: Path) -> None:
        raise NotImplementedError("LegacyRolloutDataSourceWithBuffer does not support checkpoint loading")


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


def split_by_weight(num_samples: int, weights: list[int], credits: list[int]) -> tuple[list[int], list[int]]:
    """Split `num_samples` prompts across datasets by smooth weighted round-robin.

    For each prompt every dataset gains its weight and the one with the most credit gives the prompt, paying
    back the total weight. `credits` carry over between calls, so the long-run split is exact even when every
    request asks for a single prompt.
    """
    total_weight = sum(weights)
    credits = list(credits)
    counts = [0] * len(weights)
    for _ in range(num_samples):
        credits = [credit + weight for credit, weight in zip(credits, weights, strict=True)]
        chosen = max(range(len(credits)), key=lambda i: (credits[i], -i))
        credits[chosen] -= total_weight
        counts[chosen] += 1
    return counts, credits


class RolloutDataResumeMismatch(ValueError):
    """The checkpoint's data source state cannot be resumed under the current data configuration.

    ``needs_flag`` is False when starting the data over is the only sensible resume, so the executor resets
    without --rollout-data-reset-on-load (e.g. a --rollout-data-yaml-config mix resumed under --prompt-data).
    """

    def __init__(self, message: str, *, needs_flag: bool = True) -> None:
        super().__init__(message)
        self.needs_flag = needs_flag


class MixedRolloutDataSource(DataSource):
    """Draws every request from the `--rollout-data-yaml-config` datasets in proportion to their weights.

    Datasets are mixed at the prompt level, before each prompt is copied `--n-samples-per-prompt` times, so a
    group always holds one prompt. Each dataset keeps its own position, epoch and shuffle.
    """

    def __init__(self, args):
        self.args = args
        self.dataset_configs = load_rollout_data_config(args.rollout_data_yaml_config)
        self.dataset_sha256 = {config.name: _check_dataset_sha256(config) for config in self.dataset_configs}
        for config in self.dataset_configs:
            for function_path in (config.custom_generate_function_path, config.custom_agent_function_path):
                if function_path is not None:
                    load_function(function_path)

        tokenizer, processor = _load_prompt_tokenizer_and_processor(args)
        self.datasets = {}
        for config in self.dataset_configs:
            dataset = _load_prompt_dataset(
                args,
                config.path,
                tokenizer=tokenizer,
                processor=processor,
                max_length=lower_limit(config.max_prompt_len, args.rollout_max_prompt_len),
            )
            if len(dataset) == 0:
                raise ValueError(f"rollout dataset {config.name!r} ({config.path}) has no prompts after filtering")
            # the mixer tags every sample's metadata, so a row whose metadata is not a mapping cannot be mixed
            bad = next((i for i, prompt in enumerate(dataset.samples) if not isinstance(prompt.metadata, dict)), None)
            if bad is not None:
                raise ValueError(
                    f"rollout dataset {config.name!r}: row {bad} has non-mapping metadata "
                    f"({type(dataset.samples[bad].metadata).__name__}); --metadata-key must name a JSON object"
                )
            # the file is reopened to parse it; for a pinned file a rewrite since hashing would train on unpinned bytes
            if config.sha256 is not None and (pinned := self.dataset_sha256[config.name]) is not None:
                path, _ = _parse_generalized_path(config.path)
                if (loaded := _hash_file(path)) != pinned:
                    raise ValueError(
                        f"rollout dataset {config.name!r} ({path}) changed while loading: sha256 {pinned[:8]} "
                        f"at startup, {loaded[:8]} after parsing"
                    )
            self.datasets[config.name] = dataset
            logger.info(f"rollout dataset {config.name!r}: {len(dataset)} prompts, weight {config.samples_per_step}")

        self.offsets = {name: 0 for name in self.datasets}
        self.epoch_ids = {name: 0 for name in self.datasets}
        self.credits = [0] * len(self.dataset_configs)
        self.num_requests = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self._shuffle_datasets()

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        counts, self.credits = split_by_weight(
            num_samples, [config.samples_per_step for config in self.dataset_configs], self.credits
        )
        prompts = [
            (config, prompt)
            for config, count in zip(self.dataset_configs, counts, strict=True)
            for prompt in self._take(config.name, count)
        ]
        random.Random(self.args.rollout_seed * _MIX_SEED_STRIDE + self.num_requests).shuffle(prompts)
        self.num_requests += 1
        return [self._make_group(config, prompt) for config, prompt in prompts]

    def save(self, directory: Path) -> None:
        state_dict = {
            "offsets": self.offsets,
            "epoch_ids": self.epoch_ids,
            "credits": dict(zip(self.datasets, self.credits, strict=True)),
            "num_requests": self.num_requests,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "sha256": self.dataset_sha256,
            "settings": {config.name: _resumed_settings(config) for config in self.dataset_configs},
        }
        save_simple_checkpoint(directory=directory, data=state_dict)

    def load(self, directory: Path) -> None:
        if not has_simple_checkpoint(directory):
            raise RolloutDataResumeMismatch(
                f"the checkpoint holds no data source state (a --disable-rollout-global-dataset run){_RESET_HINT}"
            )
        state_dict = load_simple_checkpoint(directory=directory)
        if "offsets" not in state_dict:
            raise RolloutDataResumeMismatch(
                f"the checkpoint holds a single-dataset (--prompt-data) data source, not a "
                f"--rollout-data-yaml-config mix{_RESET_HINT}"
            )
        if set(state_dict["offsets"]) != set(self.datasets):
            raise RolloutDataResumeMismatch(
                f"the checkpoint mixed datasets {sorted(state_dict['offsets'])}, but --rollout-data-yaml-config now "
                f"lists {sorted(self.datasets)}{_RESET_HINT}"
            )
        # groups drawn before the save carry the settings stamped then, so resuming under changed ones would mix both
        # a checkpoint saved before the settings were recorded has none to compare
        for config in self.dataset_configs if "settings" in state_dict else []:
            saved, now = state_dict["settings"][config.name], _resumed_settings(config)
            if changed := sorted(key for key in saved.keys() | now.keys() if saved.get(key) != now.get(key)):
                raise RolloutDataResumeMismatch(
                    f"rollout dataset {config.name!r} changed {', '.join(changed)} since the checkpoint was saved, "
                    f"but groups drawn before the save still carry the old values{_RESET_HINT}"
                )
        for name, saved in state_dict["sha256"].items():
            if saved != self.dataset_sha256[name]:
                raise RolloutDataResumeMismatch(
                    f"rollout dataset {name!r} changed since the checkpoint was saved: sha256 was {saved}, "
                    f"now {self.dataset_sha256[name]}{_RESET_HINT}"
                )
        # a dataset filtered shorter since the save (e.g. a lower max_prompt_len) cannot resume at its position
        if too_far := {n: o for n, o in state_dict["offsets"].items() if o > len(self.datasets[n])}:
            raise RolloutDataResumeMismatch(
                f"the checkpoint resumes {too_far} past the end of datasets now holding "
                f"{ {n: len(self.datasets[n]) for n in too_far} } prompts; keep the dataset and its filters unchanged{_RESET_HINT}"
            )
        self.offsets = dict(state_dict["offsets"])
        self.epoch_ids = dict(state_dict["epoch_ids"])
        self.credits = [state_dict["credits"][name] for name in self.datasets]
        self.num_requests = state_dict["num_requests"]
        self.sample_group_index = state_dict["sample_group_index"]
        self.sample_index = state_dict["sample_index"]
        self._shuffle_datasets()

    def load_sample_ids(self, directory: Path) -> None:
        """Instead of ``load``, for a reset: every dataset starts over, but sample ids keep counting past the run's.

        Also reads a single-dataset (--prompt-data) checkpoint, which saves the same two counters; a run that saved
        no data source state has no ids to continue from.
        """
        if not has_simple_checkpoint(directory):
            return
        state_dict = load_simple_checkpoint(directory=directory)
        self.sample_group_index = state_dict["sample_group_index"]
        self.sample_index = state_dict["sample_index"]

    def _shuffle_datasets(self) -> None:
        if self.args.rollout_shuffle:
            for name, dataset in self.datasets.items():
                dataset.shuffle(self.epoch_ids[name])

    def _take(self, name: str, count: int) -> list[Sample]:
        dataset = self.datasets[name]
        prompts = []
        while count > 0:
            chunk = dataset.samples[self.offsets[name] : self.offsets[name] + count]
            prompts += chunk
            count -= len(chunk)
            self.offsets[name] += len(chunk)
            if self.offsets[name] >= len(dataset):
                self.offsets[name] = 0
                self.epoch_ids[name] += 1
                if self.args.rollout_shuffle:
                    dataset.shuffle(self.epoch_ids[name])
        return prompts

    def _make_group(self, config: RolloutDatasetConfig, prompt: Sample) -> list[Sample]:
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt)
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            self.sample_index += 1
            sample.metadata["dataset_name"] = config.name
            if config.agent is not None:
                sample.metadata["agent_config"] = copy.deepcopy(config.agent)
            sample.agent_function_path = config.custom_agent_function_path
            sample.max_seq_len = config.max_seq_len
            sample.max_response_len = config.max_response_len
            sample.generate_function_path = config.custom_generate_function_path
            if config.custom_rm_path is not None or config.rm_type is not None:
                sample.reward_spec = RewardSpec(rm_type=config.rm_type, custom_rm_path=config.custom_rm_path)
            group.append(sample)
        self.sample_group_index += 1
        return group


_RESET_HINT = "; pass --rollout-data-reset-on-load to keep the model and optimizer but start every dataset fresh"
# samples_per_step may change on resume; the file itself is checked through sha256
_RESUME_FREE_FIELDS = {"name", "path", "samples_per_step", "sha256"}


def _resumed_settings(config: RolloutDatasetConfig) -> dict:
    return config.model_dump(exclude=_RESUME_FREE_FIELDS, exclude_none=True)


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_dataset_sha256(config: RolloutDatasetConfig) -> str | None:
    """Hash the dataset file; stop when its first 8 hex characters differ from the configured `sha256`.

    Returns the full digest, which the checkpoint keeps for an exact check on resume.
    """
    path, _ = _parse_generalized_path(config.path)
    if not os.path.isfile(path):
        if config.sha256 is not None:
            raise ValueError(
                f"rollout dataset {config.name!r}: sha256 is only supported for a single file, not {path}"
            )
        return None
    actual = _hash_file(path)
    if config.sha256 is None:
        logger.warning(f'rollout dataset {config.name!r} has no sha256; add `sha256: "{actual[:8]}"` to pin it')
    elif config.sha256 != actual[:8]:
        raise ValueError(
            f"rollout dataset {config.name!r} ({path}) does not match its sha256: expected {config.sha256}, "
            f"got {actual[:8]}"
        )
    return actual
