import copy
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from miles.utils.data import Dataset, custom_prompt_preprocessor
from miles.utils.misc import load_function
from miles.utils.types import Sample

# RLVE imports - these will be available when Gym is in PYTHONPATH
try:
    from Gym.environment import VerifiableEnvironment
    from Gym.environments import identifier2environment
    from Gym.parameter_controller import ParameterController
    from Gym.parameter_controllers import identifier2controller
    RLVE_AVAILABLE = True
except ImportError:
    RLVE_AVAILABLE = False

logger = logging.getLogger(__name__)


class RLVEManager:
    """
    Manages RLVE (Reinforcement Learning with Verifiable Environments) training.
    Handles procedural problem generation and adaptive difficulty scheduling.
    """
    def __init__(self, args, tokenizer):
        if not RLVE_AVAILABLE:
            raise ImportError(
                "RLVE Gym not available. Ensure Gym package is in PYTHONPATH. "
                "Example: export PYTHONPATH=/path/to/RLVE:$PYTHONPATH"
            )

        self.args = args
        self.tokenizer = tokenizer

        assert args.environment_list, "Environment list is not set."

        self.environment2difficulty = {
            environment: args.initial_difficulty
            for environment in args.environment_list
        }
        self.environment2accuracy = {
            environment: dict(accuracy=0, num_samples=0)
            for environment in args.environment_list
        }
        self.problem_generation_seed = 0

    def generate_problem(self) -> Tuple[str, int, Optional["VerifiableEnvironment"]]:
        """
        Generate a problem from a randomly selected environment at appropriate difficulty.
        Returns: (environment_name, problem_difficulty, problem_instance or None)
        """
        environment: str = random.choice(self.args.environment_list)

        parameter_controller: "ParameterController" = identifier2controller[environment]()
        maximum_difficulty: int = self.environment2difficulty[environment]
        parameter_lists: List[Tuple[int, List[Dict]]] = []

        for problem_difficulty in range(maximum_difficulty + 1):
            if problem_difficulty > maximum_difficulty - self.args.difficulty_sliding_window_size:
                parameter_lists.append(
                    (problem_difficulty, copy.deepcopy(parameter_controller.get_parameter_list()))
                )
            parameter_controller.update()

        problem_difficulty, parameter_list = random.choice(parameter_lists)
        parameter: Dict = random.choice(parameter_list)
        problem: "VerifiableEnvironment" = identifier2environment[environment]()

        if problem.generator(seed=self.problem_generation_seed, parameter=parameter):
            generated_problem = problem
        else:
            generated_problem = None
            logger.warning(
                f"Generating problem for environment {environment} failed\n"
                f"parameter: {parameter}"
            )
        self.problem_generation_seed += 1

        return environment, problem_difficulty, generated_problem

    def get_sample(self) -> Optional[Sample]:
        """
        Generate a sample with prompt and metadata for training.
        Returns None if problem generation fails.
        """
        environment, problem_difficulty, problem = self.generate_problem()
        if problem is None:
            return None

        user_prompt = problem.prompt_generator()
        prompt = custom_prompt_preprocessor(
            args=self.args,
            user_prompt=user_prompt,
            apply_chat_template=self.args.apply_chat_template
        )

        apply_chat_template = self.args.apply_chat_template
        tokenizer = self.tokenizer
        max_length = getattr(self.args, 'rollout_max_prompt_len', None)
        tool_key = getattr(self.args, 'tool_key', None)

        if apply_chat_template:
            if tool_key is not None:
                raise NotImplementedError("Tool key is not supported for RLVE yet.")
            tools = None
            prompt = tokenizer.apply_chat_template(
                prompt, tools, tokenize=False, add_generation_prompt=True
            )

        # Note: max_length check disabled as per RLVE original code
        # if max_length is not None:
        #     if len(tokenizer(prompt)["input_ids"]) > max_length:
        #         return None

        return Sample(
            prompt=prompt,
            label=None,
            metadata=dict(
                environment=environment,
                problem_difficulty=problem_difficulty,
                config=problem.get_config()
            ),
        )

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return dict(
            environment2difficulty=self.environment2difficulty,
            environment2accuracy=self.environment2accuracy,
            problem_generation_seed=self.problem_generation_seed,
        )

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.environment2difficulty = state["environment2difficulty"]
        self.environment2accuracy = state["environment2accuracy"]
        self.problem_generation_seed = state["problem_generation_seed"]

    def update(self, samples: List[Sample]) -> Dict[str, Any]:
        """
        Update accuracy statistics based on completed samples.
        Also update the difficulty when necessary (adaptive curriculum).
        This should be called after rewards have been computed.

        Returns: Dictionary of metrics to log
        """
        log_dict = {}

        for sample in samples:
            environment = sample.metadata["environment"]
            problem_difficulty = sample.metadata["problem_difficulty"]
            maximum_difficulty = self.environment2difficulty[environment]

            assert problem_difficulty <= maximum_difficulty, (
                "The difficulty of the sample is higher than the current difficulty "
                "of the problem, which should not happen."
            )

            # Only track accuracy at the current maximum difficulty
            if problem_difficulty < maximum_difficulty:
                continue

            self.environment2accuracy[environment]["num_samples"] += 1
            self.environment2accuracy[environment]["accuracy"] += sample.reward["accuracy"]

        log_dict["rollout/problem_generation_seed"] = self.problem_generation_seed

        # Check if we should increase difficulty for each environment
        for environment in self.args.environment_list:
            num_samples = self.environment2accuracy[environment]["num_samples"]
            accuracy_sum = self.environment2accuracy[environment]["accuracy"]

            if num_samples >= self.args.min_prompts_before_difficulty_check * self.args.n_samples_per_prompt:
                accuracy = accuracy_sum / num_samples
                log_dict[f"RLVE/{environment}/accuracy"] = accuracy

                if accuracy >= self.args.min_metric_to_increase_difficulty:
                    self.environment2difficulty[environment] += 1
                    log_dict[f"RLVE/{environment}/difficulty"] = self.environment2difficulty[environment]

                # Reset counters after difficulty check
                self.environment2accuracy[environment] = dict(accuracy=0, num_samples=0)

        return log_dict


# TODO may further refactor data-loading part later
class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        # RLVE manager for procedural generation
        self.rlve_manager = None

        # Validate data source configuration
        has_global_dataset = getattr(args, 'rollout_global_dataset', False)
        has_rlve = getattr(args, 'rlve', False)

        if has_rlve and has_global_dataset:
            raise ValueError(
                "args.rollout_global_dataset and args.rlve are mutually exclusive. "
                "Set only one of them."
            )

        if has_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=getattr(args, 'apply_chat_template_kwargs', None),
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        elif has_rlve:
            self.dataset = None
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            self.rlve_manager = RLVEManager(args, tokenizer)
        else:
            # No data source - will generate empty samples
            self.dataset = None

    def get_samples(self, num_samples):
        samples = []

        if self.dataset is not None:
            # Standard dataset-based sampling
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

        elif self.rlve_manager is not None:
            # RLVE procedural generation
            while len(samples) < num_samples:
                prompt_sample = self.rlve_manager.get_sample()
                if prompt_sample is None:
                    continue

                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                self.sample_group_index += 1
                samples.append(group)
            assert len(samples) == num_samples

        else:
            # Fallback: empty samples
            for _ in range(num_samples):
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = Sample(index=self.sample_index)
                    self.sample_index += 1
                    group.append(sample)
                self.sample_group_index += 1
                samples.append(group)

        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        # Save RLVE state if using RLVE
        if self.rlve_manager is not None:
            state_dict = self.rlve_manager.get_state()
            path = os.path.join(self.args.save, f"rollout/rlve_manager_state_dict_{rollout_id}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)
            logger.info(f"Saved RLVE state to {path}")

        # Save dataset state if using global dataset
        if self.args.rollout_global_dataset:
            state_dict = {
                "sample_offset": self.sample_offset,
                "epoch_id": self.epoch_id,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "metadata": self.metadata,
            }
            path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return
        if rollout_id == -1:
            return

        # Load RLVE state if using RLVE
        if self.rlve_manager is not None:
            path = os.path.join(self.args.load, f"rollout/rlve_manager_state_dict_{rollout_id}.pt")
            if os.path.exists(path):
                state_dict = torch.load(path)
                self.rlve_manager.set_state(state_dict)
                logger.info(f"Loaded RLVE state from {path}")
            else:
                logger.warning(f"RLVE checkpoint {path} does not exist.")

        # Load dataset state if using global dataset
        if self.args.rollout_global_dataset:
            path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
            if not os.path.exists(path):
                logger.info(f"Checkpoint {path} does not exist.")
                return

            logger.info(f"load metadata from {path}")
            logger.info(f"load metadata: {self.metadata}")
            state_dict = torch.load(path)
            self.sample_offset = state_dict.get("sample_offset", 0)
            self.epoch_id = state_dict.get("epoch_id", 0)
            self.sample_group_index = state_dict.get("sample_group_index", 0)
            self.sample_index = state_dict.get("sample_index", 0)
            self.metadata = state_dict.get("metadata", {})

            if self.args.rollout_global_dataset and self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
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


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
