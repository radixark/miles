import copy
import glob
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from miles.utils.data import Dataset
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
# === openevolve_adapted Evolving Gym ===
from openevolve.evolving_gym import SingleTaskEvolvingGym
from miles.rollout.rm_hub.evolving_gym_rm import set_gym as _set_evolving_gym_to_rm
        

class EvolvingGymManager:
    """
    Lightweight wrapper:
    - Synchronously initialize gym (call initialize_sync() to ensure initial program runs once)
    - Generate Sample: wrap gym's {'system', 'user'} prompt with tokenizer.apply_chat_template
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        assert args.evolving_gym_initial_program and args.evolving_gym_evaluator_file, \
            "EvolvingGym need --evolving-gym-initial-program and --evolving-gym-evaluator-file"

        self.gym = SingleTaskEvolvingGym(
            initial_program_path=args.evolving_gym_initial_program,
            evaluation_file=args.evolving_gym_evaluator_file,
            config_path=getattr(args, "evolving_gym_config_path", None),
            config=None,
            max_concurrent_evaluations=getattr(args, "evolving_gym_max_concurrent_evals", 8),
            log_prompts=getattr(args, "evolving_gym_log_prompts", True),
            lazy_output_penalty_level=getattr(args, "evolving_gym_lazy_output_penalty_level", 2),
            database_reinit_ratio=getattr(args, "evolving_gym_database_reinit_ratio", 0.0),
            smallest_restart_step=getattr(args, "evolving_gym_smallest_restart_step", 0),
            largest_restart_step=getattr(args, "evolving_gym_largest_restart_step", None),
            add_historical_programs=getattr(args, "evolving_gym_add_historical_programs", 0),
            reward_process_type=args.evolving_gym_reward_process_type,
            seed=args.evolving_gym_seed,
        )

        # Defer initialization to load() method
        # All initialization logic is in load() method:
        #   - If checkpoint exists, load it
        #   - If no checkpoint, call gym.initialize_sync()
        # This centralizes initialization and prevents duplicate runs
        print(f"[EvolvingGym] Deferring initialization to load() method")

        # Enable recorder if requested
        # Recording initialization is handled by load() method
        if getattr(self.args, "evolving_gym_record", False):
            self.gym.enable_recording(getattr(self.args, "evolving_gym_record_dir", "gym_records"))

        print(f"successfully init EvolvingGymManager")
        

        # Make gym accessible to RM (same process)
        _set_evolving_gym_to_rm(self.gym)


    def get_sample(self) -> Optional[Sample]:
        """
        Get a task from gym (prompt_dict, parent_program), then format as string prompt.
        """
        prompt_dict, parent_program = self.gym.problem_generator()
        system_txt = prompt_dict.get("system") or ""
        user_txt = prompt_dict.get("user") or ""

        # prompt_dict like {'system': ..., 'user': ...}
        apply_chat_template = self.args.apply_chat_template
        tokenizer = self.tokenizer
        max_length = self.args.rollout_max_prompt_len
        tool_key = self.args.tool_key # TODO: double check what's tool key doing here?
        if apply_chat_template:
            if tool_key is not None:
                assert False, "Tool key is not supported for evolving rl"
            tools = None
            messages = []
            if system_txt:
                messages.append({"role": "system", "content": system_txt})
            if not user_txt:
                assert False, "user part should not be empty"
            messages.append({"role": "user", "content": user_txt})

            prompt_str = tokenizer.apply_chat_template(messages, tools, tokenize=False, add_generation_prompt=True)
        else:
            assert False, "We need apply chat template now"
            prompt_str = (system_txt + "\n\n" + user_txt).strip()

        # Optional length check (consistent with global, no filtering by default)
        if max_length is not None:
            assert False, "For now, we don't discard overlong prompts"
            if len(tokenizer(prompt_dict)["input_ids"]) > max_length:
                return None
        
        return Sample(
            prompt=prompt_str,
            label=None,
            metadata={
                "parent_program": parent_program,
                "evolving_gym": True,
            },
        )


def _find_latest_database_checkpoint(load_dir: str) -> Optional[int]:
    """Find the latest database checkpoint rollout_id from save directory."""
    rollout_dir = os.path.join(load_dir, "rollout")
    if not os.path.exists(rollout_dir):
        return None

    db_files = glob.glob(os.path.join(rollout_dir, "evolving_gym_database_*"))
    if not db_files:
        return None

    rollout_ids = []
    for db_file in db_files:
        match = re.search(r'evolving_gym_database_(\d+)', os.path.basename(db_file))
        if match:
            rollout_ids.append(int(match.group(1)))

    return max(rollout_ids) if rollout_ids else None


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
        flags = [
            bool(args.rollout_global_dataset),
            bool(getattr(args, "evolving_gym", False)),
        ]
        assert sum(flags) == 1, (
            "Exactly one of --rollout-global-dataset, --evolving-gym "
            "must be selected."
        )
        
        self.evolving_gym_manager = None

        if args.rollout_global_dataset:
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
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        elif getattr(args, "evolving_gym", False):
            self.dataset = None
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            self.evolving_gym_manager = EvolvingGymManager(args, tokenizer)

        else:
            assert False, "No valid data source."
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
        elif self.evolving_gym_manager is not None:
            prompt_samples = []
            while len(prompt_samples) < num_samples:
                prompt_sample = self.evolving_gym_manager.get_sample()
                if prompt_sample is None:
                    continue
                prompt_samples.append(prompt_sample)
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]
            assert False, "There is no valid data source."

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

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            if self.args.evolving_gym:
                database_path = os.path.join(self.args.save, f"rollout/evolving_gym_database_{rollout_id}")
                os.makedirs(os.path.dirname(database_path), exist_ok=True)
                self.evolving_gym_manager.gym.database.save(database_path, rollout_id)
            else:
                assert False, "None of args.rollout_global_dataset or args.evolving_gym is set."
        else :
            assert not self.args.evolving_gym, "If args.rollout_global_dataset is set, args.evolving_gym must not be set."

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
        print(f"[LOAD] load() called with rollout_id={rollout_id}")
        if self.args.load is None:
            print(f"[LOAD] args.load is None, returning")
            return None

        detected_rollout_id = None

        # Unified handling: auto-detect latest checkpoint when rollout_id == -1
        if rollout_id == -1:
            print(f"[LOAD] rollout_id=-1, attempting auto-detection...")
            if getattr(self.args, "evolving_gym", False):
                detected_id = _find_latest_database_checkpoint(self.args.load)
                if detected_id is not None:
                    rollout_id = detected_id
                    detected_rollout_id = detected_id
                    mode_str = "DEBUG-ROLLOUT-ONLY" if getattr(self.args, "debug_rollout_only", False) else "NORMAL"
                    print(f"[LOAD] ({mode_str}) Auto-detected database checkpoint at rollout_id={rollout_id}")
                else:
                    mode_str = "DEBUG-ROLLOUT-ONLY" if getattr(self.args, "debug_rollout_only", False) else "NORMAL"
                    print(f"[LOAD] ({mode_str}) No database checkpoint found, initializing new database")
                    # Initialize database for first time (both modes: debug-rollout-only and normal)
                    print(f"[LOAD] Calling gym.initialize_sync()...")
                    self.evolving_gym_manager.gym.initialize_sync()
                    # Print initial database state
                    if self.evolving_gym_manager.gym.database.programs:
                        initial_prog = list(self.evolving_gym_manager.gym.database.programs.values())[0]
                        print(f"[LOAD] Initial program metrics: {initial_prog.metrics}")
                    return None
            else:
                print(f"[LOAD] evolving_gym=False, returning None")
                return None

        # If still rollout_id == -1, return None
        if rollout_id == -1:
            return None

        if not self.args.rollout_global_dataset:
            if self.args.evolving_gym:
                # Load evolving gym database
                database_path = os.path.join(self.args.load, f"rollout/evolving_gym_database_{rollout_id}")
                if os.path.exists(database_path):
                    self.evolving_gym_manager.gym.database.load(database_path)
                    # Set initialization flag to avoid duplicate initialization
                    self.evolving_gym_manager.gym._initialized = True
                    print(f"[LOAD] Loaded evolving gym database with {len(self.evolving_gym_manager.gym.database.programs)} programs from rollout_id={rollout_id}")

                    # Debug output: print database distribution (similar to sglang_rollout.py style)
                    if self.evolving_gym_manager.gym.recording_enabled and self.evolving_gym_manager.gym._recorder:
                        self.evolving_gym_manager.gym._recorder.print_database_score_distribution()
                else:
                    # assert False, f"Evolving gym database {database_path} does not exist."
                    print(f"[LOAD] Warning: Evolving gym database {database_path} does not exist, using empty database")
            else:
                assert False, "None of args.rollout_global_dataset or args.evolving_gym is set."
            # return
        else :
            assert not self.args.evolving_gym, "If args.rollout_global_dataset is set, args.evolving_gym must not be set."

        # Load RolloutDataSource base state
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

        return detected_rollout_id


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
