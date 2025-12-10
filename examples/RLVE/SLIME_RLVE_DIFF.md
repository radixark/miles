# RLVE Modifications to Slime (v0.5.0rc0)

This document captures all changes RLVE made to the base Slime codebase for future reference.
Base: slime @ commit aa33750 (2025-09-12)

## Summary of Changes

### New Files
- `rollout/rm_hub/rlve_rm.py` - RLVE reward model using Gym verifiers
- `rollout/rm_hub/bbeh.py` - BBEH benchmark reward
- `rollout/rm_hub/livecodebench.py` - LiveCodeBench reward
- `backends/utils/` - New utilities directory

### Modified Files
- `ray/rollout_data_source.py` - Major changes (RLVEManager class)
- `utils/arguments.py` - Added RLVE arguments
- `utils/data.py` - Added custom_prompt_preprocessor
- `rollout/rm_hub/__init__.py` - Added RLVE reward branch

---

## File: `ray/rollout_data_source.py`

### Added RLVEManager Class (Lines 12-137)

```python
import random
from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from Gym.parameter_controller import ParameterController
from Gym.parameter_controllers import identifier2controller
from typing import List, Optional, Tuple, Dict, Any

from slime.utils.data import custom_prompt_preprocessor


class RLVEManager:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        assert args.environment_list, "Environment list is not set."

        self.environment2difficulty = {environment: args.initial_difficulty for environment in args.environment_list}
        self.environment2accuracy = {environment: dict(accuracy=0, num_samples=0) for environment in args.environment_list}
        self.problem_generation_seed = 0

    def generate_problem(self) -> Tuple[str, Optional[VerifiableEnvironment]]:
        environment: str = random.choice(self.args.environment_list)

        parameter_controller: ParameterController = identifier2controller[environment]()
        maximum_difficulty: int = self.environment2difficulty[environment]
        parameter_lists: List[List[Dict]] = []
        for problem_difficulty in range(maximum_difficulty + 1):
            if problem_difficulty > maximum_difficulty - self.args.difficulty_sliding_window_size:
                parameter_lists.append((problem_difficulty, copy.deepcopy(parameter_controller.get_parameter_list())))
            parameter_controller.update()

        problem_difficulty, parameter_list = random.choice(parameter_lists)
        parameter: Dict = random.choice(parameter_list)
        problem: VerifiableEnvironment = identifier2environment[environment]()
        if problem.generator(seed=self.problem_generation_seed, parameter=parameter):
            generated_problem = problem
        else:
            generated_problem = None
            print("Generating problem for environment {} failed\nparameter: {}\n\n\n".format(environment, parameter), flush=True)
        self.problem_generation_seed += 1

        return environment, problem_difficulty, generated_problem

    def get_sample(self) -> Optional[Sample]:
        environment, problem_difficulty, problem = self.generate_problem()
        if problem is None:
            return None

        user_prompt = problem.prompt_generator()
        prompt = custom_prompt_preprocessor(args=self.args, user_prompt=user_prompt, apply_chat_template=self.args.apply_chat_template)

        apply_chat_template = self.args.apply_chat_template
        tokenizer = self.tokenizer
        max_length = self.args.rollout_max_prompt_len
        tool_key = self.args.tool_key
        if apply_chat_template:
            if tool_key is not None:
                assert False, "Tool key is not supported for RLVE yet."
            else:
                tools = None
            prompt = tokenizer.apply_chat_template(prompt, tools, tokenize=False, add_generation_prompt=True)

        return Sample(
            prompt=prompt,
            label=None,
            metadata=dict(environment=environment, problem_difficulty=problem_difficulty, config=problem.get_config()),
        )

    def get_state(self) -> Dict[str, Any]:
        return dict(
            environment2difficulty=self.environment2difficulty,
            environment2accuracy=self.environment2accuracy,
            problem_generation_seed=self.problem_generation_seed,
        )

    def set_state(self, state: Dict[str, Any]) -> None:
        self.environment2difficulty = state["environment2difficulty"]
        self.environment2accuracy = state["environment2accuracy"]
        self.problem_generation_seed = state["problem_generation_seed"]

    def update(self, samples: List[Sample]) -> Dict[str, Any]:
        """
        Update accuracy statistics based on completed samples.
        Also update the difficulty when necessary.
        This should be called after rewards have been computed.
        """
        log_dict = {}

        for sample in samples:
            environment = sample.metadata["environment"]

            problem_difficulty, maximum_difficulty = sample.metadata["problem_difficulty"], self.environment2difficulty[environment]
            assert problem_difficulty <= maximum_difficulty
            if problem_difficulty < maximum_difficulty:
                continue
            self.environment2accuracy[environment]["num_samples"] += 1
            self.environment2accuracy[environment]["accuracy"] += sample.reward["accuracy"]

        log_dict["rollout/problem_generation_seed"] = self.problem_generation_seed

        for environment in self.args.environment_list:
            num_samples, accuracy = self.environment2accuracy[environment]["num_samples"], self.environment2accuracy[environment]["accuracy"]
            if num_samples >= self.args.min_prompts_before_difficulty_check * self.args.n_samples_per_prompt:
                accuracy = accuracy / num_samples
                log_dict["RLVE/{}/accuracy".format(environment)] = accuracy

                if accuracy >= self.args.min_metric_to_increase_difficulty:
                    self.environment2difficulty[environment] += 1
                    log_dict["RLVE/{}/difficulty".format(environment)] = self.environment2difficulty[environment]

                self.environment2accuracy[environment] = dict(accuracy=0, num_samples=0)

        return log_dict
```

### Modified RolloutDataSource.__init__

Added mutual exclusivity check and RLVE manager initialization:

```python
# In __init__:
assert (args.rollout_global_dataset and not args.rlve) or (not args.rollout_global_dataset and args.rlve), \
    "Incompatible arguments: args.rollout_global_dataset and args.rlve must be mutually exclusive"
self.rlve_manager = None

# ... existing dataset code ...

elif args.rlve:
    self.dataset = None
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    self.rlve_manager = RLVEManager(args, tokenizer)
```

### Modified get_samples()

Added RLVE sample generation path:

```python
elif self.rlve_manager is not None:
    while len(samples) < num_samples:
        prompt_sample = self.rlve_manager.get_sample()
        if prompt_sample is None:
            continue
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = self.sample_index
            self.sample_index += 1
            group.append(sample)
        samples.append(group)
    assert len(samples) == num_samples
```

### Modified save()

Added RLVE state saving:

```python
def save(self, rollout_id):
    if not self.args.rollout_global_dataset:
        if self.args.rlve:
            state_dict = self.rlve_manager.get_state()
            path = os.path.join(self.args.save, f"rollout/rlve_manager_state_dict_{rollout_id}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)
```

---

## File: `utils/arguments.py`

### Added add_custom_arguments Function

```python
def add_custom_arguments(parser):
    parser.add_argument(
        "--rlve",
        action="store_true",
        default=False,
        help="Enable RLVE training.",
    )
    parser.add_argument(
        "--environment-list",
        type=str,
        nargs='+',
        default=None,
        help="List of verifiable environments to train on.",
    )
    parser.add_argument(
        "--initial-difficulty",
        type=int,
        default=0,
        help="Initial difficulty (upper bound) for each environment.",
    )
    parser.add_argument(
        "--difficulty-sliding-window-size",
        type=int,
        default=4,
        help="Size of the sliding window for problem difficulty.",
    )
    parser.add_argument(
        "--min-metric-to-increase-difficulty",
        type=float,
        default=0.9,
        help="Threshold to increase difficulty.",
    )
    parser.add_argument(
        "--min-prompts-before-difficulty-check",
        type=int,
        default=8,
        help="Minimum prompts before difficulty check.",
    )
    parser.add_argument(
        "--answer-marker-type",
        type=str,
        default=r"\boxed{}",
        help="The type of answer marker to use.",
    )
    parser.add_argument(
        "--custom-prompt-preprocessor",
        type=str,
        required=True,
        choices=("TinyZero", "ChatTemplate_NoSystemPrompt"),
        help="Choose a custom prompt preprocessor.",
    )
    return parser
```

---

## File: `utils/data.py`

### Added TinyZero Template and custom_prompt_preprocessor

```python
TinyZero_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Show your work in <think> </think> tags, and return the final answer in <answer> </answer> tags.
User: {prompt}
Assistant: Let me solve this step by step.
<think>"""

def custom_prompt_preprocessor(args, user_prompt: str, apply_chat_template: bool) -> Union[str, List[Dict[str, str]]]:
    if args.custom_prompt_preprocessor == "TinyZero":
        assert not apply_chat_template
        return TinyZero_TEMPLATE.format(prompt=user_prompt)
    elif args.custom_prompt_preprocessor == "ChatTemplate_NoSystemPrompt":
        assert apply_chat_template
        if isinstance(user_prompt, list) and len(user_prompt) > 0 and isinstance(user_prompt[0], dict) and "role" in user_prompt[0]:
            return user_prompt
        return [{"role": "user", "content": user_prompt}]
    else:
        raise NotImplementedError(f"User prompt processor {args.custom_prompt_preprocessor} not implemented.")
```

### Modified Dataset.__init__

Added args parameter and custom_prompt_preprocessor call:

```python
def __init__(
    self,
    path,
    tokenizer,
    max_length,
    # ... other params ...
    apply_chat_template=False,
    args=None,  # NEW
):
    self.args = args  # NEW

    self.origin_samples = []
    for data in read_file(path):
        prompt = data[prompt_key]
        prompt = custom_prompt_preprocessor(args=self.args, user_prompt=prompt, apply_chat_template=apply_chat_template)  # NEW
        # ... rest of init ...
```

---

## File: `rollout/rm_hub/__init__.py`

### Added RLVE Import and Branch

```python
from .rlve_rm import rlve_rm
import json

# In async_rm function:
elif rm_type == "rlve":
    metadata = sample.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return rlve_rm(args=args, environment=metadata["environment"], config=metadata["config"], response=response)
```

---

## File: `rollout/rm_hub/rlve_rm.py` (NEW)

```python
from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from typing import Dict, Any


def rlve_rm(args, environment: str, config: Dict, response: str) -> Dict[str, Any]:
    if args.answer_marker_type == r"\boxed{}":
        answer_markers = (r"\boxed{", r"}")
        assert args.custom_prompt_preprocessor in ("ChatTemplate_NoSystemPrompt",)
    elif args.answer_marker_type == r"<answer></answer>":
        answer_markers = (r"<answer>", r"</answer>")
        assert args.custom_prompt_preprocessor in ("TinyZero",)
    else:
        raise NotImplementedError(f"Answer marker type {args.answer_marker_type} not implemented.")

    problem: VerifiableEnvironment = identifier2environment[environment](answer_markers=answer_markers)
    problem.set_config(config)
    return problem.verifier(response)
```

---

## Key Design Decisions

1. **Mutual Exclusivity**: RLVE and rollout_global_dataset are mutually exclusive data sources
2. **Adaptive Difficulty**: Per-environment difficulty tracking with sliding window
3. **Prompt Preprocessing**: Two modes - TinyZero (raw template) and ChatTemplate_NoSystemPrompt
4. **Answer Markers**: Support for `\boxed{}` and `<answer></answer>` formats
5. **State Persistence**: RLVE manager state saved/loaded for checkpoint resume

## Environment Dependencies

- `Gym.environment.VerifiableEnvironment`
- `Gym.environments.identifier2environment`
- `Gym.parameter_controller.ParameterController`
- `Gym.parameter_controllers.identifier2controller`

These must be in PYTHONPATH (from RLVE-GYM).
