import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.arguments import parse_args
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.types import Sample


class FakeGenerateState(GenerateState):
    def __init__(self, generate_function: Callable, *, args: Any = None) -> None:
        self.args = args or SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=True,
            sglang_router_policy="round_robin",
        )
        self.generate_fn_semaphore = asyncio.Semaphore(2)
        self.aborted = False
        self.generate_function = generate_function


def make_state(generate_function: Callable, *, args: Any = None) -> FakeGenerateState:
    return FakeGenerateState(generate_function, args=args)


class StampRecordingGenerate:
    def __init__(self) -> None:
        self.stamps: list[str | None] = []

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        self.stamps.append(input.sample.kv_cache_namespace)
        input.sample.status = Sample.Status.COMPLETED
        input.sample.reward = 1.0
        return GenerateFnOutput(samples=input.sample)

    def take_stamps(self) -> set[str | None]:
        stamps, self.stamps = set(self.stamps), []
        return stamps


def make_eval_args(*, namespaced_radix_cache: bool = True) -> dict[str, Any]:
    return dict(
        namespaced_radix_cache=namespaced_radix_cache,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        sglang_router_policy="round_robin",
        eval_datasets=[EvalDatasetConfig(name="fake_ds", path="fake.jsonl", n_samples_per_eval_prompt=2)],
        hf_checkpoint="fake-checkpoint",
        apply_chat_template=False,
        chat_template_path=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        eval_reward_key=None,
        reward_key=None,
    )


def make_eval_prompt_dataset_cache(args: Any) -> dict[Any, Any]:
    [dataset_cfg] = args.eval_datasets
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    return {cache_key: SimpleNamespace(samples=[Sample(prompt="p")])}


def _build_mock_args(extra_argv: list[str] | None = None):
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--ci-test",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "4",
        "--rollout-num-gpus-per-engine",
        "2",
        "--hf-checkpoint",
        "Qwen/Qwen3-0.6B",
        "--prompt-data",
        "/dev/null",
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        "30000",
    ] + (extra_argv or [])
    with patch("sys.argv", argv):
        return parse_args()


@pytest.fixture
def mock_args():
    return _build_mock_args()
