from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
)
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.types import Sample


def expected_sample(*, group_index: int | None) -> Sample:
    return Sample(
        group_index=group_index,
        index=0,
        prompt="What is 1+7?",
        tokens=[3838, 374, 220, 16, 10, 22, 30, 59, 79075, 90, 23, 92],
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response="\\boxed{8}",
        response_length=5,
        label="8",
        reward=1,
        loss_mask=None,
        weight_versions=[],
        rollout_log_probs=[-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125],
        rollout_routed_experts=None,
        remove_sample=False,
        status=Sample.Status.COMPLETED,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(
            spec_accept_token_num=0, spec_draft_token_num=0, spec_verify_ct=0, completion_token_num=0
        ),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=0, total_prompt_tokens=7),
    )


MODULAR_ROLLOUT_BASE_ARGV = [
    "--rollout-function-path",
    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
]

MIXED_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "9"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "7"},
]


def integration_env_config(
    extra_argv: list[str],
    data_rows: list[dict] | None = None,
    latency: float = 0.0,
    variant: str = "single_turn",
):
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant) + extra_argv,
        data_rows=data_rows,
        latency=latency,
    )


def load_and_call_rollout(args, data_source, mode: str = "train") -> RolloutFnOutput:
    function_path = args.rollout_function_path if mode == "train" else args.eval_function_path
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        function_path,
    )
    if mode == "train":
        return call_rollout_function(fn, RolloutFnTrainInput(rollout_id=0))
    else:
        return call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))


def load_and_call_train(args, data_source):
    return load_and_call_rollout(args, data_source, mode="train")


def filter_by_reward(args, samples, **kwargs):
    reward = samples[0].reward if not isinstance(samples[0], list) else samples[0][0].reward
    if reward == 1:
        return DynamicFilterOutput(keep=True)
    return DynamicFilterOutput(keep=False, reason="reward_zero")
