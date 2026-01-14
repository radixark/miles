import asyncio


from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.modular_rollout.orchestration_eval import SimpleEvalRolloutFn
from miles.rollout.modular_rollout.orchestration_train import SimpleTrainRolloutFn
from miles.utils.types import Sample


def test_simple_train_rollout_fn_integration(rollout_integration_env):
    args, data_source = rollout_integration_env
    fn = SimpleTrainRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    out = asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))

    assert len(out.samples) == args.rollout_batch_size
    group = out.samples[0]
    assert len(group) == args.n_samples_per_prompt
    sample = group[0]
    assert sample == Sample(
        group_index=0,
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


def test_simple_eval_rollout_fn_integration(rollout_integration_env):
    args, data_source = rollout_integration_env
    fn = SimpleEvalRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    out = asyncio.run(fn(RolloutFnEvalInput(rollout_id=0)))

    assert "toy" in out.data
    rewards = out.data["toy"]["rewards"]
    samples = out.data["toy"]["samples"]
    assert len(rewards) == len(samples) == args.n_samples_per_eval_prompt
    assert rewards[0] == 1
    assert samples[0] == Sample(
        group_index=None,
        index=0,
        prompt="What is 1+5?",
        tokens=[3838, 374, 220, 16, 10, 20, 30, 59, 79075, 90, 21, 92],
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response="\\boxed{6}",
        response_length=5,
        label="6",
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
