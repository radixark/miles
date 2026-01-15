from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

sample_filter_call_log = {"called": False, "data_len": None, "rewards": None}


def reset_sample_filter_call_log():
    sample_filter_call_log["called"] = False
    sample_filter_call_log["data_len"] = None
    sample_filter_call_log["rewards"] = None


def sample_filter_hook(args, data):
    sample_filter_call_log["called"] = True
    sample_filter_call_log["data_len"] = len(data)
    sample_filter_call_log["rewards"] = [g[0][0].reward if isinstance(g[0], list) else g[0].reward for g in data]


all_samples_process_call_log = {
    "called": False,
    "all_samples_len": None,
    "rewards": None,
    "has_data_source": False,
}


def reset_all_samples_process_call_log():
    all_samples_process_call_log["called"] = False
    all_samples_process_call_log["all_samples_len"] = None
    all_samples_process_call_log["rewards"] = None
    all_samples_process_call_log["has_data_source"] = False


def all_samples_process_hook(args, all_samples, data_source):
    all_samples_process_call_log["called"] = True
    all_samples_process_call_log["all_samples_len"] = len(all_samples)
    all_samples_process_call_log["rewards"] = [
        g[0][0].reward if isinstance(g[0], list) else g[0].reward for g in all_samples
    ]
    all_samples_process_call_log["has_data_source"] = data_source is not None


def filter_by_reward(args, samples, **kwargs):
    reward = samples[0].reward if not isinstance(samples[0], list) else samples[0][0].reward
    if reward == 1:
        return DynamicFilterOutput(keep=True)
    return DynamicFilterOutput(keep=False, reason="reward_zero")


async def multi_sample_generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample = input.sample
    s1 = Sample(
        prompt=sample.prompt,
        response="\\boxed{8}",
        response_length=5,
        tokens=sample.tokens + [59, 79075, 90, 23, 92],
        label=sample.label,
        reward=None,
        status=Sample.Status.COMPLETED,
    )
    s2 = Sample(
        prompt=sample.prompt,
        response="\\boxed{8}",
        response_length=5,
        tokens=sample.tokens + [59, 79075, 90, 23, 92],
        label=sample.label,
        reward=0.5,
        status=Sample.Status.COMPLETED,
    )
    return GenerateFnOutput(samples=[s1, s2])
