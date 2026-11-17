"""`Sample.max_response_len` lowers `max_new_tokens` for that sample only, in both rollout paths."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.rollout.inference_rollout.conftest import make_state

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import apply_sample_max_response_len
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample

_RUN_PARAMS = {"temperature": 1.0, "max_new_tokens": 100}


class TestApplySampleMaxResponseLen:
    def test_unset_keeps_the_run_params(self):
        assert apply_sample_max_response_len(_RUN_PARAMS, Sample(prompt="p")) is _RUN_PARAMS

    @pytest.mark.parametrize("limit", [1, 40, 100])
    def test_a_limit_up_to_the_run_value_is_applied(self, limit):
        params = apply_sample_max_response_len(_RUN_PARAMS, Sample(prompt="p", max_response_len=limit))

        assert params == {"temperature": 1.0, "max_new_tokens": limit}
        assert _RUN_PARAMS["max_new_tokens"] == 100

    def test_a_run_without_a_bound_accepts_any_positive_limit(self):
        unbounded = {"temperature": 1.0, "max_new_tokens": None}
        params = apply_sample_max_response_len(unbounded, Sample(prompt="p", max_response_len=10**6))

        assert params["max_new_tokens"] == 10**6
        with pytest.raises(ValueError, match="must be an int"):
            apply_sample_max_response_len(unbounded, Sample(prompt="p", max_response_len=0))

    @pytest.mark.parametrize("limit", [101, 0, -1, True, 40.0, "40"])
    def test_a_limit_outside_the_run_bound_is_rejected(self, limit):
        with pytest.raises(ValueError, match=r"must be an int in \[1, 100\]"):
            apply_sample_max_response_len(_RUN_PARAMS, Sample(prompt="p", index=3, max_response_len=limit))


class _RecordingGenerate:
    def __init__(self) -> None:
        self.max_new_tokens: list[int] = []

    async def __call__(self, input):
        self.max_new_tokens.append(input.sampling_params["max_new_tokens"])
        return GenerateFnOutput(samples=input.sample)


@pytest.mark.parametrize(("limit", "expected"), [(None, 100), (40, 40)])
async def test_inference_rollout_passes_the_sample_budget(limit, expected):
    generate = _RecordingGenerate()
    run_params = dict(_RUN_PARAMS)

    await common.generate_and_rm(
        make_state(generate), Sample(index=1, max_response_len=limit), sampling_params=run_params
    )

    assert generate.max_new_tokens == [expected]
    assert run_params == _RUN_PARAMS


@pytest.mark.parametrize(("limit", "expected"), [(None, 100), (40, 40)])
async def test_legacy_rollout_passes_the_sample_budget(monkeypatch, limit, expected):
    generate = _RecordingGenerate()
    run_params = dict(_RUN_PARAMS)
    args = make_args(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        custom_generate_function_path="recording",
    )
    legacy_state = SimpleNamespace(aborted=False, semaphore=nullcontext(), dp_rank_context=nullcontext)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: legacy_state)
    monkeypatch.setattr(sglang_rollout, "load_generate_function", lambda _path: generate)

    await sglang_rollout.generate_and_rm(args, Sample(index=1, max_response_len=limit), run_params)

    assert generate.max_new_tokens == [expected]
    assert run_params == _RUN_PARAMS


def test_the_budget_survives_retry_and_serialization():
    sample = Sample(prompt="p", max_response_len=40)
    sample.reset_for_retry()

    assert sample.max_response_len == 40
    assert Sample.from_dict(sample.to_dict()).max_response_len == 40
