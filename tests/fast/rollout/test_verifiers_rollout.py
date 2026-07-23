import asyncio
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from packaging.version import Version

if sys.version_info < (3, 11):
    pytest.skip("Verifiers requires Python 3.11+", allow_module_level=True)

from miles.rollout.verifiers_rollout import (
    MilesSGLangTransport,
    VerifiersRolloutFn,
    _check_version,
    _finish_reason,
    _load_config_data,
    _make_eval_args,
    _raise_for_unsupported_trace_errors,
    _renderer_identity,
    _trace_eval_reward,
    _train_client,
    _validate_group_reward_sample_counts,
    trace_to_sample,
    trace_to_samples,
)
from miles.utils.types import Sample


def _args(**overrides) -> Namespace:
    values = {
        "lora_adapter_path": None,
        "lora_rank": 0,
        "reward_key": None,
        "rollout_max_context_len": 64,
        "rollout_max_prompt_len": None,
        "rollout_max_response_len": 8,
        "rollout_skip_special_tokens": True,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "sglang_model_routers": None,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_policy": "round_robin",
        "sglang_router_port": 30000,
        "sglang_tokenizer_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _branch(*, index=0, token_ids=None, sampled_mask=None, logprobs=None):
    return SimpleNamespace(
        index=index,
        token_ids=[10, 11, 20, 21, 22] if token_ids is None else token_ids,
        sampled_mask=[False, False, True, False, True] if sampled_mask is None else sampled_mask,
        logprobs=[0.0, 0.0, -0.1, 0.0, -0.2] if logprobs is None else logprobs,
    )


def _trace(**overrides):
    values = {
        "id": "trace-1",
        "branches": [_branch()],
        "task": SimpleNamespace(data=SimpleNamespace(prompt="solve this", idx="task-1")),
        "rewards": {"score": 1.25, "bonus": 0.75},
        "metrics": {"turns": 2.0},
        "stop_condition": "done",
        "error": None,
        "has_error": False,
        "is_truncated": False,
        "reward": 2.0,
        "last_reply": "answer",
        "num_turns": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_loader_uses_verifiers_toml_format(tmp_path):
    path = tmp_path / "verifiers.toml"
    path.write_text('[taskset]\nid = "gsm8k-v1"\n')

    assert _load_config_data(path) == {"taskset": {"id": "gsm8k-v1"}}


def test_config_loader_rejects_non_toml_formats(tmp_path):
    path = tmp_path / "verifiers.yaml"
    path.write_text("taskset: gsm8k-v1\n")

    with pytest.raises(ValueError, match="Verifiers TOML"):
        _load_config_data(path)


def test_verifiers_0_2_1_is_rejected_with_compatibility_reason():
    with pytest.raises(RuntimeError, match="SGLang 0.5.15 pins OpenAI"):
        _check_version("verifiers", "0.2.1", Version("0.2.0"), Version("0.2.1"))


def test_transport_does_not_treat_aborted_generation_as_complete():
    with pytest.raises(RuntimeError, match="aborted"):
        _finish_reason({"meta_info": {"finish_reason": {"type": "abort"}}})


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("/models/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
        (
            "/cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/revision",
            "Qwen/Qwen3-4B-Instruct-2507",
        ),
        ("/models/private-finetune", None),
    ],
)
def test_renderer_identity_is_inferred_from_standard_checkpoint_paths(checkpoint, expected):
    pytest.importorskip("renderers", minversion="0.1.8")

    assert _renderer_identity(checkpoint) == expected


def test_train_client_uses_local_tokenizer_with_inferred_renderer_identity(monkeypatch):
    renderers = pytest.importorskip("renderers", minversion="0.1.8")
    checkpoint = "/cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/revision"
    seen = {}

    class BaseTrainClient:
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            self.openai = openai
            self.pool_size = pool_size
            self.config = config
            self.renderer_model_name = renderer_model_name
            self._pool = None

    class RendererPool:
        def __init__(self, factory, size):
            seen["size"] = size
            seen["renderer"] = factory()

    def load_tokenizer(source):
        seen["source"] = source
        return SimpleNamespace(name_or_path=source)

    def create_renderer(tokenizer, config, *, chat_template_kwargs=None):
        seen["identity"] = tokenizer.name_or_path
        seen["config"] = config
        seen["kwargs"] = chat_template_kwargs
        return "renderer"

    monkeypatch.setattr(renderers, "RendererPool", RendererPool)
    monkeypatch.setattr(renderers, "create_renderer", create_renderer)
    monkeypatch.setattr("renderers.base.load_tokenizer", load_tokenizer)
    runtime = SimpleNamespace(TrainClient=BaseTrainClient)

    args = _args(sglang_tokenizer_path="/models/custom-tokenizer")
    client = _train_client(runtime, args, checkpoint, pool_size=3)
    pool = client._renderer_pool(checkpoint, chat_template_kwargs={"enable_thinking": False})

    assert isinstance(pool, RendererPool)
    assert seen == {
        "size": 3,
        "source": "/models/custom-tokenizer",
        "identity": "Qwen/Qwen3-4B-Instruct-2507",
        "config": None,
        "kwargs": {"enable_thinking": False},
        "renderer": "renderer",
    }


@pytest.mark.asyncio
async def test_train_client_reports_unsupported_tool_renderer_as_configuration_error():
    class ProviderError(Exception):
        def __init__(self, message, *, status_code):
            super().__init__(message)
            self.status_code = status_code

    class BaseTrainClient:
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            pass

        async def get_response(self, *args, **kwargs):
            raise ValueError("RendererPool does not support tools.")

    runtime = SimpleNamespace(ProviderError=ProviderError, TrainClient=BaseTrainClient)
    client = _train_client(runtime, _args(), "/models/private-finetune", pool_size=1)

    with pytest.raises(ProviderError, match="--sglang-tokenizer-path") as error:
        await client.get_response()

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_train_client_reports_unsupported_dialect_as_configuration_error():
    class ProviderError(Exception):
        def __init__(self, message, *, status_code):
            super().__init__(message)
            self.status_code = status_code

    class BaseTrainClient:
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            pass

        async def get_response(self, *args, **kwargs):
            raise NotImplementedError("only the chat-completions dialect is supported")

    runtime = SimpleNamespace(ProviderError=ProviderError, TrainClient=BaseTrainClient)
    client = _train_client(runtime, _args(), "/models/test", pool_size=1)

    with pytest.raises(ProviderError, match="does not support this request") as error:
        await client.get_response()

    assert error.value.status_code == 400


@pytest.mark.parametrize(("method", "message"), [("relay", "streaming"), ("relay_aux", "auxiliary")])
@pytest.mark.asyncio
async def test_train_client_reports_unsupported_relay_paths_as_configuration_errors(method, message):
    class ProviderError(Exception):
        def __init__(self, text, *, status_code):
            super().__init__(text)
            self.status_code = status_code

    class BaseTrainClient:
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            pass

    runtime = SimpleNamespace(ProviderError=ProviderError, TrainClient=BaseTrainClient)
    client = _train_client(runtime, _args(), "/models/test", pool_size=1)

    with pytest.raises(ProviderError, match=message) as error:
        await getattr(client, method)()

    assert error.value.status_code == 400


def test_trace_to_sample_preserves_training_fields_and_verifiers_reward():
    sample = trace_to_sample(_args(), _trace(), group_index=3, index=9)

    assert sample.group_index == 3
    assert sample.index == 9
    assert sample.prompt == "solve this"
    assert sample.tokens == [10, 11, 20, 21, 22]
    assert sample.response == "answer"
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2]
    assert sample.reward == 2.0
    assert sample.routing_key == "trace-1"
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata["verifiers"]["task_index"] == "task-1"
    sample.validate()


def test_trace_to_sample_preserves_named_rewards_when_reward_key_is_set():
    args = _args(reward_key="score")

    sample = trace_to_sample(args, _trace(), group_index=0, index=0)

    assert sample.reward == {"score": 1.25, "bonus": 0.75, "reward": 2.0}
    assert sample.get_reward_value(args) == 1.25


def test_trace_to_sample_serializes_structured_prompt_messages():
    class Message:
        def model_dump(self, **kwargs):
            assert kwargs == {"mode": "json", "exclude_none": True}
            return {"role": "user", "content": "solve this"}

    trace = _trace(task=SimpleNamespace(data=SimpleNamespace(prompt=[Message()], idx="task-1")))

    sample = trace_to_sample(_args(), trace, group_index=0, index=0)

    assert sample.prompt == [{"role": "user", "content": "solve this"}]


def test_trace_to_sample_marks_error_before_truncation():
    error = SimpleNamespace(model_dump=lambda **_kwargs: {"type": "ProviderError"})

    sample = trace_to_sample(
        _args(),
        _trace(error=error, has_error=True, is_truncated=True),
        group_index=0,
        index=0,
    )

    assert sample.status == Sample.Status.FAILED
    assert sample.metadata["verifiers"]["error"] == {"type": "ProviderError"}


def test_failed_eval_trace_with_missing_named_reward_returns_none():
    trace = _trace(has_error=True, rewards={})

    assert _trace_eval_reward(trace, "score") is None


def test_successful_eval_trace_requires_configured_named_reward():
    with pytest.raises(KeyError, match="score"):
        _trace_eval_reward(_trace(rewards={}), "score")


def test_unsupported_trace_error_is_not_resampled_forever():
    error = SimpleNamespace(message="Miles' Verifiers adapter does not support this request: ResponsesDialect")

    with pytest.raises(RuntimeError, match="ResponsesDialect"):
        _raise_for_unsupported_trace_errors([_trace(error=error, has_error=True)])


def test_graph_branches_fail_before_miles_can_corrupt_trace_groups():
    trace = _trace(branches=[_branch(index=0), _branch(index=1)])

    with pytest.raises(NotImplementedError, match="multiple graph branches"):
        trace_to_samples(_args(), trace, group_index=4, index_start=10)


def test_convert_group_uses_standard_miles_group_shape():
    from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data

    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args()
    adapter._next_sample_index = 0

    group = adapter._convert_group(
        [_trace(id="first"), _trace(id="second")],
        group_index=2,
    )

    assert len(group) == 2
    assert all(isinstance(sample, Sample) for sample in group)
    args = SimpleNamespace(
        disable_rollout_trim_samples=True,
        global_batch_size=1,
        use_dynamic_global_batch_size=False,
    )
    flattened, _ = postprocess_rollout_data(args, [group], train_parallel_config={"dp_size": 1})
    assert [sample.routing_key for sample in flattened] == ["first", "second"]


def test_standard_dynamic_filter_accepts_converted_branch_group():
    from miles.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std

    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args()
    adapter._next_sample_index = 0
    group = adapter._convert_group(
        [_trace(id="low", reward=0.0), _trace(id="high", reward=1.0)],
        group_index=0,
    )

    assert check_reward_nonzero_std(adapter.args, group).keep


@pytest.mark.parametrize(
    ("overrides", "option"),
    [
        (
            {"eval_interval": None, "n_samples_per_prompt": 1, "n_samples_per_eval_prompt": 1},
            "--n-samples-per-prompt",
        ),
        (
            {"eval_interval": 1, "n_samples_per_prompt": 2, "n_samples_per_eval_prompt": 1},
            "--n-samples-per-eval-prompt",
        ),
    ],
)
def test_group_reward_tasks_require_multiple_rollouts(overrides, option):
    args = _args(**overrides)
    tasks = [object()]

    with pytest.raises(ValueError, match=option):
        _validate_group_reward_sample_counts(args, tasks, lambda _task, _kind: [object()])


def test_group_reward_eval_count_is_ignored_when_eval_is_disabled():
    args = _args(
        eval_interval=None,
        n_samples_per_prompt=2,
        n_samples_per_eval_prompt=1,
    )

    _validate_group_reward_sample_counts(args, [object()], lambda _task, _kind: [object()])


def test_group_reward_train_count_is_ignored_for_eval_only_runs():
    args = _args(
        num_rollout=0,
        eval_interval=1,
        n_samples_per_prompt=1,
        n_samples_per_eval_prompt=2,
    )

    _validate_group_reward_sample_counts(args, [object()], lambda _task, _kind: [object()])


@pytest.mark.asyncio
async def test_verifiers_episode_owns_group_reward_computation():
    traces = [_trace(id="a", reward=0.0), _trace(id="b", reward=0.0)]

    class Episode:
        rollouts = []

        async def run(self, semaphore):
            assert semaphore is not None
            traces[0].reward = -1.0
            traces[1].reward = 1.0
            return traces

    class Environment:
        def episode(self, task, ctx, n):
            assert task == "task"
            assert ctx == "ctx"
            assert n == 2
            return Episode()

    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args(sglang_enable_deterministic_inference=False)
    adapter.env = Environment()
    adapter.ctx = "ctx"

    result = await adapter._run_task_group("task", 2, asyncio.Semaphore(2), seed_base=0)

    assert [trace.reward for trace in result] == [-1.0, 1.0]


def test_sampling_config_preserves_miles_minimum_tokens():
    class SamplingConfig:
        @staticmethod
        def model_validate(data):
            return data

    config = VerifiersRolloutFn._sampling_config(
        SamplingConfig,
        _args(
            apply_chat_template_kwargs={},
            rollout_min_new_tokens=3,
            rollout_temperature=0.7,
            rollout_top_k=20,
            rollout_top_p=0.9,
        ),
    )

    assert config["min_tokens"] == 3


def test_eval_args_clear_training_prompt_cap_and_preserve_other_fallbacks():
    args = _args(
        eval_max_context_len=128,
        eval_max_prompt_len=None,
        eval_max_response_len=None,
        eval_min_new_tokens=None,
        eval_reward_key=None,
        eval_temperature=None,
        eval_top_k=None,
        eval_top_p=None,
        reward_key="score",
        rollout_max_context_len=64,
        rollout_max_prompt_len=32,
        rollout_max_response_len=8,
    )

    eval_args = _make_eval_args(args)

    assert eval_args.rollout_max_context_len == 128
    assert eval_args.rollout_max_prompt_len is None
    assert eval_args.rollout_max_response_len == 8
    assert eval_args.reward_key == "score"


@pytest.mark.asyncio
async def test_transport_translates_renderer_request_to_sglang(monkeypatch):
    requests = []

    async def fake_post(url, payload, headers=None):
        requests.append((url, payload, headers))
        return {
            "request_id": "request-id",
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20], [-0.2, 21]],
            },
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    transport = MilesSGLangTransport(_args(sglang_router_policy="manual"))

    response = await transport.post(
        "http://127.0.0.1:30000/inference/v1/generate",
        body={
            "model": "test/model",
            "token_ids": [10, 11],
            "sampling_params": {"temperature": 0.2, "max_tokens": 2, "stop_token_ids": [99], "logprobs": 1},
        },
        options={"headers": {"X-Session-ID": "trace-id"}},
    )

    assert response.json()["choices"][0]["token_ids"] == [20, 21]
    assert requests == [
        (
            "http://127.0.0.1:30000/generate",
            {
                "input_ids": [10, 11],
                "sampling_params": {
                    "temperature": 0.2,
                    "stop_token_ids": [99],
                    "max_new_tokens": 2,
                    "skip_special_tokens": True,
                    "no_stop_trim": True,
                    "spaces_between_special_tokens": False,
                    "n": 1,
                },
                "return_logprob": True,
            },
            {"X-SMG-Routing-Key": "trace-id"},
        )
    ]


@pytest.mark.asyncio
async def test_transport_rejects_multimodal_features():
    transport = MilesSGLangTransport(_args())

    with pytest.raises(NotImplementedError, match="multimodal"):
        await transport.post(
            "http://127.0.0.1:30000/inference/v1/generate",
            body={"token_ids": [1], "sampling_params": {}, "features": {}},
        )


@pytest.mark.asyncio
async def test_transport_bounds_seen_sessions(monkeypatch):
    async def fake_post(_url, _payload, headers=None):
        return {
            "meta_info": {
                "completion_tokens": 1,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20]],
            }
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    transport = MilesSGLangTransport(_args())
    transport._session_cache_size = 2
    body = {"token_ids": [10], "sampling_params": {}}

    for session_id in ("one", "two", "three"):
        await transport.post(
            "http://127.0.0.1:30000/inference/v1/generate",
            body=body,
            options={"headers": {"X-Session-ID": session_id}},
        )

    assert list(transport._seen_sessions) == ["two", "three"]


def test_transport_resolves_router_address_lazily():
    args = _args(sglang_router_ip=None, sglang_router_port=None)
    transport = MilesSGLangTransport(args)

    args.sglang_router_ip = "10.0.0.4"
    args.sglang_router_port = 3210

    assert transport.base_url == "http://10.0.0.4:3210/v1"


def test_transport_uses_default_model_router():
    args = _args(
        sglang_model_routers={
            "default": ("10.0.0.5", 3211),
            "ref": ("10.0.0.6", 3212),
        }
    )

    assert MilesSGLangTransport(args).base_url == "http://10.0.0.5:3211/v1"


def test_eval_transport_keeps_live_router_args():
    rollout_args = _args(sglang_router_ip=None, sglang_router_port=None)
    eval_args = _args(sglang_router_ip=None, sglang_router_port=None)
    transport = MilesSGLangTransport(eval_args, router_args=rollout_args)

    rollout_args.sglang_model_routers = {"default": ("10.0.0.7", 3213)}

    assert transport.base_url == "http://10.0.0.7:3213/v1"


@pytest.mark.asyncio
async def test_eval_rejects_miles_group_reward_model():
    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args(group_rm=True)

    with pytest.raises(AssertionError, match="Group RM is not supported for eval rollout"):
        await adapter._call_eval(SimpleNamespace(rollout_id=0))


@pytest.mark.asyncio
async def test_eval_extracts_structured_miles_reward(monkeypatch):
    import miles.utils as miles_utils

    @asynccontextmanager
    async def serving():
        yield

    async def configure_sglang(_args):
        return None

    async def run_group(*_args, **_kwargs):
        return [_trace(rewards={"bonus": 0.25})]

    async def apply_reward(group):
        group[0].reward = {"score": 0.75, "details": "ok"}

    dumper_utils = SimpleNamespace(configure_sglang=configure_sglang)
    monkeypatch.setitem(sys.modules, "miles.utils.dumper_utils", dumper_utils)
    monkeypatch.setattr(miles_utils, "dumper_utils", dumper_utils, raising=False)
    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args(
        custom_rm_path="tests.fake_reward",
        eval_reward_key="score",
        group_rm=False,
        n_samples_per_eval_prompt=1,
        rm_type=None,
        rollout_batch_size=1,
        rollout_seed=1,
    )
    adapter.eval_args = _args(reward_key="score")
    adapter.config = SimpleNamespace(env_id="test-env")
    adapter.env = SimpleNamespace(serving=serving)
    adapter.eval_ctx = object()
    adapter.max_concurrent = 1
    adapter.model = "test/model"
    adapter._tasks = [object()]
    adapter._next_sample_index = 0
    adapter._run_task_group = run_group
    adapter._apply_miles_rewards = apply_reward

    output = await adapter._call_eval(SimpleNamespace(rollout_id=0))

    assert output.data["test-env"]["rewards"] == [0.75]
