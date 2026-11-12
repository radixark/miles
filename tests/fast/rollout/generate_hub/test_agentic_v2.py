import argparse
import asyncio
import functools
import logging
from types import SimpleNamespace

import httpx
import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.router.test_sessions_v2 import _serve_router
from tests.fast.utils.output_store.test_dropped import sample_owning

import miles.rollout.generate_hub.agentic_tool_call as agentic_tool_call
from miles.ray.rollout.metrics import _compute_metrics_from_samples
from miles.ray.rollout.rollout_data_conversion import validate_compact_rollout_ids
from miles.rollout import sglang_rollout
from miles.rollout.agentic.agent_function import InfraAbort
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout import inference_rollout_train
from miles.rollout.session.samples.codec import SamplesReply
from miles.rollout.session.types import SessionServerInstance
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.function_registry import function_registry
from miles.utils.output_store import dropped
from miles.utils.output_store.dropped import take_dropped_refs
from miles.utils.output_store.plan import StorePlanError
from miles.utils.retry_utils import NonRetryableError
from miles.utils.test_utils import agent_function_stubs as stubs
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer
from miles.utils.types import Sample


class _Tracer:
    session_id = "sid-1"
    session_server_id = "127.0.0.1:12345"
    session_server_instance_id = None
    base_url = "http://127.0.0.1:12345/sessions/sid-1"
    agent_base_url = base_url

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.agent_metadata = None

    async def collect_samples(self, input_sample, *, max_seq_len, agent_metadata=None):
        self.agent_metadata = agent_metadata
        if self.error is not None:
            raise self.error
        return self.reply


def _generate_input(
    *, evaluation=False, sampling_params=None, speculative_algorithm=None, **args_kwargs
) -> GenerateFnInput:
    args = SimpleNamespace(
        **{
            "session_server_instances": [SessionServerInstance(addr="127.0.0.1:12345")],
            "custom_agent_function_path": "test.fake_agent",
            "custom_agent_function_mode": "inline",
            "rollout_dataset_agent_functions": [],
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "max_seq_len": None,
            "partial_rollout": False,
            "use_session_server": "v2",
            "sglang": make_sglang_config(speculative_algorithm=speculative_algorithm),
            **args_kwargs,
        }
    )
    state = SimpleNamespace(args=args, aborted=False)
    sample = Sample(
        group_index=3,
        index=7,
        prompt=[{"role": "user", "content": "hello"}],
        label="label",
        metadata={"source": "test"},
    )
    return GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params or {}, evaluation=evaluation)


async def _fake_agent(**kwargs):
    return {"agent_result": "done"}


def _session_metadata(spec_info=None):
    return {
        SESSION_ROLLOUT_METRICS_KEY: {
            "session_id": "sid-1",
            "metrics": {"spec_info": spec_info or Sample.SpecInfo().to_dict()},
        }
    }


def _patch_agent(monkeypatch, tracer) -> list[str | None]:
    extra_keys: list[str | None] = []

    async def fake_create(args, *, evaluation=False, sampling_params=None, extra_key=None):
        tracer.evaluation = evaluation
        tracer.sampling_params = sampling_params
        extra_keys.append(extra_key)
        return tracer

    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)
    return extra_keys


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
async def test_success_returns_list_and_forwards_agent_metadata(monkeypatch, evaluation):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    generate_input = _generate_input(evaluation=evaluation, sampling_params={"temperature": 0.7, "max_new_tokens": 8})
    output = await agentic_tool_call.generate(generate_input)

    assert tracer.evaluation is evaluation
    # The session fills these into requests the agent sends without them.
    assert tracer.sampling_params == {"temperature": 0.7, "max_new_tokens": 8}
    assert output.samples == [sample]
    assert output.samples[0].rollout_id is None
    assert tracer.agent_metadata == {"agent_result": "done"}


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_session_identity_for_penalty_logs_is_owned_by_tracer(monkeypatch, version: str) -> None:
    sample = Sample(status=Sample.Status.COMPLETED, response_length=1, tokens=[1], metadata={"session_id": "spoofed"})
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    await agentic_tool_call.generate(_generate_input(use_session_server=version))
    assert sample.metadata["session_id"] == "sid-1"


@pytest.mark.asyncio
async def test_a_max_seq_len_warns_once_that_miles_does_not_stop_the_agent(monkeypatch, caplog):
    """The limit only reaches the agent as metadata and truncates collected samples; the warning says so once."""
    # a cache of its own, so neither earlier nor later tests share the once-per-process state
    warn = agentic_tool_call._warn_max_seq_len_does_not_stop_the_agent
    monkeypatch.setattr(agentic_tool_call, warn.__name__, functools.cache(warn.__wrapped__))
    sample = Sample(status=Sample.Status.COMPLETED, response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with caplog.at_level(logging.WARNING, logger=agentic_tool_call.__name__):
        for _ in range(2):
            await agentic_tool_call.generate(_generate_input(max_seq_len=4096))

    assert caplog.text.count("does not stop the agent") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("input_rollout_id", "expected_rollout_id"), [(None, 7), (11, 11)])
async def test_success_assigns_shared_rollout_id_to_v2_leaves(monkeypatch, input_rollout_id, expected_rollout_id):
    leaves = [
        Sample(status=Sample.Status.COMPLETED, response="one", response_length=1, tokens=[1]),
        Sample(status=Sample.Status.COMPLETED, response="two", response_length=1, tokens=[2]),
    ]
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()
    generate_input.sample.rollout_id = input_rollout_id

    output = await agentic_tool_call.generate(generate_input)

    assert [sample.rollout_id for sample in output.samples] == [expected_rollout_id] * 2
    validate_compact_rollout_ids([[output.samples]])


@pytest.mark.asyncio
async def test_v2_requires_input_rollout_identity(monkeypatch):
    leaves = [
        Sample(status=Sample.Status.COMPLETED, response="one", response_length=1, tokens=[1]),
        Sample(status=Sample.Status.COMPLETED, response="two", response_length=1, tokens=[2]),
    ]
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()
    generate_input.sample.index = None
    generate_input.sample.rollout_id = None

    with pytest.raises(AssertionError, match="require input Sample.rollout_id or Sample.index"):
        await agentic_tool_call.generate(generate_input)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("empty_reason", "exit_status"), [("no_records", "NoModelCalls"), ("all_truncated", "AllTruncated")]
)
async def test_empty_reply_returns_aborted_list(monkeypatch, empty_reason, exit_status):
    tracer = _Tracer(SamplesReply(samples=[], session_metadata={}, empty_reason=empty_reason))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()

    output = await agentic_tool_call.generate(generate_input)

    assert isinstance(output.samples, list)
    assert len(output.samples) == 1
    assert output.samples[0] is not generate_input.sample
    assert output.samples[0].status == Sample.Status.ABORTED
    assert output.samples[0].metadata["exit_status"] == exit_status
    assert SESSION_ROLLOUT_METRICS_KEY not in output.samples[0].metadata


@pytest.mark.asyncio
async def test_infra_abort_discards_the_sample_with_its_cause(monkeypatch):
    monkeypatch.setattr(dropped, "_dropped", {})
    sample = sample_owning("collected")
    sample.status = Sample.Status.COMPLETED
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _abort_for_infra)
    generate_input = _generate_input()

    output = await agentic_tool_call.generate(generate_input)

    (aborted,) = output.samples
    assert aborted.status == Sample.Status.ABORTED
    assert aborted.metadata == {"source": "test", "exit_status": "ServerUnreachable"}
    assert tracer.agent_metadata is None
    assert list(take_dropped_refs()) == ["collected"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collect_error", [TimeoutError("samples unavailable"), RuntimeError("HTTP 503 from /samples")]
)
async def test_infra_abort_cause_outranks_a_failed_collection(monkeypatch, collect_error):
    tracer = _Tracer(error=collect_error)
    _patch_agent(monkeypatch, tracer)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _abort_for_infra)

    output = await agentic_tool_call.generate(_generate_input())

    (aborted,) = output.samples
    assert aborted.metadata["exit_status"] == "ServerUnreachable"


@pytest.mark.asyncio
async def test_any_other_agent_error_keeps_the_recorded_session(monkeypatch):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fail)

    output = await agentic_tool_call.generate(_generate_input())

    assert output.samples == [sample]
    assert sample.status == Sample.Status.COMPLETED


async def _abort_for_infra(**kwargs):
    raise InfraAbort("ServerUnreachable", "the environment host went away")


async def _fail(**kwargs):
    raise RuntimeError("the agent crashed")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["subproc", "inline"])
async def test_every_mode_sends_the_call_with_its_router_scope(monkeypatch, mode):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    _patch_agent(
        monkeypatch, _Tracer(SamplesReply(samples=[sample], session_metadata=_session_metadata(), empty_reason=None))
    )
    seen: dict = {}

    async def fake_call(fn, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(agentic_tool_call, "call_agent_function", fake_call)

    await agentic_tool_call.generate(
        _generate_input(custom_agent_function_mode=mode, sglang_router_ip="10.0.0.1", sglang_router_port=31000)
    )

    assert seen["abort_scope"] == "10.0.0.1:31000"
    assert "worker" not in seen and "extra_env" not in seen


@pytest.mark.asyncio
async def test_worker_mode_sends_the_call_to_the_worker_with_the_runs_env(monkeypatch):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    _patch_agent(monkeypatch, _Tracer(SamplesReply(samples=[sample], session_metadata={}, empty_reason=None)))
    monkeypatch.setenv("TEST_HARNESS_DIR", "/tasks")
    monkeypatch.setenv("TEST_RECIPE_KNOB", "not-in-worker-env")
    monkeypatch.setenv("E2B_API_URL", "https://sandbox-service.example")
    monkeypatch.setenv("OTHER_HARNESS_DIR", "/not-this-harness")
    monkeypatch.setenv("TEST_HARNESS_API_KEY", "not-by-value")
    seen: dict = {}

    async def fake_call(fn, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(agentic_tool_call, "call_agent_function", fake_call)

    with function_registry.temporary("test_harness.agent.WORKER_ENV", ("TEST_HARNESS_*",)):
        await agentic_tool_call.generate(
            _generate_input(
                custom_agent_function_path=_hook("test_harness.agent.run"),
                custom_agent_function_mode="worker",
                custom_agent_function_worker="http://worker:11100",
                sglang_router_ip="10.0.0.1",
                sglang_router_port=31000,
            )
        )

    assert seen["mode"] == "worker"
    assert seen["worker"] == "http://worker:11100"
    assert seen["extra_env"]["TEST_HARNESS_DIR"] == "/tasks"
    assert "TEST_RECIPE_KNOB" not in seen["extra_env"]
    assert seen["extra_env"]["E2B_API_URL"] == "https://sandbox-service.example"
    assert "OTHER_HARNESS_DIR" not in seen["extra_env"]
    assert "TEST_HARNESS_API_KEY" not in seen["extra_env"]
    assert seen["abort_scope"] == "10.0.0.1:31000"


@pytest.mark.asyncio
@pytest.mark.parametrize("worker", [None, "http://worker:abc"])
async def test_worker_mode_without_a_usable_worker_fails_before_a_session_opens_and_stops_the_rollout(
    monkeypatch, worker
):
    extra_keys = _patch_agent(monkeypatch, _Tracer())

    with pytest.raises(NonRetryableError, match="--custom-agent-function-worker"):
        await agentic_tool_call.generate(
            _generate_input(custom_agent_function_mode="worker", custom_agent_function_worker=worker)
        )

    assert extra_keys == []


@pytest.mark.asyncio
async def test_a_worker_secret_no_header_can_carry_fails_before_a_session_opens(monkeypatch):
    extra_keys = _patch_agent(monkeypatch, _Tracer())
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET", "first-half\nsecond-half")

    with pytest.raises(NonRetryableError, match="MILES_AGENT_WORKER_SECRET") as raised:
        await agentic_tool_call.generate(
            _generate_input(custom_agent_function_mode="worker", custom_agent_function_worker="http://worker:11100")
        )

    assert "first-half" not in str(raised.value)
    assert extra_keys == []


def test_the_worker_flags_parse_and_the_worker_env_flag_is_gone():
    parser = argparse.ArgumentParser(exit_on_error=False)
    agentic_tool_call.generate.add_arguments(parser)

    args = parser.parse_args(
        ["--custom-agent-function-mode", "worker", "--custom-agent-function-worker", "http://w:1"]
    )

    assert (args.custom_agent_function_mode, args.custom_agent_function_worker) == ("worker", "http://w:1")
    with pytest.raises((argparse.ArgumentError, SystemExit)):
        parser.parse_args(["--custom-agent-function-worker-env", "TEST_*"])


def _running_agent_input(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(dropped, "_dropped", {})
    collected = sample_owning("collected")
    collected.status = Sample.Status.COMPLETED
    _patch_agent(
        monkeypatch,
        _Tracer(SamplesReply(samples=[collected], session_metadata=_session_metadata(), empty_reason=None)),
    )
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: stubs.wait_for_metadata_paths)
    generate_input = _generate_input(custom_agent_function_mode=mode, custom_agent_function_path=None)
    pid_file = tmp_path / "pid"
    generate_input.sample.metadata.update(pid_file=str(pid_file), cleaned_up=str(tmp_path / "cleaned_up"))
    return generate_input, pid_file


async def _running_group(generate_input, pid_file) -> set:
    async def group():
        return (await agentic_tool_call.generate(generate_input)).samples

    pending = {asyncio.create_task(group())}
    while not pid_file.exists():
        await asyncio.sleep(0.05)
    return pending


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["subproc", "inline"])
async def test_the_rollout_abort_discards_a_running_agent_and_hands_its_objects_back_once(monkeypatch, tmp_path, mode):
    generate_input, pid_file = _running_agent_input(monkeypatch, tmp_path, mode)
    logged: list = []

    async def no_engines(args):
        return []

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", no_engines)
    monkeypatch.setattr(
        inference_rollout_train.SampleOwnershipRecorder, "log_dropped_samples", lambda **kw: logged.append(kw)
    )
    pending = await _running_group(generate_input, pid_file)
    returned = await asyncio.wait_for(
        inference_rollout_train.abort(generate_input.state, pending, rollout_id=0), timeout=30
    )

    assert returned == []
    assert list(take_dropped_refs()) == ["collected"]
    [drop] = logged
    [aborted] = drop["samples"]
    assert aborted.status == Sample.Status.ABORTED
    assert aborted.metadata["exit_status"] == "Cancelled"
    assert drop["reason"] == "aborted"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["subproc", "inline"])
async def test_an_episode_whose_session_opens_after_the_rollout_abort_never_starts_its_agent(
    monkeypatch, tmp_path, mode
):
    generate_input, pid_file = _running_agent_input(monkeypatch, tmp_path, mode)
    logged: list = []
    opening, open_the_session = asyncio.Event(), asyncio.Event()
    create = agentic_tool_call.OpenAIEndpointTracer.create

    async def open_when_released(*args, **kwargs):
        opening.set()
        await open_the_session.wait()
        return await create(*args, **kwargs)

    async def no_engines(args):
        return []

    async def group():
        return (await agentic_tool_call.generate(generate_input)).samples

    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", open_when_released)
    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", no_engines)
    monkeypatch.setattr(
        inference_rollout_train.SampleOwnershipRecorder, "log_dropped_samples", lambda **kw: logged.append(kw)
    )

    pending = {asyncio.create_task(group())}
    await opening.wait()
    aborting = asyncio.create_task(inference_rollout_train.abort(generate_input.state, pending, rollout_id=0))
    while not generate_input.state.aborted:
        await asyncio.sleep(0.01)
    open_the_session.set()
    returned = await asyncio.wait_for(aborting, timeout=10)

    assert returned == []
    assert not pid_file.exists()
    assert list(take_dropped_refs()) == ["collected"]
    [drop] = logged
    [aborted] = drop["samples"]
    assert aborted.metadata["exit_status"] == "Cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["subproc", "inline"])
async def test_the_legacy_rollout_abort_discards_a_running_agent_and_hands_its_objects_back_once(
    monkeypatch, tmp_path, mode
):
    generate_input, pid_file = _running_agent_input(monkeypatch, tmp_path, mode)
    args = generate_input.args
    args.enable_sample_ownership_checker = True
    logged: list = []

    async def no_engines(url):
        return {"workers": []}

    state = generate_input.state
    state.pendings = await _running_group(generate_input, pid_file)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "get", no_engines)
    monkeypatch.setattr(sglang_rollout.SampleOwnershipRecorder, "log_dropped_groups", lambda **kw: logged.append(kw))

    returned = await asyncio.wait_for(sglang_rollout.abort(args, rollout_id=0), timeout=30)

    assert returned == []
    assert not state.pendings
    assert list(take_dropped_refs()) == ["collected"]
    [drop] = logged
    [[aborted]] = drop["before"]
    assert aborted.metadata["exit_status"] == "Cancelled"


@pytest.mark.asyncio
async def test_transport_collection_error_has_no_metrics_owner(monkeypatch):
    tracer = _Tracer(error=TimeoutError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input(speculative_algorithm="EAGLE")
    generate_input.sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = {"session_id": "stale", "metrics": {}}

    output = await agentic_tool_call.generate(generate_input)

    (sample,) = output.samples
    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata["exit_status"] == "CollectFailed"
    assert SESSION_ROLLOUT_METRICS_KEY not in sample.metadata


@pytest.mark.asyncio
async def test_v2_replaces_stale_metrics_with_shared_authoritative_carrier(monkeypatch):
    stale = {"session_id": "stale", "metrics": {"agent": "plant"}}
    leaves = [
        Sample(metadata={SESSION_ROLLOUT_METRICS_KEY: stale}),
        Sample(metadata={SESSION_ROLLOUT_METRICS_KEY: stale}),
    ]
    spec_info = {
        "spec_num_correct_drafts": 1,
        "spec_num_proposed_drafts": 2,
        "spec_verify_ct": 1,
        "completion_tokens": 2,
    }
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata=_session_metadata(spec_info), empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    output = await agentic_tool_call.generate(_generate_input(speculative_algorithm="EAGLE"))

    expected = {"session_id": "sid-1", "metrics": {"spec_info": spec_info}}
    assert [sample.metadata[SESSION_ROLLOUT_METRICS_KEY] for sample in output.samples] == [expected, expected]


@pytest.mark.asyncio
async def test_v2_rejects_missing_server_session_metrics(monkeypatch):
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(KeyError, match=SESSION_ROLLOUT_METRICS_KEY):
        await agentic_tool_call.generate(_generate_input())


@pytest.mark.asyncio
async def test_v2_rejects_metrics_from_another_session(monkeypatch):
    session_metadata = _session_metadata()
    session_metadata[SESSION_ROLLOUT_METRICS_KEY]["session_id"] = "sid-2"
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata=session_metadata, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(ValueError, match="does not match the collected session"):
        await agentic_tool_call.generate(_generate_input(speculative_algorithm="EAGLE"))


@pytest.mark.asyncio
async def test_v2_rejects_unavailable_metrics_from_successful_collect(monkeypatch):
    session_metadata = _session_metadata()
    session_metadata[SESSION_ROLLOUT_METRICS_KEY]["metrics"] = None
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata=session_metadata, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(ValueError, match="successful session collect must carry metrics"):
        await agentic_tool_call.generate(_generate_input(speculative_algorithm="EAGLE"))


def _empty_tracer() -> _Tracer:
    return _Tracer(SamplesReply(samples=[], session_metadata={}, empty_reason="no_records"))


class TestSessionServerInstancesValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("instances", [None, []], ids=["none", "empty"])
    async def test_empty_session_server_instances_is_rejected(self, monkeypatch, instances):
        """generate() raises the documented AssertionError when session_server_instances is null or empty, without creating a tracer."""
        created_for: list[object] = []

        async def fake_create(args, *, evaluation=False, sampling_params=None, extra_key=None):
            created_for.append(args)
            return _empty_tracer()

        monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
        monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)

        generate_input = _generate_input()
        generate_input.args.session_server_instances = instances

        with pytest.raises(AssertionError, match="requires session_server_instances"):
            await agentic_tool_call.generate(generate_input)

        assert created_for == []


class TestRadixCacheExtraKey:
    @pytest.mark.asyncio
    async def test_generate_hands_the_sample_key_to_the_session(self, monkeypatch):
        """A started sample creates its session under its own namespace."""
        extra_keys = _patch_agent(monkeypatch, _empty_tracer())
        generate_input = _generate_input()
        generate_input.sample.kv_cache_namespace = "train:-:7"

        await agentic_tool_call.generate(generate_input)

        assert extra_keys == ["train:-:7"]

    @pytest.mark.asyncio
    async def test_an_unstarted_sample_creates_an_unkeyed_session(self, monkeypatch):
        """With the partition off the sample carries no namespace, so the session is created unkeyed."""
        extra_keys = _patch_agent(monkeypatch, _empty_tracer())

        await agentic_tool_call.generate(_generate_input())

        assert extra_keys == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, StorePlanError])
async def test_collection_error_propagates(monkeypatch, error_type):
    """Only timeouts and transport errors become an ABORTED sample; a StorePlanError must reach the rollout loop."""
    tracer = _Tracer(error=error_type("samples unavailable"))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(error_type, match="samples unavailable"):
        await agentic_tool_call.generate(_generate_input())


async def _agent_dropping_reasoning(*, base_url, **kwargs):
    messages = [{"role": "user", "content": "turn 0"}]
    async with httpx.AsyncClient(timeout=30.0) as client:
        for turn in range(3):
            response = await client.post(f"{base_url}/v1/chat/completions", json={"messages": messages})
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]
            assert reply["reasoning_content"] == "thought about it"
            messages += [
                {"role": "assistant", "content": reply["content"]},
                {"role": "user", "content": f"turn {turn + 1}"},
            ]
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("matcher", "roots"),
    [("strict", 3), ("loose_tool_call", 3), ("loose_tool_call_ignore_reasoning", 1)],
)
async def test_session_roots_reach_rollout_metrics_through_the_session_server(monkeypatch, matcher, roots):
    original_response = MockSGLangServer._compute_chat_completions_response

    def reasoning_response(mock_self, payload: dict) -> dict:
        response = original_response(mock_self, payload)
        response["choices"][0]["message"]["reasoning_content"] = "thought about it"
        return response

    monkeypatch.setattr(MockSGLangServer, "_compute_chat_completions_response", reasoning_response)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _agent_dropping_reasoning)
    with _serve_router({"session_message_matcher": matcher}) as env:
        generate_input = _generate_input(
            session_server_instances=[SessionServerInstance(addr=env.url.removeprefix("http://"))],
            use_sampling_support_replay=False,
            rollout_temperature=1.0,
        )
        output = await agentic_tool_call.generate(generate_input)

    assert len(output.samples) == roots
    metrics = _compute_metrics_from_samples(
        make_args(advantage_estimator="ppo", use_session_server="v2"), output.samples
    )
    assert metrics["session/roots_per_episode"] == roots
    assert metrics["session/leaves_per_episode"] == roots
    assert metrics["session/one_turn_per_root_ratio"] == (1.0 if roots == 3 else 0.0)
