"""Multi-LoRA wrapper selection invariants: whole child batches are atomic,
the round-robin cursor persists, failures isolate per adapter, and the merge
emits the BatchPlan exactly once per selected adapter."""

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

import miles.rollout.multi_lora.rollout_fn as rollout_fn_module
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.multi_lora.rollout_fn import AdapterRolloutRuntime, MultiLoRARolloutFn, leaf_sample_count
from miles.utils.adapter_config import AdapterRun, AdapterRunConfig
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.types import AdapterRef, Sample


def make_wrapper(
    rollout_batch_size=2,
    n_samples_per_prompt=2,
    coalesce_wait=0.05,
    empty_wait=0.2,
) -> MultiLoRARolloutFn:
    wrapper = MultiLoRARolloutFn.__new__(MultiLoRARolloutFn)
    wrapper.args = SimpleNamespace(
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
        multi_lora_max_coalesce_wait_s=coalesce_wait,
        multi_lora_max_empty_wait_s=empty_wait,
    )
    wrapper.runtimes = {}
    wrapper.rotation = deque()
    wrapper._ready = asyncio.Event()
    return wrapper


def make_ready_runtime(name: str, reg: str, slot: int, n_groups: int, group_size: int = 2, reward=None):
    config = AdapterRunConfig(data="/dev/null", rollout_batch_size=n_groups, n_samples_per_prompt=group_size)
    run = AdapterRun(name=name, config=config, slot=slot, version=1, registration_id=reg)
    runtime = AdapterRolloutRuntime.__new__(AdapterRolloutRuntime)
    runtime.run = run
    runtime.state = AdapterRolloutRuntime.READY
    runtime.task = None
    runtime.error = None
    ref = AdapterRef(name=name, registration_id=reg, serving_version=1, slot=slot)
    groups = [
        [Sample(prompt="p", adapter=ref, metadata={}, reward=reward) for _ in range(group_size)]
        for _ in range(n_groups)
    ]
    runtime.ready_output = RolloutFnTrainOutput(samples=groups, metrics={"reward": slot})
    return runtime


def add_runtime(wrapper, runtime) -> None:
    wrapper.runtimes[runtime.tenant] = runtime
    wrapper.rotation.append(runtime.tenant)


class _FakeControllerHandle:
    """Keep-warm controller: every tenant is admitted on its stamped slot."""

    def __init__(self, calls):
        self.record_train_selection = SimpleNamespace(
            remote=lambda *args: calls.append(("record_train_selection", args))
        )
        self.abort_bind = SimpleNamespace(remote=lambda txn: calls.append(("abort_bind", txn)))
        slot_by_name = {"A": 0, "B": 1}
        self.plan_bind = SimpleNamespace(
            remote=lambda txn, tenants: {
                tuple(t): {"slot": slot_by_name[t[0]], "evict": None, "txn_id": txn} for t in tenants
            }
        )


@pytest.fixture()
def controller_calls(monkeypatch):
    calls: list = []
    monkeypatch.setattr(rollout_fn_module, "ray", SimpleNamespace(get=lambda ref: ref))
    import miles.ray.multi_lora.controller as controller_module

    monkeypatch.setattr(controller_module, "get_multi_lora_controller", lambda: _FakeControllerHandle(calls))
    return calls


class TestSelection:
    def test_whole_batches_overshoot_and_never_split(self):
        # soft target = 2*2 = 4 samples; A has 2 (1 group), B has 6 (3 groups):
        # both whole batches ship, total 8 > 4, nothing is trimmed.
        wrapper = make_wrapper(rollout_batch_size=2, n_samples_per_prompt=2)
        add_runtime(wrapper, make_ready_runtime("A", "ra", 0, n_groups=1))
        add_runtime(wrapper, make_ready_runtime("B", "rb", 1, n_groups=3))
        selected = asyncio.run(wrapper._select())
        assert [r.run.name for r in selected] == ["A", "B"]
        assert sum(leaf_sample_count(r.ready_output.samples) for r in selected) == 8

    def test_first_batch_larger_than_target_ships_alone(self):
        wrapper = make_wrapper(rollout_batch_size=1, n_samples_per_prompt=2)
        add_runtime(wrapper, make_ready_runtime("A", "ra", 0, n_groups=5))
        selected = asyncio.run(wrapper._select())
        assert [r.run.name for r in selected] == ["A"]

    def test_round_robin_cursor_persists_across_selections(self):
        wrapper = make_wrapper(rollout_batch_size=1, n_samples_per_prompt=2)
        a = make_ready_runtime("A", "ra", 0, n_groups=1)
        b = make_ready_runtime("B", "rb", 1, n_groups=1)
        add_runtime(wrapper, a)
        add_runtime(wrapper, b)
        first = asyncio.run(wrapper._select())
        assert [r.run.name for r in first] == ["A"]
        # A consumed; refill BOTH — the cursor must now favor B, not restart at A.
        a.state = AdapterRolloutRuntime.READY
        b.state = AdapterRolloutRuntime.READY
        second = asyncio.run(wrapper._select())
        assert second[0].run.name == "B"

    def test_coalesce_deadline_ships_partial_selection(self):
        # One adapter ready, the other still in flight: after the coalesce
        # window the selection ships with what it has.
        wrapper = make_wrapper(rollout_batch_size=4, n_samples_per_prompt=2, coalesce_wait=0.05)
        a = make_ready_runtime("A", "ra", 0, n_groups=1)
        b = make_ready_runtime("B", "rb", 1, n_groups=1)
        b.state = AdapterRolloutRuntime.IN_FLIGHT
        add_runtime(wrapper, a)
        add_runtime(wrapper, b)
        selected = asyncio.run(wrapper._select())
        assert [r.run.name for r in selected] == ["A"]

    def test_empty_timeout_raises_instead_of_spinning(self):
        wrapper = make_wrapper(empty_wait=0.05)
        with pytest.raises(EmptyBatchTimeoutError):
            asyncio.run(wrapper._select())


class TestMerge:
    def test_merge_emits_batch_plan_and_books_the_selection(self, controller_calls):
        # The wrapper args (SimpleNamespace) deliberately carry no reward_key:
        # the reward observability must tolerate minimal args, and B's
        # reward-less samples must not emit a reward metric.
        wrapper = make_wrapper()
        a = make_ready_runtime("A", "ra", 0, n_groups=2, reward=0.5)
        b = make_ready_runtime("B", "rb", 1, n_groups=1, group_size=3)
        add_runtime(wrapper, a)
        add_runtime(wrapper, b)
        output = asyncio.run(wrapper._merge(7, [a, b]))

        plan = {entry["name"]: entry for entry in output.metadata["batch_plan"]}
        assert plan["A"] == dict(
            name="A",
            registration_id="ra",
            bound_slot=0,
            evict=None,
            actual_sample_count=4,
            actual_rollout_count=4,  # classic 1:1 — every sample is its own execution
            prompt_group_sizes=[2, 2],
        )
        assert plan["B"]["actual_sample_count"] == 3
        assert output.metadata["train_txn_id"]
        # The BatchPlan is the ONLY control plane: nothing rides in sample
        # metadata, and the selection is booked exactly once on the controller.
        head = output.samples[0][0]
        assert "step_slots" not in head.metadata
        assert controller_calls == [("record_train_selection", (7, ["A", "B"]))]
        # Metrics are namespaced per adapter — no cross-adapter key collisions.
        # A's numeric rewards surface as a per-run mean; B has none to report.
        assert output.metrics == {"A/reward": 0, "B/reward": 1, "A/rollout_reward_mean": 0.5}
        # Selected runtimes are consumed and gated until the next generate call.
        assert a.state == AdapterRolloutRuntime.IDLE
        assert a.ready_output is None

    def test_agentic_siblings_count_once_per_execution(self, controller_calls):
        # An agentic child emits K sibling samples per rollout execution
        # (shared rollout_id). The loss aggregates each execution to ONE unit,
        # so the step normalizer must divide by executions — counting samples
        # would shrink a K-trajectory adapter's update by 1/K.
        wrapper = make_wrapper()
        a = make_ready_runtime("A", "ra", 0, n_groups=2, group_size=3)
        for execution_id, group in enumerate(a.ready_output.samples):
            for sample in group:
                sample.rollout_id = execution_id
        add_runtime(wrapper, a)
        output = asyncio.run(wrapper._merge(9, [a]))

        [entry] = output.metadata["batch_plan"]
        assert entry["actual_sample_count"] == 6  # 2 executions x 3 trajectories
        assert entry["actual_rollout_count"] == 2


class TestChildIsolation:
    def test_foreign_samples_fail_only_that_adapter(self):
        wrapper = make_wrapper()
        runtime = make_ready_runtime("A", "ra", 0, n_groups=1)
        foreign_ref = AdapterRef(name="B", registration_id="rb", serving_version=1, slot=1)
        foreign = RolloutFnTrainOutput(samples=[[Sample(prompt="p", adapter=foreign_ref, metadata={})]])

        async def bad_child(_input):
            return foreign

        runtime.child_fn = bad_child
        runtime.state = AdapterRolloutRuntime.IN_FLIGHT
        asyncio.run(wrapper._run_child(runtime, rollout_id=1))
        assert runtime.state == AdapterRolloutRuntime.FAILED
        assert "must draw from their own adapter data source" in str(runtime.error)

    def test_empty_child_batch_fails_that_adapter(self):
        wrapper = make_wrapper()
        runtime = make_ready_runtime("A", "ra", 0, n_groups=1)

        async def empty_child(_input):
            return RolloutFnTrainOutput(samples=[])

        runtime.child_fn = empty_child
        runtime.state = AdapterRolloutRuntime.IN_FLIGHT
        asyncio.run(wrapper._run_child(runtime, rollout_id=1))
        assert runtime.state == AdapterRolloutRuntime.FAILED

    def test_successful_child_becomes_ready(self):
        wrapper = make_wrapper()
        runtime = make_ready_runtime("A", "ra", 0, n_groups=1)
        payload = runtime.ready_output
        runtime.ready_output = None

        async def good_child(_input):
            return payload

        runtime.child_fn = good_child
        runtime.state = AdapterRolloutRuntime.IN_FLIGHT
        asyncio.run(wrapper._run_child(runtime, rollout_id=1))
        assert runtime.state == AdapterRolloutRuntime.READY
        assert runtime.ready_output is payload


class TestChildAbortScoping:
    """The end-of-collection abort targets only the child's own rid namespace,
    never another tenant's in-flight requests."""

    def test_end_of_collection_abort_targets_only_own_namespace(self, monkeypatch):
        from miles.rollout.inference_rollout import inference_rollout_train as irt
        from miles.utils.multi_lora import rid_prefix

        posted = []

        async def record_post(url, payload):
            posted.append((url, payload))

        async def one_engine(_args):
            return ["http://engine:1"]

        async def no_hook(_args):
            return None

        monkeypatch.setattr(irt, "post", record_post)
        monkeypatch.setattr(irt, "get_worker_urls", one_engine)
        monkeypatch.setattr(irt, "call_agent_abort_hook", no_hook)

        args = SimpleNamespace(
            multi_lora=True,
            multi_lora_adapter_identity=("a", "reg1"),
            partial_rollout=False,
        )
        state = SimpleNamespace(args=args, aborted=False)
        asyncio.run(irt.abort(state, set(), rollout_id=0))

        assert posted == [("http://engine:1/abort_request", {"rid": rid_prefix("a", "reg1"), "prefix": True})]

        # The identity the abort keys on is stamped at data-source construction.
        monkeypatch.setattr(
            rollout_fn_module,
            "RolloutDataSourceWithBuffer",
            lambda child_args: SimpleNamespace(args=child_args),
        )
        config = AdapterRunConfig(data="/dev/null", rollout_batch_size=2, n_samples_per_prompt=2)
        run = AdapterRun(name="a", config=config, slot=0, version=1, registration_id="reg1")
        source_args = SimpleNamespace(
            input_key="q", label_key="l", metadata_key=None, save=None, load=None, n_samples_per_prompt=2
        )
        source = rollout_fn_module._AdapterDataSource(source_args, run)
        assert source.args.multi_lora_adapter_identity == ("a", "reg1")
