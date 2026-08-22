import asyncio
from types import SimpleNamespace

import pytest
import ray
from train_multi_lora_operations import ActorGroupWeightUpdater, generate_with_failure_cap, run_control_phase

from miles.utils.operation_contract import EmptyBatchTimeoutError


class Remote:
    def __init__(self, log, name, value=None):
        self._log, self._name, self._value = log, name, value

    async def remote(self, *args, **kwargs):
        self._log.append((self._name, args))
        return self._value


def test_control_phase_completes_deferred_publishes_only_after_the_push():
    log: list = []

    operations = [
        dict(operation_id="opt1", name="A", kind="optim_step"),
        dict(operation_id="pub1", name="A", kind="save_weights_for_sampler"),
        dict(operation_id="load1", name="A", kind="load_state"),
    ]
    lease = {
        "dispatch_id": "lease-7",
        "bindings_by_operation": [["opt1", ["A", "r-A", 0]], ["pub1", ["A", "r-A", 0]], ["load1", ["A", "r-A", 0]]],
    }
    controller = SimpleNamespace(
        claim_ready_control_operations=Remote(log, "claim", {"operations": operations, "lease": lease}),
        complete_control_operations=Remote(log, "complete"),
        release_batch_lease=Remote(log, "release"),
    )

    async def execute(ops, lease_metadata):
        log.append(("execute", tuple(op["operation_id"] for op in ops)))
        assert lease_metadata == lease  # every rank receives the batch lease
        return {
            "opt1": dict(ok=True, result=dict(grad_norm=1.0, learning_rate=1e-4)),
            "pub1": dict(ok=True, deferred="publish"),
            "load1": dict(ok=True, deferred="publish", result=dict(step=4, path="/s")),
        }

    async def update_weights():
        log.append(("update_weights", ()))

    actor_model = SimpleNamespace(execute_tinker_controls=execute, update_weights=update_weights)
    asyncio.run(run_control_phase(actor_model, controller, ActorGroupWeightUpdater(actor_model)))

    order = [name for name, _ in log]
    # A deferred batch holds its lease through the publish barrier: release
    # comes strictly AFTER the deferred completions.
    assert order == ["claim", "execute", "complete", "update_weights", "complete", "release"]
    first_complete = log[2][1][0]
    assert set(first_complete) == {"opt1"}
    deferred_complete = log[4][1][0]
    # Deferred completions carry the ORIGINAL execution results (a load_state
    # keeps its restored step; the backend sets the step clock from it).
    assert deferred_complete == {
        "pub1": dict(ok=True),
        "load1": dict(ok=True, result=dict(step=4, path="/s")),
    }
    assert log[5][1] == (lease,)


def test_immediate_only_batch_releases_at_its_completion_boundary():
    log: list = []
    operations = [dict(operation_id="opt1", name="A", kind="optim_step")]
    lease = {"dispatch_id": "lease-8", "bindings_by_operation": [["opt1", ["A", "r-A", 0]]]}
    controller = SimpleNamespace(
        claim_ready_control_operations=Remote(log, "claim", {"operations": operations, "lease": lease}),
        complete_control_operations=Remote(log, "complete"),
        release_batch_lease=Remote(log, "release"),
    )

    async def execute(ops, lease_metadata):
        log.append(("execute", ()))
        return {"opt1": dict(ok=True, result=dict(grad_norm=1.0, learning_rate=1e-4))}

    async def update_weights():
        log.append(("update_weights", ()))

    actor_model = SimpleNamespace(execute_tinker_controls=execute, update_weights=update_weights)
    asyncio.run(run_control_phase(actor_model, controller, ActorGroupWeightUpdater(actor_model)))
    assert [name for name, _ in log] == ["claim", "execute", "complete", "release", "update_weights"]


def test_control_phase_still_pushes_with_no_operations():
    # load_state re-publishes ride pending_push without a claimed operation
    # this cycle; the push call must not be gated on claims.
    log: list = []
    controller = SimpleNamespace(
        claim_ready_control_operations=Remote(log, "claim", {"operations": [], "lease": None}),
        complete_control_operations=Remote(log, "complete"),
        release_batch_lease=Remote(log, "release"),
    )

    async def update_weights():
        log.append(("update_weights", ()))

    actor_model = SimpleNamespace(execute_tinker_controls=None, update_weights=update_weights)
    asyncio.run(run_control_phase(actor_model, controller, ActorGroupWeightUpdater(actor_model)))
    assert [name for name, _ in log] == ["claim", "update_weights"]


def test_validate_tinker_args_defaults_the_rollout_plane():
    from miles.rollout.multi_lora.rollout_fn import MultiLoraOperationBatchFn, TinkerNullDataSource
    from miles.utils.misc import load_function
    from miles.utils.tinker import validate_tinker_args

    args = SimpleNamespace(
        tinker_backend=True,
        multi_lora_n_adapters=4,
        rollout_function_path=None,
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        use_dynamic_global_batch_size=False,
    )
    validate_tinker_args(args)
    assert args.rollout_function_path == "miles.rollout.multi_lora.rollout_fn.MultiLoraOperationBatchFn"
    assert args.data_source_path == "miles.rollout.multi_lora.rollout_fn.TinkerNullDataSource"
    assert args.use_dynamic_global_batch_size is True
    assert load_function(args.rollout_function_path) is MultiLoraOperationBatchFn
    assert load_function(args.data_source_path) is TinkerNullDataSource

    args.rollout_function_path = "my.custom.Fn"
    args.data_source_path = "my.custom.Source"
    validate_tinker_args(args)
    assert args.rollout_function_path == "my.custom.Fn"
    assert args.data_source_path == "my.custom.Source"

    off = SimpleNamespace(tinker_backend=False)
    validate_tinker_args(off)


class TestDataBatchFinalizer:
    """Every non-normal train exit finalizes claimed operations and releases the lease."""

    def _pack(self):
        lease = {
            "dispatch_id": "lease-9",
            "bindings_by_operation": [["fb1", ["A", "r-A", 0]], ["fb2", ["B", "r-B", 1]]],
        }
        pack = {"data_ref": None, "tinker_dispatch": {"operation_ids": ["fb1", "fb2"], "lease": lease}}
        return pack, lease

    def test_normal_outcome_never_calls_the_finalizer(self):
        from train_multi_lora_operations import train_data_batch

        from miles.backends.megatron_utils.ft.types import TrainStepOutcome

        log: list = []
        controller = SimpleNamespace(fail_tinker_batch=Remote(log, "fail"))

        async def train(rollout_id, rollout_data):
            return [TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL]

        pack, _ = self._pack()
        asyncio.run(train_data_batch(SimpleNamespace(train=train), controller, 0, pack))
        assert log == []

    def test_abnormal_outcome_fails_the_batch_operations_and_releases_the_lease(self):
        from train_multi_lora_operations import train_data_batch

        from miles.backends.megatron_utils.ft.types import TrainStepOutcome

        log: list = []
        controller = SimpleNamespace(fail_tinker_batch=Remote(log, "fail"))

        async def train(rollout_id, rollout_data):
            # One rank reporting an abnormal outcome is enough: the batch did
            # not commit anywhere.
            return [TrainStepOutcome.NORMAL, TrainStepOutcome.DISCARDED_SHOULD_RETRY]

        pack, lease = self._pack()
        asyncio.run(train_data_batch(SimpleNamespace(train=train), controller, 3, pack))
        [(name, (operation_ids, error, lease_arg))] = log
        assert name == "fail" and operation_ids == ["fb1", "fb2"] and lease_arg == lease
        # Retry ownership is explicit in the message: the client resubmits.
        assert "discarded_should_retry" in error and "resubmit" in error

    def test_train_exception_finalizes_then_reraises(self):
        import pytest
        from train_multi_lora_operations import train_data_batch

        log: list = []
        controller = SimpleNamespace(fail_tinker_batch=Remote(log, "fail"))

        async def train(rollout_id, rollout_data):
            raise RuntimeError("trainer rank died")

        pack, lease = self._pack()
        with pytest.raises(RuntimeError, match="trainer rank died"):
            asyncio.run(train_data_batch(SimpleNamespace(train=train), controller, 3, pack))
        [(name, (operation_ids, error, lease_arg))] = log
        assert name == "fail" and operation_ids == ["fb1", "fb2"] and lease_arg == lease
        assert "trainer rank died" in error and "poisoned" in error

    def test_missing_dispatch_summary_still_finalizes_with_empty_ids(self):
        # A pack without the summary (defensive: custom conversion path) must
        # not crash the driver; the finalizer degrades to a lease-less no-op
        # call rather than an AttributeError.
        from train_multi_lora_operations import train_data_batch

        from miles.backends.megatron_utils.ft.types import TrainStepOutcome

        log: list = []
        controller = SimpleNamespace(fail_tinker_batch=Remote(log, "fail"))

        async def train(rollout_id, rollout_data):
            return [TrainStepOutcome.DISCARDED_SHOULD_RETRY]

        asyncio.run(train_data_batch(SimpleNamespace(train=train), controller, 0, {"data_ref": None}))
        [(name, (operation_ids, error, lease_arg))] = log
        assert operation_ids == [] and lease_arg is None


class FakeRayTaskError(ray.exceptions.RayTaskError):
    """Real RayTaskError construction needs a serialized traceback; tests only need the cause surface."""

    def __init__(self, cause):
        Exception.__init__(self, str(cause))
        self.cause = cause
        self.function_name = "generate"
        self.traceback_str = f"fake traceback: {cause}"

    def as_instanceof_cause(self):
        return self.cause


class TestGenerateFailureCap:
    """Generate failures skip rounds up to the cap instead of killing the shared multi-tenant service."""

    class Executor:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        async def generate(self, rollout_id):
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def attempt(self, executor, streak, cap=3):
        return asyncio.run(generate_with_failure_cap(executor, 0, streak, cap))

    def test_a_failure_below_the_cap_skips_the_round(self):
        executor = self.Executor([FakeRayTaskError(RuntimeError("engine died"))])
        assert self.attempt(executor, streak=0) == (None, 1)

    def test_a_success_resets_the_streak(self):
        executor = self.Executor([{"batch": 1}])
        assert self.attempt(executor, streak=2) == ({"batch": 1}, 0)

    def test_the_cap_reraises(self):
        executor = self.Executor([FakeRayTaskError(RuntimeError("engine died"))])
        with pytest.raises(ray.exceptions.RayTaskError):
            self.attempt(executor, streak=2, cap=3)

    def test_zero_cap_fails_fast(self):
        executor = self.Executor([FakeRayTaskError(RuntimeError("engine died"))])
        with pytest.raises(ray.exceptions.RayTaskError):
            self.attempt(executor, streak=0, cap=0)

    def test_empty_batch_timeout_neither_counts_nor_resets(self):
        executor = self.Executor([FakeRayTaskError(EmptyBatchTimeoutError("idle"))])
        assert self.attempt(executor, streak=2) == (None, 2)

    def test_interleaved_successes_keep_the_loop_alive(self):
        # fail, succeed, fail: with a cap of 2 the reset means neither failure is the second consecutive one.
        executor = self.Executor(
            [FakeRayTaskError(RuntimeError("a")), {"batch": 1}, FakeRayTaskError(RuntimeError("b"))]
        )
        data, streak = self.attempt(executor, streak=0, cap=2)
        assert data is None and streak == 1
        data, streak = self.attempt(executor, streak=streak, cap=2)
        assert data == {"batch": 1} and streak == 0
        data, streak = self.attempt(executor, streak=streak, cap=2)
        assert data is None and streak == 1
