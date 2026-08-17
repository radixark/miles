"""Driver wiring: the control phase's claim → execute → publish barrier →
deferred completion order, the tinker arg defaults, and the serving identity
stamped onto completed publishes."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio
from types import SimpleNamespace

from train_tinker_backend import ActorGroupWeightPublisher, run_control_phase


class Remote:
    """Async .remote(...) recorder returning a scripted value."""

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
    asyncio.run(run_control_phase(actor_model, controller, ActorGroupWeightPublisher(actor_model)))

    order = [name for name, _ in log]
    # A deferred batch holds its lease through the publish barrier: release
    # comes strictly AFTER the deferred completions.
    assert order == ["claim", "execute", "complete", "update_weights", "complete", "release"]
    first_complete = log[2][1][0]
    assert set(first_complete) == {"opt1"}  # deferred ops are NOT completed pre-push
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
    asyncio.run(run_control_phase(actor_model, controller, ActorGroupWeightPublisher(actor_model)))
    # Immediate controls release after controller completion, before the push.
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
    asyncio.run(run_control_phase(actor_model, controller, ActorGroupWeightPublisher(actor_model)))
    assert [name for name, _ in log] == ["claim", "update_weights"]


def test_validate_tinker_args_defaults_the_rollout_plane():
    from miles.utils.tinker_backend import validate_tinker_args

    args = SimpleNamespace(
        tinker_backend=True,
        multi_lora_n_adapters=4,
        rollout_function_path=None,
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        use_dynamic_global_batch_size=False,
    )
    validate_tinker_args(args)
    assert args.rollout_function_path == "miles.rollout.tinker_backend.rollout_fn.TinkerRolloutFn"
    assert args.data_source_path == "miles.rollout.tinker_backend.rollout_fn.TinkerNullDataSource"
    assert args.use_dynamic_global_batch_size is True

    # Explicit user choices are honored.
    args.rollout_function_path = "my.custom.Fn"
    args.data_source_path = "my.custom.Source"
    validate_tinker_args(args)
    assert args.rollout_function_path == "my.custom.Fn"
    assert args.data_source_path == "my.custom.Source"

    off = SimpleNamespace(tinker_backend=False)
    validate_tinker_args(off)  # no-op without the flag


class TestDataBatchFinalizer:
    """train_data_batch: a NORMAL train commits rank-side; every other exit
    (abnormal TrainStepOutcome, raised train error) must fail the batch's
    CLAIMED operations typed server and release the lease — never leave the
    SDK futures CLAIMED forever (external review P1)."""

    def _pack(self):
        lease = {
            "dispatch_id": "lease-9",
            "bindings_by_operation": [["fb1", ["A", "r-A", 0]], ["fb2", ["B", "r-B", 1]]],
        }
        pack = {"data_ref": None, "tinker_dispatch": {"operation_ids": ["fb1", "fb2"], "lease": lease}}
        return pack, lease

    def test_normal_outcome_never_calls_the_finalizer(self):
        from train_tinker_backend import train_data_batch

        from miles.backends.megatron_utils.ft.types import TrainStepOutcome

        log: list = []
        controller = SimpleNamespace(fail_tinker_batch=Remote(log, "fail"))

        async def train(rollout_id, rollout_data):
            return [TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL]

        pack, _ = self._pack()
        asyncio.run(train_data_batch(SimpleNamespace(train=train), controller, 0, pack))
        assert log == []

    def test_abnormal_outcome_fails_the_batch_operations_and_releases_the_lease(self):
        from train_tinker_backend import train_data_batch

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
        from train_tinker_backend import train_data_batch

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
        from train_tinker_backend import train_data_batch

        from miles.backends.megatron_utils.ft.types import TrainStepOutcome

        log: list = []
        controller = SimpleNamespace(fail_tinker_batch=Remote(log, "fail"))

        async def train(rollout_id, rollout_data):
            return [TrainStepOutcome.DISCARDED_SHOULD_RETRY]

        asyncio.run(train_data_batch(SimpleNamespace(train=train), controller, 0, {"data_ref": None}))
        [(name, (operation_ids, error, lease_arg))] = log
        assert operation_ids == [] and lease_arg is None
