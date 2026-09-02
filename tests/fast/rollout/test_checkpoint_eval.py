from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip, retarget_args


def make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_router_ip="10.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy=None,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        eval_uses_snapshots=True,
        eval_function_path=None,
        debug_train_only=False,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_retarget_args_swaps_router_and_sizing():
    args = make_args()
    eval_args = retarget_args(args, "10.0.0.9", 39000, num_gpus=2, num_gpus_per_engine=2)

    assert (eval_args.sglang_router_ip, eval_args.sglang_router_port) == ("10.0.0.9", 39000)
    assert eval_args.rollout_num_gpus == 2
    assert eval_args.rollout_num_gpus_per_engine == 2
    # The original namespace is untouched.
    assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 30000)
    assert args.rollout_num_gpus == 4


# ---------------- RolloutManager._eval_checkpoint (the single snapshot path) ----------------


class CheckpointFnStub(CheckpointEvalFn):
    def __init__(self, skip_reason=None):
        self.inputs = []
        self.skip_reason = skip_reason
        self.disposed = False

    async def evaluate_checkpoint(self, checkpoint_dir, input):
        self.inputs.append(input)
        if self.skip_reason is not None:
            raise EvalSkip(self.skip_reason)
        return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})

    def dispose(self):
        self.disposed = True


def make_manager(args, eval_fn=None, fleet=None):
    mgr = object.__new__(rollout_manager_mod.RolloutManager.__ray_actor_class__)
    mgr.args = args
    mgr.rollout_id = 7
    mgr._eval_lock = asyncio.Lock()
    mgr._health_monitors = []
    mgr.use_legacy_rollout_v1 = False
    mgr._metric_checker = None
    mgr.eval_generate_rollout = eval_fn
    mgr._eval_fleet = fleet
    args.eval_uses_snapshots = eval_fn is not None and (fleet is not None or isinstance(eval_fn, CheckpointEvalFn))
    return mgr


@pytest.fixture
def controller_env(monkeypatch):
    logged = {}
    monkeypatch.setattr(rollout_manager_mod, "save_debug_rollout_data", lambda *a, **k: None)
    monkeypatch.setattr(
        rollout_manager_mod,
        "log_eval_rollout_data",
        lambda rollout_id, args, data, extra: logged.setdefault("eval", (rollout_id, data, extra)) or {},
    )
    monkeypatch.setattr(
        rollout_manager_mod,
        "log_eval_skip",
        lambda rollout_id, args, reason: logged.setdefault("skip", (rollout_id, reason)),
    )
    return SimpleNamespace(logged=logged)


async def test_eval_checkpoint_threads_input_and_logs(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot), export_time_seconds=1.5)

    assert len(fn.inputs) == 1
    assert fn.inputs[0].hf_dir == str(snapshot)
    assert fn.inputs[0].weight_version == "5"
    assert fn.inputs[0].generate_state is None
    rollout_id, _data, extra = controller_env.logged["eval"]
    assert rollout_id == 5
    assert extra["eval/lag_steps"] == 2
    assert extra["eval/export_time_seconds"] == 1.5


async def test_bare_eval_reuses_save_hf_snapshot(controller_env, tmp_path):
    """train.py calls eval() with no hf_dir; under snapshot eval the manager
    reuses the periodic --save-hf checkpoint, mirroring EvalDispatcher's
    no-export branch, instead of crashing on the hf_dir assert."""
    snapshot = tmp_path / "hf_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub()
    args = make_args(
        debug_train_only=True,
        hf_checkpoint="/base",
        eval_hf_dir=None,
        save_hf=str(tmp_path / "hf_{rollout_id}"),
        eval_keep_snapshots=2,
    )
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5)

    assert len(fn.inputs) == 1
    assert fn.inputs[0].hf_dir == str(snapshot)


async def test_bare_eval_without_save_hf_skips(controller_env, tmp_path):
    """--eval-hf-dir exports need the async driver's dispatcher; a bare eval
    call cannot export, so it degrades to a skipped point rather than a crash."""
    fn = CheckpointFnStub()
    args = make_args(
        debug_train_only=True,
        hf_checkpoint="/base",
        eval_hf_dir=str(tmp_path),
        save_hf=None,
        eval_keep_snapshots=2,
    )
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5)

    assert fn.inputs == []
    assert controller_env.logged["skip"] == (5, "no_snapshot")


async def test_debug_train_only_runs_snapshot_eval(controller_env, tmp_path):
    """Pure SFT skips shared-engine eval, but a separately pinned snapshot is valid."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub()
    args = make_args(
        debug_train_only=True,
        hf_checkpoint="/base",
        eval_hf_dir=str(tmp_path),
        eval_keep_snapshots=2,
    )
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert len(fn.inputs) == 1
    assert fn.inputs[0].weight_version == "5"


async def test_eval_checkpoint_runs_the_eval_fn_on_the_fleet(controller_env, monkeypatch, tmp_path):
    """The fleet only delivers weights — the configured eval fn still generates, and it
    is the same object the shared posture would call."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    seen_inputs = []

    def eval_generate_rollout(input):
        seen_inputs.append(input)
        return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})

    class FakeFleet:
        def __init__(self):
            self.pins = []

        async def pin(self, checkpoint_dir, weight_version):
            self.pins.append((checkpoint_dir, weight_version))
            return "fleet-state"

    fleet = FakeFleet()
    monkeypatch.setattr(rollout_manager_mod, "call_rollout_function", lambda fn, input: fn(input))
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path))
    mgr = make_manager(args, eval_fn=eval_generate_rollout, fleet=fleet)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert fleet.pins == [(str(snapshot), "5")]
    assert seen_inputs[0].generate_state == "fleet-state"
    assert seen_inputs[0].hf_dir == str(snapshot)
    assert seen_inputs[0].weight_version == "5"


async def test_eval_checkpoint_missing_marker_skips(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()  # no .complete marker

    fn = CheckpointFnStub()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert fn.inputs == []
    assert controller_env.logged["skip"] == (5, "ckpt_missing")


async def test_eval_checkpoint_skip_reason_propagates(controller_env, tmp_path):
    """EvalSkip from any checkpoint backend becomes an attributable skipped point."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub(skip_reason="pin_violation")
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert controller_env.logged["skip"] == (5, "pin_violation")
    assert "eval" not in controller_env.logged


async def test_eval_shared_path_shape_unchanged(controller_env, monkeypatch):
    """No snapshot posture must keep today's shared-engine call shape: no snapshot
    fields threaded, no lag/duration metrics added."""
    seen_inputs = []

    def eval_generate_rollout(input):
        seen_inputs.append(input)
        return RolloutFnEvalOutput(data={})

    monkeypatch.setattr(rollout_manager_mod, "call_rollout_function", lambda fn, input: fn(input))
    args = make_args(hf_checkpoint="/base", eval_num_gpus=0)
    mgr = make_manager(args, eval_fn=eval_generate_rollout)

    await mgr.eval(5)

    assert len(seen_inputs) == 1
    assert seen_inputs[0].generate_state is None
    assert seen_inputs[0].weight_version is None
    assert seen_inputs[0].hf_dir is None
    _rollout_id, _data, extra = controller_env.logged["eval"]
    assert extra is None


# ---------------- driver (train_async.EvalDispatcher) ----------------


class FakeManagerActor:
    def __init__(self):
        self.eval_calls = []
        self.marker_flags = []
        self.skip_calls = []
        self._futures = []

        outer = self

        class _Eval:
            def remote(self, rollout_id, hf_dir=None, export_time_seconds=None, require_marker=True):
                outer.eval_calls.append((rollout_id, hf_dir, export_time_seconds))
                outer.marker_flags.append(require_marker)
                fut = asyncio.get_event_loop().create_future()
                outer._futures.append(fut)
                return fut

        class _Skip:
            def remote(self, rollout_id, reason):
                outer.skip_calls.append((rollout_id, reason))
                fut = asyncio.get_event_loop().create_future()
                fut.set_result(None)
                return fut

        self.eval = _Eval()
        self.report_eval_skip = _Skip()

    def finish(self, index=0):
        self._futures[index].set_result(None)


class FakeActorModel:
    def __init__(self, fail=False):
        self.exports = []
        self.fail = fail

    async def export_hf(self, rollout_id, path):
        if self.fail:
            raise RuntimeError("export boom")
        self.exports.append((rollout_id, path))


@pytest.fixture
def dispatcher_env(monkeypatch):
    import miles.ray.rollout.eval_dispatch as eval_dispatch

    # ray.wait/ray.get over asyncio futures: done iff the future is resolved.
    monkeypatch.setattr(
        eval_dispatch.ray, "wait", lambda refs, timeout=0: (refs, []) if refs[0].done() else ([], refs)
    )
    monkeypatch.setattr(eval_dispatch.ray, "get", lambda ref: ref.result())
    return eval_dispatch


def make_dispatcher(eval_dispatch, manager, actor_model, **arg_overrides):
    dispatcher_defaults = dict(
        eval_uses_snapshots=True,
        eval_hf_dir="/dev/shm/eval_hf",
        eval_max_in_flight=2,
        eval_overflow_policy="backpressure",
        eval_keep_snapshots=2,
        save_hf=None,
    )
    dispatcher_defaults.update(arg_overrides)
    args = make_args(**dispatcher_defaults)
    return eval_dispatch.EvalDispatcher(args, actor_model, manager), args


async def test_dispatcher_exports_and_fires(dispatcher_env):
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, actor_model)

    await dispatcher.dispatch(4)

    assert actor_model.exports == [(4, "/dev/shm/eval_hf/step_4")]
    assert len(manager.eval_calls) == 1
    rollout_id, hf_dir, export_time = manager.eval_calls[0]
    assert (rollout_id, hf_dir) == (4, "/dev/shm/eval_hf/step_4")
    assert export_time is not None
    assert len(dispatcher.pending) == 1


async def test_dispatcher_export_failure_skips(dispatcher_env):
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(fail=True))

    await dispatcher.dispatch(4)

    assert manager.eval_calls == []
    assert manager.skip_calls == [(4, "export_failed")]


async def test_dispatcher_skip_policy_drops_before_export(dispatcher_env):
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env, manager, actor_model, eval_max_in_flight=1, eval_overflow_policy="skip"
    )

    await dispatcher.dispatch(1)
    await dispatcher.dispatch(2)  # at cap: dropped, no export

    assert manager.skip_calls == [(2, "busy")]
    assert actor_model.exports == [(1, "/dev/shm/eval_hf/step_1")]


async def test_dispatcher_force_overrides_skip_policy(dispatcher_env):
    """The final eval point must never be dropped: training is already over."""
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env, manager, actor_model, eval_max_in_flight=1, eval_overflow_policy="skip"
    )

    await dispatcher.dispatch(1)

    async def finish_soon():
        await asyncio.sleep(0.01)
        manager.finish(0)

    finisher = asyncio.create_task(finish_soon())
    await dispatcher.dispatch(2, force=True)  # at cap: waits instead of dropping
    await finisher

    assert manager.skip_calls == []
    assert [c[0] for c in manager.eval_calls] == [1, 2]


async def test_dispatcher_backpressure_awaits_oldest(dispatcher_env):
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(), eval_max_in_flight=1)

    await dispatcher.dispatch(1)

    async def finish_soon():
        await asyncio.sleep(0.01)
        manager.finish(0)

    finisher = asyncio.create_task(finish_soon())
    await dispatcher.dispatch(2)  # must wait for eval 1 to finish
    await finisher

    assert [c[0] for c in manager.eval_calls] == [1, 2]
    assert len(dispatcher.pending) == 1  # only eval 2 pending


async def test_dispatcher_reuse_mode_uses_save_hf(dispatcher_env):
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env, manager, actor_model, eval_hf_dir=None, save_hf="/ckpt/hf/{rollout_id}"
    )

    await dispatcher.dispatch(10)

    assert actor_model.exports == []  # no extra export in reuse mode
    assert manager.eval_calls[0][:2] == (10, "/ckpt/hf/10")
    assert manager.marker_flags == [True]  # reused checkpoints still need their marker


async def test_dispatcher_caller_supplied_dir_skips_marker(dispatcher_env):
    """A caller-supplied hf_dir is an existing checkpoint (the pre-training baseline),
    not one of the dispatcher's exports: it never wrote a marker to check."""
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel())

    await dispatcher.dispatch(0, hf_dir="/base/hf_checkpoint")

    assert manager.eval_calls[0][:2] == (0, "/base/hf_checkpoint")
    assert manager.marker_flags == [False]


class TestSnapshotOwnership:
    """The dispatcher exports the snapshot, so the dispatcher deletes it — on every
    outcome, and only for dirs it exported itself."""

    def _make(self, dispatcher_env, tmp_path, **overrides):
        manager = FakeManagerActor()
        dispatcher, _ = make_dispatcher(
            dispatcher_env, manager, FakeActorModel(), eval_hf_dir=str(tmp_path), **overrides
        )
        return dispatcher, manager

    async def _dispatch_and_finish(self, dispatcher, manager, rollout_id, tmp_path, *, crash=False):
        snapshot = tmp_path / f"step_{rollout_id}"
        snapshot.mkdir()
        await dispatcher.dispatch(rollout_id)
        index = len(manager._futures) - 1
        if crash:
            manager._futures[index].set_exception(RuntimeError("eval boom"))
        else:
            manager.finish(index)
        await dispatcher.drain()
        return snapshot

    async def test_keeps_only_the_ring(self, dispatcher_env, tmp_path):
        dispatcher, manager = self._make(dispatcher_env, tmp_path, eval_keep_snapshots=2)

        dirs = [await self._dispatch_and_finish(dispatcher, manager, i, tmp_path) for i in (1, 2, 3)]

        assert not dirs[0].exists()
        assert dirs[1].exists() and dirs[2].exists()

    async def test_crashed_eval_still_retires_its_snapshot(self, dispatcher_env, tmp_path):
        """The leak this ownership move exists to prevent: a failed point used to
        keep its snapshot forever, filling the recommended tmpfs staging dir."""
        dispatcher, manager = self._make(dispatcher_env, tmp_path, eval_keep_snapshots=1)

        first = await self._dispatch_and_finish(dispatcher, manager, 1, tmp_path, crash=True)
        await self._dispatch_and_finish(dispatcher, manager, 2, tmp_path)

        assert manager.skip_calls == [(1, "crashed")]
        assert not first.exists()

    async def test_failed_export_leaves_nothing_behind(self, dispatcher_env, tmp_path):
        manager = FakeManagerActor()
        dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(fail=True), eval_hf_dir=str(tmp_path))
        partial = tmp_path / "step_4"
        partial.mkdir()

        await dispatcher.dispatch(4)

        assert manager.skip_calls == [(4, "export_failed")]
        assert not partial.exists()

    async def test_never_deletes_what_it_did_not_export(self, dispatcher_env, tmp_path):
        """--save-hf checkpoints and --hf-checkpoint are not the dispatcher's to delete."""
        save_hf = tmp_path / "save_hf"
        save_hf.mkdir()
        manager = FakeManagerActor()
        dispatcher, _ = make_dispatcher(
            dispatcher_env, manager, FakeActorModel(), eval_hf_dir=None, save_hf=str(save_hf), eval_keep_snapshots=0
        )

        await dispatcher.dispatch(1)
        manager.finish()
        await dispatcher.drain()

        assert save_hf.exists()
        assert dispatcher._exported == []


async def test_dispatcher_shared_engine_blocks_like_today(dispatcher_env):
    """No snapshot posture in args -> the plain blocking call, no dispatch machinery."""
    manager = FakeManagerActor()

    class _LegacyEval:
        def __init__(self):
            self.calls = []

        def remote(self, rollout_id):
            self.calls.append(rollout_id)
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

    manager.eval = _LegacyEval()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(), eval_uses_snapshots=False)

    await dispatcher.dispatch(3)
    assert manager.eval.calls == [3]
    assert len(dispatcher.pending) == 0


# ------- example fn (examples/infra_features/fully_async/external_eval_fn.py) -------


@pytest.fixture
def external_fn_env(monkeypatch):
    import importlib

    mod = importlib.import_module("examples.infra_features.fully_async.external_eval_fn")
    calls = []

    server = SimpleNamespace(loaded_version=None)

    async def fake_post(url, payload):
        calls.append(("post", url, payload))
        server.loaded_version = payload["weight_version"] if server.loaded_version != "stuck" else "stuck"

    async def fake_get(url):
        calls.append(("get", url))
        return {"weight_version": server.loaded_version}

    async def fake_run_eval(state, cache):
        calls.append(("eval", state))
        return {"ds": {"rewards": [1.0]}}

    async def fake_wait_ok(url, **kwargs):
        calls.append(("health", url))

    monkeypatch.setattr(mod, "post", fake_post)
    monkeypatch.setattr(mod, "get", fake_get)
    monkeypatch.setattr(mod, "run_eval_datasets", fake_run_eval)
    monkeypatch.setattr(mod, "GenerateState", lambda args: SimpleNamespace(args=args))
    monkeypatch.setattr(mod, "wait_http_ok", fake_wait_ok)
    for var in ("URL", "GPUS", "PORT", "SERVER_ARGS"):
        monkeypatch.delenv(f"MILES_EXTERNAL_EVAL_{var}", raising=False)
    return SimpleNamespace(mod=mod, calls=calls, server=server, monkeypatch=monkeypatch)


def make_external_fn(external_fn_env, **env):
    from miles.rollout.base_types import RolloutFnConstructorInput

    for var, value in env.items():
        external_fn_env.monkeypatch.setenv(f"MILES_EXTERNAL_EVAL_{var}", value)
    args = make_args(hf_checkpoint="/base")
    return external_fn_env.mod.ExternalSglangEvalFn(RolloutFnConstructorInput(args=args, data_source=None))


async def test_external_eval_fn_waits_pins_then_evals(external_fn_env):
    fn = make_external_fn(external_fn_env, URL="http://eval-host:31000")

    output = await fn(RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/snap/step_5"))

    assert external_fn_env.calls[0] == ("health", "http://eval-host:31000/health_generate")
    assert external_fn_env.calls[1] == (
        "post",
        "http://eval-host:31000/update_weights_from_disk",
        {"model_path": "/snap/step_5", "weight_version": "5"},
    )
    assert external_fn_env.calls[2] == ("get", "http://eval-host:31000/model_info")
    assert external_fn_env.calls[3][0] == "eval"
    # The eval state targets the external server, built from the real training args.
    state = external_fn_env.calls[3][1]
    assert (state.args.sglang_router_ip, state.args.sglang_router_port) == ("eval-host", 31000)
    assert output.data == {"ds": {"rewards": [1.0]}}


async def test_external_eval_fn_pin_failure_retries_then_raises(external_fn_env):
    fn = make_external_fn(external_fn_env, URL="http://eval-host:31000")
    external_fn_env.server.loaded_version = "stuck"  # server never reports the pinned version

    with pytest.raises(RuntimeError, match="pin failed"):
        await fn(RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/snap/step_5"))

    assert len([c for c in external_fn_env.calls if c[0] == "post"]) == 2  # one retry
    assert not [c for c in external_fn_env.calls if c[0] == "eval"]


def test_external_eval_fn_launches_own_server(external_fn_env, monkeypatch):
    """Launch mode is the black-box promise: init prepares everything, pinned to
    the GPUs the user names, extra sglang flags passed through; dispose tears down."""
    procs = []

    def fake_popen(cmd, env=None):
        procs.append(SimpleNamespace(cmd=cmd, env=env, terminated=False))
        procs[-1].terminate = lambda p=procs[-1]: setattr(p, "terminated", True)
        return procs[-1]

    monkeypatch.setattr(external_fn_env.mod.subprocess, "Popen", fake_popen)

    fn = make_external_fn(external_fn_env, GPUS="6,7", SERVER_ARGS="--attention-backend fa3")

    (proc,) = procs
    assert proc.env["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert proc.cmd[proc.cmd.index("--tp") + 1] == "2"
    assert proc.cmd[proc.cmd.index("--model-path") + 1] == "/base"
    assert proc.cmd[-2:] == ["--attention-backend", "fa3"]
    assert fn._url == "http://127.0.0.1:31000"
    fn.dispose()
    assert proc.terminated
