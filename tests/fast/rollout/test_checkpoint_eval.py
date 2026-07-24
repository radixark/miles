from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
import miles.rollout.checkpoint_eval as checkpoint_eval_mod
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.checkpoint_eval import (
    CheckpointEvalFn,
    EvalSkip,
    FleetEvalFn,
    make_eval_args,
    resolve_checkpoint_eval_fn,
    retarget_args,
)


def make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_router_ip="10.0.0.1",
        sglang_router_port=30000,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
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


def test_make_eval_args_reads_router_registry():
    args = make_args()
    eval_args = make_eval_args(args)

    assert (eval_args.sglang_router_ip, eval_args.sglang_router_port) == ("10.0.0.2", 31000)
    assert eval_args.rollout_num_gpus == args.eval_num_gpus
    assert eval_args.rollout_num_gpus_per_engine == args.eval_num_gpus_per_engine


async def test_run_eval_datasets_merges_datasets(monkeypatch):
    import miles.rollout.inference_rollout.inference_rollout_eval as eval_mod

    async def fake_single_dataset(state, cfg, cache):
        return {cfg.name: {"rewards": [1.0], "truncated": [False], "samples": []}}

    monkeypatch.setattr(eval_mod, "eval_rollout_single_dataset", fake_single_dataset)

    state = SimpleNamespace(
        args=Namespace(group_rm=False, eval_datasets=[SimpleNamespace(name="a"), SimpleNamespace(name="b")])
    )
    results = await eval_mod.run_eval_datasets(state, {})
    assert set(results.keys()) == {"a", "b"}


def _eval_dataset_env(monkeypatch, generate):
    import miles.rollout.inference_rollout.inference_rollout_eval as eval_mod
    from miles.utils.types import Sample

    monkeypatch.setattr(eval_mod, "generate_and_rm", generate)
    monkeypatch.setattr(eval_mod, "compute_sampling_params", lambda args, **kw: {})
    args = Namespace(
        group_rm=False,
        hf_checkpoint="hf",
        apply_chat_template=False,
        chat_template_path=None,
        reward_key=None,
        eval_reward_key=None,
    )
    dataset_cfg = SimpleNamespace(
        name="ds",
        cache_key=("ds",),
        n_samples_per_eval_prompt=1,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_response_len=16,
        inject_metadata=lambda md: md,
    )
    samples = [Sample(index=i, prompt="p", response="r", label="l", reward=1) for i in range(4)]
    cache = {dataset_cfg.cache_key + ("hf", False, None): SimpleNamespace(samples=samples)}
    return eval_mod, args, dataset_cfg, cache


async def test_single_dataset_tolerates_partial_failures(monkeypatch):
    from miles.utils.types import Sample

    async def generate(state, sample, sampling_params, evaluation):
        if sample.index == 0:
            raise RuntimeError("engine died")
        if sample.index == 1:
            sample.status = Sample.Status.ABORTED
        return sample

    eval_mod, args, dataset_cfg, cache = _eval_dataset_env(monkeypatch, generate)
    result = await eval_mod.eval_rollout_single_dataset(SimpleNamespace(args=args), dataset_cfg, cache)

    assert result["ds"]["rewards"] == [1, 1]
    assert result["ds"]["failed_samples"] == 2  # one raised + one aborted


async def test_single_dataset_all_failures_raise(monkeypatch):
    async def generate(state, sample, sampling_params, evaluation):
        raise RuntimeError("engine died")

    eval_mod, args, dataset_cfg, cache = _eval_dataset_env(monkeypatch, generate)
    with pytest.raises(RuntimeError, match="all 4 sample generations failed"):
        await eval_mod.eval_rollout_single_dataset(SimpleNamespace(args=args), dataset_cfg, cache)


# ---------------- FleetEvalFn (the dedicated fleet as a checkpoint backend) ----------------


class FakeRemoteMethod:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args, **kwargs):
        self.engine.log.append((self.name, args, kwargs))
        result = self.engine.responses[self.name](*args, **kwargs)
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(result)
        return fut


class FakeEngine:
    def __init__(self, log):
        self.log = log
        self.weight_version = None

        def load(model_path, weight_version=None):
            self.weight_version = weight_version
            return None

        self.responses = {
            "update_weights_from_disk": load,
            "get_weight_version": lambda: self.weight_version,
        }

    def __getattr__(self, name):
        if name in ("update_weights_from_disk", "get_weight_version"):
            return FakeRemoteMethod(self, name)
        raise AttributeError(name)


class FakeServerEngineWrapper:
    def __init__(self, actor):
        self._actor = actor
        self.is_allocated = True
        self.stopped = False

    @property
    def actor_handle(self):
        return self._actor

    def mark_stopped(self):
        self.stopped = True
        self.is_allocated = False


class FakeEvalServer:
    def __init__(self, engines):
        self._engines = engines
        self.wrappers = [FakeServerEngineWrapper(e) for e in engines]
        self.recover_calls = 0

    @property
    def server_groups(self):
        return [SimpleNamespace(all_engines=self.wrappers)]

    @property
    def engines(self):
        return [SimpleNamespace(actor_handle=e) for e in self._engines]

    async def recover(self):
        self.recover_calls += 1

    async def wait_all_engines_alive(self):
        pass


@pytest.fixture
def fleet_env(monkeypatch):
    state_builds = []

    async def noop_router_ready(self, timeout=180.0):
        return None

    monkeypatch.setattr(checkpoint_eval_mod.FleetEvalFn, "_wait_router_ready", noop_router_ready)
    monkeypatch.setattr(
        checkpoint_eval_mod,
        "make_eval_generate_state",
        lambda args: state_builds.append(args) or "fake-fleet-state",
    )
    return SimpleNamespace(state_builds=state_builds)


def make_fleet_fn(args, engines, inner=None):
    inner = inner or (lambda input: RolloutFnEvalOutput(data={}))
    return FleetEvalFn(args, FakeEvalServer(engines), inner=inner)


def eval_input(rollout_id, hf_dir):
    return RolloutFnEvalInput(rollout_id=rollout_id, weight_version=str(rollout_id), hf_dir=hf_dir)


async def test_fleet_pins_all_engines_then_delegates(fleet_env):
    log = []
    engines = [FakeEngine(log), FakeEngine(log)]
    seen_inputs = []

    def inner(input):
        log.append(("generate", input.rollout_id))
        seen_inputs.append(input)
        return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})

    fn = make_fleet_fn(make_args(), engines, inner=inner)

    await fn.evaluate_checkpoint("/snap/step_5", eval_input(5, "/snap/step_5"))

    load_events = [e for e in log if e[0] == "update_weights_from_disk"]
    assert len(load_events) == 2
    assert all(e[2]["weight_version"] == "5" for e in load_events)
    # Every load strictly precedes delegation to the inner fn.
    assert log.index(("generate", 5)) > max(i for i, e in enumerate(log) if e[0] == "update_weights_from_disk")
    # The inner fn gets the fleet's state; RolloutFn classes stay unaware of the fleet.
    assert seen_inputs[0].generate_state == "fake-fleet-state"
    assert seen_inputs[0].weight_version == "5"
    assert seen_inputs[0].hf_dir == "/snap/step_5"

    # The fleet state is built lazily on first use, then cached.
    await fn.evaluate_checkpoint("/snap/step_6", eval_input(6, "/snap/step_6"))
    assert len(fleet_env.state_builds) == 1


async def test_fleet_pin_violation_skips_after_retry(fleet_env):
    log = []
    engine = FakeEngine(log)
    engine.responses["get_weight_version"] = lambda: "999"  # never matches
    fn = make_fleet_fn(make_args(), [engine])

    with pytest.raises(EvalSkip) as exc:
        await fn.evaluate_checkpoint("/snap/step_5", eval_input(5, "/snap/step_5"))

    assert exc.value.reason == "pin_violation"
    assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 2  # one retry


async def test_fleet_pin_requires_all_engines_to_match(fleet_env):
    """The router load-balances across engines: one stale engine = mixed versions."""
    log = []
    good, stale = FakeEngine(log), FakeEngine(log)
    stale.responses["get_weight_version"] = lambda: "999"
    fn = make_fleet_fn(make_args(), [good, stale])

    with pytest.raises(EvalSkip) as exc:
        await fn.evaluate_checkpoint("/snap/step_5", eval_input(5, "/snap/step_5"))

    assert exc.value.reason == "pin_violation"


async def test_fleet_marks_dead_engine_for_recovery(fleet_env):
    """A dead actor is marked stopped for revival; the eval degrades to a skip."""
    log = []
    engine = FakeEngine(log)

    def dead(*args, **kwargs):
        raise RuntimeError("actor died")

    engine.responses["get_weight_version"] = dead
    fn = make_fleet_fn(make_args(), [engine])

    with pytest.raises(EvalSkip):
        await fn.evaluate_checkpoint("/snap/step_5", eval_input(5, "/snap/step_5"))

    assert fn._srv.wrappers[0].stopped  # probed, found unreachable, marked for revival
    assert fn._srv.recover_calls == 1


async def test_fleet_router_not_ready_skips(fleet_env, monkeypatch):
    async def router_never_ready(self, timeout=180.0):
        raise TimeoutError("router not ready")

    monkeypatch.setattr(checkpoint_eval_mod.FleetEvalFn, "_wait_router_ready", router_never_ready)

    inner_calls = []
    fn = make_fleet_fn(make_args(), [FakeEngine([])], inner=lambda input: inner_calls.append(input))

    with pytest.raises(EvalSkip) as exc:
        await fn.evaluate_checkpoint("/snap/step_5", eval_input(5, "/snap/step_5"))

    assert exc.value.reason == "unhealthy"
    assert inner_calls == []


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


def make_manager(args, checkpoint_fn=None):
    mgr = object.__new__(rollout_manager_mod.RolloutManager.__ray_actor_class__)
    mgr.args = args
    mgr.rollout_id = 7
    mgr._eval_lock = asyncio.Lock()
    mgr._eval_consumed_snapshots = []
    mgr._health_monitors = []
    mgr.use_experimental_refactor = True
    mgr._metric_checker = None
    mgr._checkpoint_fn = checkpoint_fn
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
    mgr = make_manager(args, checkpoint_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot), export_time_seconds=1.5)

    assert len(fn.inputs) == 1
    assert fn.inputs[0].hf_dir == str(snapshot)
    assert fn.inputs[0].weight_version == "5"
    assert fn.inputs[0].generate_state is None
    rollout_id, _data, extra = controller_env.logged["eval"]
    assert rollout_id == 5
    assert extra["eval/lag_steps"] == 2
    assert extra["eval/export_time_seconds"] == 1.5


async def test_eval_checkpoint_missing_marker_skips(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()  # no .complete marker

    fn = CheckpointFnStub()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, checkpoint_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert fn.inputs == []
    assert controller_env.logged["skip"] == (5, "ckpt_missing")


async def test_eval_checkpoint_base_checkpoint_needs_no_marker(controller_env, tmp_path):
    fn = CheckpointFnStub()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, checkpoint_fn=fn)

    await mgr.eval(0, hf_dir="/base")

    assert len(fn.inputs) == 1
    assert "eval" in controller_env.logged


async def test_eval_checkpoint_skip_reason_propagates(controller_env, tmp_path):
    """EvalSkip from any checkpoint backend becomes an attributable skipped point."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub(skip_reason="pin_violation")
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, checkpoint_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert controller_env.logged["skip"] == (5, "pin_violation")
    assert "eval" not in controller_env.logged
    assert snapshot.exists()  # a skipped snapshot is not consumed


async def test_eval_checkpoint_gc_keeps_ring(controller_env, tmp_path):
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, checkpoint_fn=CheckpointFnStub())

    dirs = []
    for rollout_id in (1, 2, 3):
        snapshot = tmp_path / f"step_{rollout_id}"
        snapshot.mkdir()
        (snapshot / ".complete").touch()
        dirs.append(snapshot)
        await mgr.eval(rollout_id, hf_dir=str(snapshot))

    assert not dirs[0].exists()  # oldest consumed snapshot beyond keep-2 is deleted
    assert dirs[1].exists() and dirs[2].exists()


async def test_eval_checkpoint_never_deletes_outside_staging(controller_env, tmp_path):
    save_hf = tmp_path / "save_hf" / "step_1"
    save_hf.mkdir(parents=True)
    (save_hf / ".complete").touch()
    staging = tmp_path / "staging"
    staging.mkdir()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(staging), eval_keep_snapshots=2)
    mgr = make_manager(args, checkpoint_fn=CheckpointFnStub())

    await mgr.eval(1, hf_dir=str(save_hf))

    assert save_hf.exists()
    assert mgr._eval_consumed_snapshots == []


async def test_eval_shared_path_shape_unchanged(controller_env, monkeypatch):
    """No checkpoint fn resolved must keep today's shared-engine call shape: no
    snapshot fields threaded, no lag/duration metrics added."""
    seen_inputs = []

    def eval_generate_rollout(input):
        seen_inputs.append(input)
        return RolloutFnEvalOutput(data={})

    monkeypatch.setattr(rollout_manager_mod, "call_rollout_function", lambda fn, input: fn(input))
    args = make_args(hf_checkpoint="/base", eval_num_gpus=0)
    mgr = make_manager(args, checkpoint_fn=None)
    mgr.eval_generate_rollout = eval_generate_rollout

    await mgr.eval(5)

    assert len(seen_inputs) == 1
    assert seen_inputs[0].generate_state is None
    assert seen_inputs[0].weight_version is None
    assert seen_inputs[0].hf_dir is None
    _rollout_id, _data, extra = controller_env.logged["eval"]
    assert extra is None


def test_resolve_checkpoint_eval_fn():
    """The single discrimination point: fleet flag wins, then CheckpointEvalFn
    instances (validated), else shared."""
    plain_fn = lambda input: None  # noqa: E731

    fleet = resolve_checkpoint_eval_fn(make_args(eval_num_gpus=1), plain_fn, {"eval": "srv-handle"})
    assert isinstance(fleet, FleetEvalFn)
    assert fleet._inner is plain_fn

    external = CheckpointFnStub()
    args = make_args(
        eval_num_gpus=0, eval_hf_dir="/staging", save_hf=None, eval_keep_snapshots=2, eval_max_in_flight=2
    )
    assert resolve_checkpoint_eval_fn(args, external, {}) is external

    with pytest.raises(AssertionError, match="snapshot source"):
        resolve_checkpoint_eval_fn(make_args(eval_num_gpus=0, eval_hf_dir=None, save_hf=None), external, {})

    assert resolve_checkpoint_eval_fn(make_args(eval_num_gpus=0), plain_fn, {}) is None

    with pytest.raises(AssertionError, match="class-based"):
        # Legacy (non-class) loading leaves the class unconstructed.
        resolve_checkpoint_eval_fn(make_args(eval_num_gpus=0), CheckpointFnStub, {})


# ---------------- driver (train_async.EvalDispatcher) ----------------


class FakeManagerActor:
    def __init__(self, snapshot_eval=True):
        self.eval_calls = []
        self.skip_calls = []
        self._futures = []

        outer = self

        class _Eval:
            def remote(self, rollout_id, hf_dir=None, export_time_seconds=None):
                outer.eval_calls.append((rollout_id, hf_dir, export_time_seconds))
                fut = asyncio.get_event_loop().create_future()
                outer._futures.append(fut)
                return fut

        class _Skip:
            def remote(self, rollout_id, reason):
                outer.skip_calls.append((rollout_id, reason))
                fut = asyncio.get_event_loop().create_future()
                fut.set_result(None)
                return fut

        class _UsesSnapshots:
            def remote(self):
                fut = asyncio.get_event_loop().create_future()
                fut.set_result(snapshot_eval)
                return fut

        self.eval = _Eval()
        self.report_eval_skip = _Skip()
        self.eval_uses_snapshots = _UsesSnapshots()

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
        eval_hf_dir="/dev/shm/eval_hf",
        eval_dispatch="async",
        eval_max_in_flight=2,
        eval_overflow_policy="backpressure",
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


async def test_dispatcher_drain_awaits_all(dispatcher_env):
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel())

    await dispatcher.dispatch(1)
    await dispatcher.dispatch(2)
    assert len(dispatcher.pending) == 2

    manager.finish(0)
    manager.finish(1)
    await dispatcher.drain()
    assert len(dispatcher.pending) == 0


async def test_dispatcher_shared_engine_blocks_like_today(dispatcher_env):
    """The manager is the single authority: when it reports no snapshot posture,
    dispatch degrades to the plain blocking call."""
    manager = FakeManagerActor(snapshot_eval=False)

    class _LegacyEval:
        def __init__(self):
            self.calls = []

        def remote(self, rollout_id):
            self.calls.append(rollout_id)
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

    manager.eval = _LegacyEval()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel())

    await dispatcher.dispatch(3)
    assert manager.eval.calls == [3]
    assert len(dispatcher.pending) == 0


# ---------------- example fn (examples/fully_async/external_eval_fn.py) ----------------


@pytest.fixture
def external_fn_env(monkeypatch):
    import importlib

    mod = importlib.import_module("examples.fully_async.external_eval_fn")
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
    args = make_args(hf_checkpoint="/base", eval_model_path=None)
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
