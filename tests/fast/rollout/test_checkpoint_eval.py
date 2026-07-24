from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
from miles.rollout.checkpoint_eval import EvalFleetSession, make_eval_args, retarget_args


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


def test_eval_fleet_session_builds_state_once_and_caches(monkeypatch):
    """RolloutManager owns one EvalFleetSession and calls .state() on every eval;
    the router only exists once servers are up, so the state must be built lazily
    on first use, not eagerly in __init__, and reused after that."""
    import miles.rollout.checkpoint_eval as checkpoint_eval

    build_calls = []

    def fake_make_eval_generate_state(args):
        build_calls.append(args)
        return SimpleNamespace(args=args)

    monkeypatch.setattr(checkpoint_eval, "make_eval_generate_state", fake_make_eval_generate_state)

    session = EvalFleetSession(make_args())
    assert build_calls == []  # not built in __init__

    state = session.state()
    assert len(build_calls) == 1
    assert session.state() is state  # cached, not rebuilt
    assert len(build_calls) == 1


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


# ---------------- controller (RolloutManager._eval_on_dedicated_fleet) ----------------


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


def make_manager(args, engines, eval_fn_result=None):
    mgr = object.__new__(rollout_manager_mod.RolloutManager.__ray_actor_class__)
    mgr.args = args
    mgr.rollout_id = 7
    mgr._eval_lock = asyncio.Lock()
    mgr._eval_consumed_snapshots = []
    mgr._health_monitors = []
    mgr.use_experimental_refactor = True
    mgr.servers = {"eval": FakeEvalServer(engines)}
    mgr._metric_checker = None
    mgr._eval_fleet = SimpleNamespace(state=lambda: "fake-fleet-state")
    # eval_generate_rollout ignores its RolloutFnEvalInput; the fleet's generate_state
    # is opaque to this fake, only real InferenceRolloutFn/FullyAsyncRolloutFn use it.
    mgr.eval_generate_rollout = lambda input: eval_fn_result
    return mgr


@pytest.fixture
def controller_env(monkeypatch, tmp_path):
    log = []
    logged = {}

    def fake_call_rollout_function(fn, input):
        log.append(("generate", input.rollout_id))
        return fn(input)

    monkeypatch.setattr(rollout_manager_mod, "call_rollout_function", fake_call_rollout_function)

    async def noop_router_ready(self, srv, timeout=180.0):
        return None

    monkeypatch.setattr(
        rollout_manager_mod.RolloutManager.__ray_actor_class__, "_wait_eval_router_ready", noop_router_ready
    )
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
    return SimpleNamespace(log=log, logged=logged, tmp_path=tmp_path)


async def test_controller_pins_all_engines_before_generate(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    log = controller_env.log
    engines = [FakeEngine(log), FakeEngine(log)]
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    result = SimpleNamespace(data={"ds": {"rewards": [1.0]}}, metrics=None)
    mgr = make_manager(args, engines, eval_fn_result=result)

    await mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=1.5)

    load_events = [e for e in log if e[0] == "update_weights_from_disk"]
    assert len(load_events) == 2
    assert all(e[2]["weight_version"] == "5" for e in load_events)
    # Every load strictly precedes generation.
    assert log.index(("generate", 5)) > max(i for i, e in enumerate(log) if e[0] == "update_weights_from_disk")
    rollout_id, _data, extra = controller_env.logged["eval"]
    assert rollout_id == 5
    assert extra["eval/lag_steps"] == 2
    assert extra["eval/export_time_seconds"] == 1.5


async def test_controller_threads_fleet_state_and_weight_version(controller_env, tmp_path):
    """RolloutManager is the single place that decides fleet-vs-shared: it must pass
    the fleet's GenerateState and the pinned weight_version into RolloutFnEvalInput so
    RolloutFn classes stay unaware of --eval-num-gpus."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    seen_inputs = []

    def eval_generate_rollout(input):
        seen_inputs.append(input)
        return SimpleNamespace(data={}, metrics=None)

    engines = [FakeEngine(controller_env.log)]
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, engines)
    mgr.eval_generate_rollout = eval_generate_rollout

    await mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=None)

    assert len(seen_inputs) == 1
    assert seen_inputs[0].generate_state == "fake-fleet-state"
    assert seen_inputs[0].weight_version == "5"
    assert seen_inputs[0].hf_dir == str(snapshot)


async def test_controller_skips_on_missing_marker(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()  # no .complete marker

    engines = [FakeEngine(controller_env.log)]
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, engines)

    await mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=None)

    assert controller_env.logged["skip"] == (5, "ckpt_missing")
    assert not [e for e in controller_env.log if e[0] == "update_weights_from_disk"]


async def test_controller_base_checkpoint_needs_no_marker(controller_env, tmp_path):
    engines = [FakeEngine(controller_env.log)]
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    result = SimpleNamespace(data={}, metrics=None)
    mgr = make_manager(args, engines, eval_fn_result=result)

    await mgr._eval_on_dedicated_fleet(0, "/base", export_time_seconds=None)

    assert "eval" in controller_env.logged
    load_events = [e for e in controller_env.log if e[0] == "update_weights_from_disk"]
    assert len(load_events) == 1 and load_events[0][2]["weight_version"] == "0"


async def test_controller_pin_violation_skips_after_retry(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    engine = FakeEngine(controller_env.log)
    engine.responses["get_weight_version"] = lambda: "999"  # never matches
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, [engine])

    await mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=None)

    assert controller_env.logged["skip"] == (5, "pin_violation")
    load_events = [e for e in controller_env.log if e[0] == "update_weights_from_disk"]
    assert len(load_events) == 2  # one retry
    assert ("generate", 5) not in controller_env.log


async def test_controller_zombie_engine_times_out_to_skip(controller_env, tmp_path, monkeypatch):
    """A zombie engine never answers; the load timeout must skip the point
    instead of holding the eval lock forever."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    engine = FakeEngine(controller_env.log)

    class _NeverResolves:
        def remote(self, *args, **kwargs):
            return asyncio.get_event_loop().create_future()  # never resolved

    engine.update_weights_from_disk_override = _NeverResolves()
    monkeypatch.setattr(
        type(engine),
        "__getattr__",
        lambda self, name: (
            self.update_weights_from_disk_override
            if name == "update_weights_from_disk"
            else FakeRemoteMethod(self, name)
        ),
    )
    monkeypatch.setattr(rollout_manager_mod, "EVAL_WEIGHT_LOAD_TIMEOUT_SECS", 0.05)

    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, [engine])

    await asyncio.wait_for(mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=None), timeout=5)

    assert controller_env.logged["skip"] == (5, "pin_violation")
    assert not mgr._eval_lock.locked()


async def test_controller_marks_dead_engine_for_recovery(controller_env, tmp_path):
    """A dead actor is marked stopped for revival; the eval degrades to a skip."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    engine = FakeEngine(controller_env.log)

    def dead(*args, **kwargs):
        raise RuntimeError("actor died")

    engine.responses["get_weight_version"] = dead
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, [engine])

    await mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=None)

    srv = mgr.servers["eval"]
    assert srv.wrappers[0].stopped  # probed, found unreachable, marked for revival
    assert srv.recover_calls == 1
    assert controller_env.logged["skip"] == (5, "pin_violation")


async def test_controller_router_not_ready_skips(controller_env, tmp_path, monkeypatch):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    async def router_never_ready(self, srv, timeout=180.0):
        raise TimeoutError("router not ready")

    monkeypatch.setattr(
        rollout_manager_mod.RolloutManager.__ray_actor_class__, "_wait_eval_router_ready", router_never_ready
    )

    engines = [FakeEngine(controller_env.log)]
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, engines)

    await mgr._eval_on_dedicated_fleet(5, str(snapshot), export_time_seconds=None)

    assert controller_env.logged["skip"] == (5, "unhealthy")
    assert ("generate", 5) not in controller_env.log


async def test_controller_gc_keeps_ring(controller_env, tmp_path):
    engines = [FakeEngine(controller_env.log)]
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    result = SimpleNamespace(data={}, metrics=None)
    mgr = make_manager(args, engines, eval_fn_result=result)

    dirs = []
    for rollout_id in (1, 2, 3):
        snapshot = tmp_path / f"step_{rollout_id}"
        snapshot.mkdir()
        (snapshot / ".complete").touch()
        dirs.append(snapshot)
        controller_env.logged.pop("eval", None)
        await mgr._eval_on_dedicated_fleet(rollout_id, str(snapshot), export_time_seconds=None)

    assert not dirs[0].exists()  # oldest consumed snapshot beyond keep-2 is deleted
    assert dirs[1].exists() and dirs[2].exists()


async def test_controller_never_deletes_outside_staging(controller_env, tmp_path):
    engines = [FakeEngine(controller_env.log)]
    save_hf = tmp_path / "save_hf" / "step_1"
    save_hf.mkdir(parents=True)
    (save_hf / ".complete").touch()
    staging = tmp_path / "staging"
    staging.mkdir()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(staging), eval_keep_snapshots=2)
    result = SimpleNamespace(data={}, metrics=None)
    mgr = make_manager(args, engines, eval_fn_result=result)

    await mgr._eval_on_dedicated_fleet(1, str(save_hf), export_time_seconds=None)

    assert save_hf.exists()
    assert mgr._eval_consumed_snapshots == []


# ---------------- external eval fn (eval_needs_snapshot) ----------------


class ExternalFnStub:
    eval_needs_snapshot = True

    def __init__(self):
        self.inputs = []

    def __call__(self, input):
        self.inputs.append(input)
        return SimpleNamespace(data={"ds": {"rewards": [1.0]}}, metrics=None)


async def test_eval_routes_external_fn_without_fleet_machinery(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    engines = [FakeEngine(controller_env.log)]
    args = make_args(hf_checkpoint="/base", eval_num_gpus=0, eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, engines)
    fn = ExternalFnStub()
    mgr.eval_generate_rollout = fn

    await mgr.eval(5, hf_dir=str(snapshot), export_time_seconds=1.5)

    # The fn owns weight delivery: no fleet pin, no recover, no fleet state threaded.
    assert not [e for e in controller_env.log if e[0] == "update_weights_from_disk"]
    assert mgr.servers["eval"].recover_calls == 0
    assert len(fn.inputs) == 1
    assert fn.inputs[0].generate_state is None
    assert fn.inputs[0].weight_version == "5"
    assert fn.inputs[0].hf_dir == str(snapshot)
    rollout_id, _data, extra = controller_env.logged["eval"]
    assert rollout_id == 5
    assert extra["eval/lag_steps"] == 2
    assert extra["eval/export_time_seconds"] == 1.5


async def test_eval_external_fn_missing_marker_skips(controller_env, tmp_path):
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()  # no .complete marker

    args = make_args(hf_checkpoint="/base", eval_num_gpus=0, eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, [])
    fn = ExternalFnStub()
    mgr.eval_generate_rollout = fn

    await mgr.eval(5, hf_dir=str(snapshot))

    assert fn.inputs == []
    assert controller_env.logged["skip"] == (5, "ckpt_missing")


async def test_eval_external_fn_gc_keeps_ring(controller_env, tmp_path):
    args = make_args(hf_checkpoint="/base", eval_num_gpus=0, eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, [])
    mgr.eval_generate_rollout = ExternalFnStub()

    dirs = []
    for rollout_id in (1, 2, 3):
        snapshot = tmp_path / f"step_{rollout_id}"
        snapshot.mkdir()
        (snapshot / ".complete").touch()
        dirs.append(snapshot)
        await mgr.eval(rollout_id, hf_dir=str(snapshot))

    assert not dirs[0].exists()
    assert dirs[1].exists() and dirs[2].exists()


async def test_eval_shared_path_shape_unchanged(controller_env):
    """eval_num_gpus=0 + a fn without eval_needs_snapshot must keep today's shared-engine
    call shape: no snapshot fields threaded, no lag/duration metrics added."""
    seen_inputs = []

    def eval_generate_rollout(input):
        seen_inputs.append(input)
        return SimpleNamespace(data={}, metrics=None)

    args = make_args(hf_checkpoint="/base", eval_num_gpus=0)
    mgr = make_manager(args, [])
    mgr.eval_generate_rollout = eval_generate_rollout

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


async def test_dispatcher_without_fleet_blocks_like_today(dispatcher_env):
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
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(), eval_num_gpus=0)

    await dispatcher.dispatch(3)
    assert manager.eval.calls == [3]
    assert len(dispatcher.pending) == 0


EXTERNAL_FN_PATH = "tests.fast.rollout.test_checkpoint_eval.ExternalFnStub"


async def test_dispatcher_external_fn_exports_like_fleet(dispatcher_env, monkeypatch):
    """eval_needs_snapshot without a fleet must still get the snapshot + async dispatch."""
    monkeypatch.setenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "1")
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env,
        manager,
        actor_model,
        eval_num_gpus=0,
        eval_function_path=EXTERNAL_FN_PATH,
        eval_keep_snapshots=2,
    )

    await dispatcher.dispatch(4)

    assert actor_model.exports == [(4, "/dev/shm/eval_hf/step_4")]
    assert len(manager.eval_calls) == 1
    rollout_id, hf_dir, export_time = manager.eval_calls[0]
    assert (rollout_id, hf_dir) == (4, "/dev/shm/eval_hf/step_4")
    assert export_time is not None
    assert len(dispatcher.pending) == 1


async def test_dispatcher_rejects_fleet_plus_external_fn(dispatcher_env, monkeypatch):
    monkeypatch.setenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "1")
    with pytest.raises(AssertionError, match="eval_needs_snapshot"):
        make_dispatcher(
            dispatcher_env,
            FakeManagerActor(),
            FakeActorModel(),
            eval_num_gpus=1,
            eval_function_path=EXTERNAL_FN_PATH,
        )


# ------- standalone service (examples/fully_async/checkpoint_eval_service.py) -------


@pytest.fixture
def service_mod():
    import importlib

    return importlib.import_module("examples.fully_async.checkpoint_eval_service")


def test_find_ready_snapshots(service_mod, tmp_path):
    from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER

    def snapshot(name, *, marker=False, config=False, old=False):
        d = tmp_path / name
        d.mkdir()
        if marker:
            (d / HF_EXPORT_COMPLETE_MARKER).touch()
        if config:
            (d / "config.json").touch()
            (d / "model.safetensors").touch()
        if old:
            import os

            stale = time.time() - 2 * service_mod.QUIESCENCE_SECS
            for p in [d] + list(d.iterdir()):
                os.utime(p, (stale, stale))
        return d

    import time

    snapshot("step_3", marker=True)
    snapshot("step_7", config=True, old=True)  # pre-marker, quiescent
    snapshot("step_9", config=True)  # still being written
    snapshot("qwen2.5-step_12", marker=True)  # id must come from the trailing number
    snapshot("step_1", marker=True)  # below min_rollout_id
    snapshot("step_5", marker=True)  # already consumed
    (tmp_path / "notes.txt").touch()

    ready = service_mod.find_ready_snapshots(tmp_path, min_rollout_id=2, consumed={5})

    assert [(rid, p.name) for rid, p in ready] == [(3, "step_3"), (7, "step_7"), (12, "qwen2.5-step_12")]


def test_snapshot_ledger_roundtrip(service_mod, tmp_path):
    ledger = service_mod.SnapshotLedger(tmp_path)
    ledger.mark(3)
    ledger.mark(7)

    assert service_mod.SnapshotLedger(tmp_path).consumed == {3, 7}


# ---------------- example fn (examples/fully_async/external_eval_fn.py) ----------------


@pytest.fixture
def external_fn_env(monkeypatch):
    import importlib

    mod = importlib.import_module("examples.fully_async.external_eval_fn")
    calls = []

    async def fake_pin(targets, hf_dir, weight_version):
        calls.append(("pin", targets[0]._url, hf_dir, weight_version))
        return not getattr(fake_pin, "fail", False)

    async def fake_run_eval(state, cache):
        calls.append(("eval", state))
        return {"ds": {"rewards": [1.0]}}

    monkeypatch.setattr(mod, "pin_and_verify", fake_pin)
    monkeypatch.setattr(mod, "run_eval_datasets", fake_run_eval)
    monkeypatch.setattr(mod, "GenerateState", lambda args: SimpleNamespace(args=args))
    monkeypatch.setenv("MILES_EXTERNAL_EVAL_URL", "http://eval-host:31000")
    return SimpleNamespace(mod=mod, calls=calls, fake_pin=fake_pin)


async def test_external_eval_fn_pins_then_evals(external_fn_env):
    from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput

    args = make_args(hf_checkpoint="/base")
    fn = external_fn_env.mod.ExternalSglangEvalFn(RolloutFnConstructorInput(args=args, data_source=None))

    output = await fn(RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/snap/step_5"))

    assert external_fn_env.calls[0] == ("pin", "http://eval-host:31000", "/snap/step_5", "5")
    assert external_fn_env.calls[1][0] == "eval"
    # The eval state targets the external server, built from the real training args.
    state = external_fn_env.calls[1][1]
    assert (state.args.sglang_router_ip, state.args.sglang_router_port) == ("eval-host", 31000)
    assert output.data == {"ds": {"rewards": [1.0]}}


async def test_external_eval_fn_pin_failure_raises(external_fn_env):
    from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput

    args = make_args(hf_checkpoint="/base")
    fn = external_fn_env.mod.ExternalSglangEvalFn(RolloutFnConstructorInput(args=args, data_source=None))
    external_fn_env.fake_pin.fail = True

    with pytest.raises(RuntimeError, match="pin failed"):
        await fn(RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/snap/step_5"))
    assert not [c for c in external_fn_env.calls if c[0] == "eval"]
