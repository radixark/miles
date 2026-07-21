from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
from miles.rollout.checkpoint_eval import make_eval_args, retarget_args


def make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_router_ip="10.0.0.1",
        sglang_router_port=30000,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
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


# ---------------- controller (RolloutManager._eval_on_dedicated_fleet) ----------------


class FakeRemoteMethod:
    """Mimics a Ray actor method: .remote(...) returns an awaitable."""

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


class FakeEvalServer:
    def __init__(self, engines):
        self._engines = engines
        self.recover_calls = 0

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
    mgr.servers = {"eval": FakeEvalServer(engines)}
    mgr._metric_checker = None
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
    """An engine whose backend died accepts the call but never answers; the load
    timeout must convert that into a skipped point instead of holding the eval
    lock forever (which would eventually stall the driver via backpressure)."""
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
    import train_async

    # ray.wait/ray.get over asyncio futures: done iff the future is resolved.
    monkeypatch.setattr(train_async.ray, "wait", lambda refs, timeout=0: (refs, []) if refs[0].done() else ([], refs))
    monkeypatch.setattr(train_async.ray, "get", lambda ref: ref.result())
    return train_async


def make_dispatcher(train_async, manager, actor_model, **arg_overrides):
    dispatcher_defaults = dict(
        eval_hf_dir="/dev/shm/eval_hf",
        eval_dispatch="async",
        eval_max_in_flight=2,
        eval_overflow_policy="backpressure",
    )
    dispatcher_defaults.update(arg_overrides)
    args = make_args(**dispatcher_defaults)
    return train_async.EvalDispatcher(args, actor_model, manager), args


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
