"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
the production ``RolloutManager`` class.

We instantiate ``RolloutManager.__ray_actor_class__`` directly (the raw Python
class behind ``@ray.remote``) — that keeps the manager in the test process so
``monkeypatch`` reaches its dependencies, while the engines it spawns are
still real Ray actors (mocks). Methods are ``async`` and called with ``await``.

Pure routing/flag-flip helpers without Ray content live in
``tests/fast/ray/rollout/test_rollout_manager.py``."""

from __future__ import annotations

import asyncio
import textwrap
import time

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_manager import RolloutManager
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainInput, RolloutFnTrainOutput


@pytest.fixture
def patch_low_level(monkeypatch):
    """Replace, in the test process:
    - ``SGLangEngine`` → ``MockSGLangEngine`` so created actors are mocks.
    - addr allocator → deterministic stub.
    - ``init_tracking`` / ``init_http_client`` / ``start_session_server`` /
      ``load_function`` / ``load_rollout_function`` → no-ops (the production
      defaults touch wandb / network / not-importable default function paths)."""
    import miles.ray.rollout.rollout_manager as rmgr
    import miles.ray.rollout.rollout_server as rsrv
    import miles.ray.rollout.server_group as sg
    from miles.ray.rollout.addr_allocator import PortCursors
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    monkeypatch.setattr(sg, "SGLangEngine", MockSGLangEngine.__ray_actor_class__)
    # multi-model tests would otherwise spawn a real router subprocess for
    # ``model_idx > 0`` (force_new=True bypasses the args.sglang_router_ip cache).
    monkeypatch.setattr(
        rsrv,
        "start_router",
        lambda args, **kw: (args.sglang_router_ip, args.sglang_router_port),
    )

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        return (
            {
                rank: dict(
                    host="127.0.0.1",
                    port=30000 + rank,
                    nccl_port=31000 + rank,
                    engine_info_bootstrap_port=32000 + rank,
                    dist_init_addr=f"127.0.0.1:{33000 + rank}",
                )
                for rank, _ in engines
            },
            PortCursors(_values={0: 34000}),
        )

    monkeypatch.setattr(sg, "allocate_rollout_engine_addr_and_ports_normal", _fake_alloc)
    monkeypatch.setattr(rmgr, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "init_http_client", lambda args: None)
    monkeypatch.setattr(rmgr, "start_session_server", lambda args: None)
    monkeypatch.setattr(rmgr, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "load_rollout_function", lambda input, path: lambda *a, **kw: None)
    # generate()/eval() drive these — production hits wandb / tensorboard.
    monkeypatch.setattr(rmgr, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "save_debug_rollout_data", lambda *a, **kw: None)


def _make_manager(args, pg):
    return RolloutManager.__ray_actor_class__(args, pg)


def _write_sglang_config(tmp_path, *, models: list[tuple[str, bool]]) -> str:
    """Write a multi-model sglang yaml — each entry ``(name, update_weights)``.
    Each model gets one regular group with 2 engines × 1 GPU = 2 GPUs. With N
    models, total GPUs = 2N; ``args.rollout_num_gpus`` must match."""
    lines = ["sglang:"]
    for name, update_weights in models:
        lines.extend(
            [
                f"  - name: {name}",
                f"    update_weights: {str(update_weights).lower()}",
                "    server_groups:",
                "      - worker_type: regular",
                "        num_gpus: 2",
                "        num_gpus_per_engine: 1",
            ]
        )
    cfg_path = tmp_path / "sglang.yaml"
    cfg_path.write_text(textwrap.dedent("\n".join(lines)) + "\n")
    return str(cfg_path)


def _make_test_args(tmp_path, *, models: list[tuple[str, bool]]):
    """Build args that drive ``RolloutManager.__init__`` →
    ``start_rollout_servers`` → N model servers each with 1 group of 2 mock
    engines."""
    cfg = _write_sglang_config(tmp_path, models=models)
    rollout_num_gpus = 2 * len(models)
    return make_args(
        sglang_config=cfg,
        rollout_num_gpus=rollout_num_gpus,
        # short-circuit start_router (returns early when ip+port already set)
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        # disable everything else that would spawn subprocesses or hit network
        use_session_server=False,
        use_fault_tolerance=False,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        sglang_server_concurrency=1,
    )


async def _assert_engine_dies(actor_handle, *, deadline_s: float = 15.0, poll_interval_s: float = 0.2) -> None:
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            ray.get(actor_handle.health_generate.remote(timeout=1.0), timeout=5.0)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return
        except ray.exceptions.GetTimeoutError:
            pass
        if time.monotonic() >= deadline:
            pytest.fail(f"engine actor still alive {deadline_s}s after stop_cell")
        await asyncio.sleep(poll_interval_s)


@pytest.mark.asyncio
class TestRolloutManagerInit:
    async def test_init_creates_live_mock_engines_via_real_start_rollout_servers(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """End-to-end smoke: production ``__init__`` + ``start_rollout_servers``
        runs against MockSGLangEngine; resulting engines are reachable as Ray
        actor handles via the public ``get_updatable_engines_and_lock``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2
        for h in eal.rollout_engines:
            assert isinstance(h, ray.actor.ActorHandle)
            assert ray.get(h.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_stop_cell_kills_target_engine_only(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell(0)`` kills cell 0's actor; cell 1 untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await manager.stop_cell(0)

        await _assert_engine_dies(actor0)
        assert ray.get(actor1.health_generate.remote(timeout=1.0)) is True

    async def test_start_cell_recovers_after_stop_cell(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """stop_cell(0) → start_cell(0) drives a real ``recover()`` that spawns
        a fresh mock actor in place of the killed one."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        await manager.stop_cell(0)
        await manager.start_cell(0)

        eal_after = await manager.get_updatable_engines_and_lock()
        actor0_after = eal_after.rollout_engines[0]

        assert actor0_after is not actor0_before, "start_cell must produce a fresh actor"
        assert ray.get(actor0_after.health_generate.remote(timeout=1.0)) is True

    async def test_stop_cell_targets_high_id_correctly(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell(1)`` (not 0) must kill engine 1, leaving engine 0 alive —
        guards against off-by-one in ``get_cell_indexer_of_id_map``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await manager.stop_cell(1)

        assert ray.get(actor0.health_generate.remote(timeout=1.0)) is True
        await _assert_engine_dies(actor1)

    async def test_stop_cell_is_idempotent_on_already_stopped(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Calling ``stop_cell(0)`` twice does not raise — production code logs
        and proceeds when the engine is already de-allocated."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        await manager.get_updatable_engines_and_lock()  # ensure engines are alive

        await manager.stop_cell(0)
        await manager.stop_cell(0)  # must not raise


@pytest.mark.asyncio
class TestCellDispatchAcrossModels:
    async def test_cells_route_to_correct_model_by_sorted_srv_key(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Cells are flattened in sorted-srv-key order: with models ("actor",
        "ref") the cells map (0,1)→actor, (2,3)→ref. Stopping cell 2 must hit
        ref's first engine and leave actor's engines untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        actor_handles = [e.actor_handle for e in manager.servers["actor"].server_groups[0].engines]
        ref_handles = [e.actor_handle for e in manager.servers["ref"].server_groups[0].engines]

        await manager.stop_cell(2)

        # actor untouched
        for h in actor_handles:
            assert ray.get(h.health_generate.remote(timeout=1.0)) is True
        # ref engine 0 dead, ref engine 1 alive
        await _assert_engine_dies(ref_handles[0])
        assert ray.get(ref_handles[1].health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_only_updatable_servers_engines_in_multi_model_setup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned EnginesAndLock contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2  # actor's 2, not ref's 2
        assert eal.engine_gpu_counts == [1, 1]
        assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
        assert ray.get(eal.rollout_engines[0].health_generate.remote(timeout=1.0)) is True

    async def test_returns_empty_when_no_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """If every model has ``update_weights=False`` (e.g. inference-only
        deployment), the returned EnginesAndLock has empty engines list and
        the lock handle is still present (callers always need a lock)."""
        args = _make_test_args(tmp_path, models=[("ref", False)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert eal.rollout_engines == []
        assert eal.engine_gpu_counts == []
        assert eal.has_new_engines is False
        assert eal.rollout_engine_lock is not None

    async def test_has_new_engines_flag_lifecycle(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Lifecycle the trainer relies on: ``has_new_engines`` is True after
        init, False after ``clear_updatable_has_new_engines``, True again
        after ``start_cell`` spawns a fresh engine."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_init = await manager.get_updatable_engines_and_lock()
        assert eal_init.has_new_engines is True

        manager.clear_updatable_has_new_engines()
        eal_cleared = await manager.get_updatable_engines_and_lock()
        assert eal_cleared.has_new_engines is False

        await manager.stop_cell(0)
        await manager.start_cell(0)
        eal_recovered = await manager.get_updatable_engines_and_lock()
        assert eal_recovered.has_new_engines is True

    async def test_clear_does_not_affect_non_updatable_server(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``clear_updatable_has_new_engines`` must touch only the updatable
        server's flag; non-updatable (ref) servers keep their flag intact."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        # Force ref's flag True so we can detect any erroneous clear.
        manager.servers["ref"].server_groups[0].has_new_engines = True

        manager.clear_updatable_has_new_engines()

        assert manager.servers["ref"].server_groups[0].has_new_engines is True
        assert manager.servers["actor"].server_groups[0].has_new_engines is False

    async def test_multiple_updatable_servers_raises_assertion(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Production guards against misconfiguration where two models both set
        ``update_weights=True``; that's ambiguous for the trainer."""
        args = _make_test_args(tmp_path, models=[("actor1", True), ("actor2", True)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        with pytest.raises(ValueError, match="Multiple servers"):
            await manager.get_updatable_engines_and_lock()


@pytest.mark.asyncio
class TestCheckWeights:
    async def test_check_weights_targets_only_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``check_weights`` targets only the updatable model. The snapshot/reset/
        compare round-trip is meaningless for a frozen model (restored from disk,
        never re-synced via update_weights), so it must be skipped there."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        await manager.get_updatable_engines_and_lock()  # wait for engines to be alive

        results = await manager.check_weights(action="pre_update")

        # Updatable server only: nested gather is [group][engine]; 1 group × 2 engines.
        assert len(results) == 1
        for per_group in results:
            assert len(per_group) == 2
            for engine_result in per_group:
                assert engine_result == {"_mock": True}

        # Frozen (non-updatable) servers must not have been touched.
        for srv in manager.servers.values():
            if srv.update_weights:
                continue
            for group in srv.server_groups:
                for engine in group.engines:
                    if not engine.is_allocated:
                        continue
                    calls = ray.get(engine.actor_handle.get_calls.remote())
                    assert not any(c[0] == "check_weights" for c in calls)


@pytest.mark.asyncio
class TestRecoverUpdatableEngines:
    async def test_skips_recovery_when_no_rollout_started(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``recover_updatable_engines`` is a no-op while ``rollout_id == -1``
        (initial state) — the trainer hasn't issued a rollout yet, so even if
        a slot looks dead the manager must not pre-emptively recover."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        # Kill engine 0 directly + mark stopped (simulates a fault before any
        # rollout). recover_updatable_engines must not bring it back yet.
        ray.kill(actor0_before)
        manager.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        await manager.recover_updatable_engines()

        # Slot 0 is still de-allocated; recovery skipped because rollout_id=-1.
        assert not manager.servers["actor"].server_groups[0].all_engines[0].is_allocated

    async def test_recovers_dead_engine_after_rollout_started(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Once ``rollout_id`` advances past -1 (mid-training), a dead slot on
        the updatable server is brought back by ``recover_updatable_engines``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        ray.kill(actor0_before)
        manager.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        manager.rollout_id = 0  # simulates "rollout has started"
        await manager.recover_updatable_engines()

        slot0 = manager.servers["actor"].server_groups[0].all_engines[0]
        assert slot0.is_allocated
        assert slot0.actor_handle is not actor0_before
        assert ray.get(slot0.actor_handle.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestGenerate:
    """``generate(rollout_id)`` is the trainer's per-iteration rollout entry
    point. It must (1) advance ``self.rollout_id``, (2) call the rollout
    function with ``RolloutFnTrainInput(rollout_id=N)``, (3) postprocess +
    convert + DP-split the returned samples. Nothing else covers this path."""

    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        # global_batch_size = number of samples we'll produce (postprocess
        # trims to a multiple, so equality avoids losing samples).
        args.global_batch_size = 8
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}

        captured: list = []

        def fake_rollout_fn(input):
            captured.append(input)
            return RolloutFnTrainOutput(
                samples=[make_samples_grouped(n_groups=2, group_size=4)],
                metrics={"my_metric": 1.23},
            )

        manager.generate_rollout = fake_rollout_fn

        result = await manager.generate(rollout_id=42)

        assert manager.rollout_id == 42
        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnTrainInput)
        assert captured[0].rollout_id == 42
        # split_train_data_by_dp returns Box(ObjectRef) per dp rank
        assert len(result) == 2
        partitions = ray.get([box.inner for box in result])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            # 8 samples / 2 dp = 4 per rank
            assert len(partition["tokens"]) == 4


@pytest.mark.asyncio
class TestEval:
    async def test_invokes_eval_fn_with_eval_input(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)

        captured: list = []

        def fake_eval_fn(input):
            captured.append(input)
            return RolloutFnEvalOutput(
                data={"my_dataset": {"rewards": [0.5, 1.0]}},
                metrics={},
            )

        manager.eval_generate_rollout = fake_eval_fn

        await manager.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10

    async def test_skipped_in_debug_train_only_mode(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``debug_train_only=True`` must short-circuit ``eval`` before the
        rollout function is invoked — used by trainer-only debug runs that
        have no rollout cluster."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)

        called: list = []
        manager.eval_generate_rollout = lambda inp: called.append(inp)

        await manager.eval(rollout_id=10)

        assert called == []
