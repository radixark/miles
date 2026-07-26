from __future__ import annotations

import asyncio
import textwrap
import time

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController


@pytest.fixture
def patch_low_level(monkeypatch, mock_engine_http_servers):
    """Replace, in the test process:
    - ``SGLangEngine`` → ``MockSGLangEngine`` so created actors are mocks.
    - addr allocator → deterministic stub pointing at the mock http servers.
    - ``start_session_server`` → no-op (the production default touches network)."""
    import miles.ray.rollout.inference_controller as ictl
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
                    host=mock_engine_http_servers.new_for_rank(rank).host,
                    port=mock_engine_http_servers.for_rank(rank).port,
                    nccl_port=31000 + rank,
                    engine_info_bootstrap_port=32000 + rank,
                    dist_init_addr=f"127.0.0.1:{33000 + rank}",
                )
                for rank, _ in engines
            },
            PortCursors(_values={0: 34000}),
        )

    monkeypatch.setattr(sg, "allocate_rollout_engine_addr_and_ports_normal", _fake_alloc)
    monkeypatch.setattr(ictl, "start_session_server", lambda args: None)


def _make_controller(args, pg):
    return InferenceController(args, pg)


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
    """Build args that drive ``InferenceController.__init__`` →
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
class TestInferenceControllerInit:
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

        controller = _make_controller(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
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

        controller = _make_controller(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await controller.stop_cell(0)

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

        controller = _make_controller(args, pg)
        eal_before = await controller.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        await controller.stop_cell(0)
        await controller.start_cell(0)

        eal_after = await controller.get_updatable_engines_and_lock()
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

        controller = _make_controller(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await controller.stop_cell(1)

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

        controller = _make_controller(args, pg)
        await controller.get_updatable_engines_and_lock()  # ensure engines are alive

        await controller.stop_cell(0)
        await controller.stop_cell(0)  # must not raise


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

        controller = _make_controller(args, pg)
        actor_handles = [e.actor_handle for e in controller.servers["actor"].server_groups[0].engines]
        ref_handles = [e.actor_handle for e in controller.servers["ref"].server_groups[0].engines]

        await controller.stop_cell(2)

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

        controller = _make_controller(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
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

        controller = _make_controller(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
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

        controller = _make_controller(args, pg)
        eal_init = await controller.get_updatable_engines_and_lock()
        assert eal_init.has_new_engines is True

        await controller.clear_updatable_has_new_engines()
        eal_cleared = await controller.get_updatable_engines_and_lock()
        assert eal_cleared.has_new_engines is False

        await controller.stop_cell(0)
        await controller.start_cell(0)
        eal_recovered = await controller.get_updatable_engines_and_lock()
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

        controller = _make_controller(args, pg)
        # Force ref's flag True so we can detect any erroneous clear.
        controller.servers["ref"].server_groups[0].has_new_engines = True

        await controller.clear_updatable_has_new_engines()

        assert controller.servers["ref"].server_groups[0].has_new_engines is True
        assert controller.servers["actor"].server_groups[0].has_new_engines is False

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

        controller = _make_controller(args, pg)
        with pytest.raises(ValueError, match="Multiple servers"):
            await controller.get_updatable_engines_and_lock()


@pytest.mark.asyncio
class TestCheckWeights:
    async def test_check_weights_targets_only_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        mock_engine_http_servers,
    ):
        """``check_weights`` targets only the updatable model. The snapshot/reset/
        compare round-trip is meaningless for a frozen model (restored from disk,
        never re-synced via update_weights), so it must be skipped there."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        controller = _make_controller(args, pg)
        await controller.get_updatable_engines_and_lock()  # wait for engines to be alive

        results = await controller.check_weights(action="pre_update")

        # Updatable server only: nested gather is [group][engine]; 1 group × 2 engines.
        assert len(results) == 1
        for per_group in results:
            assert len(per_group) == 2
            for engine_result in per_group:
                assert engine_result == {"mock": True}

        updatable_urls = {
            engine.server_url
            for srv in controller.servers.values()
            if srv.update_weights
            for group in srv.server_groups
            for engine in group.engines
            if engine.is_allocated
        }
        frozen_urls = {
            engine.server_url
            for srv in controller.servers.values()
            if not srv.update_weights
            for group in srv.server_groups
            for engine in group.engines
            if engine.is_allocated
        }
        assert updatable_urls and frozen_urls

        for rank in range(4):
            server = mock_engine_http_servers.for_rank(rank)
            asked = "/weights_checker" in server.paths
            if server.url in updatable_urls:
                assert asked, f"updatable engine {server.url} was not checked"
            if server.url in frozen_urls:
                assert not asked, f"frozen engine {server.url} must not be checked"


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
        a slot looks dead the controller must not pre-emptively recover."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = _make_controller(args, pg)
        eal_before = await controller.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        # Kill engine 0 directly + mark stopped (simulates a fault before any
        # rollout). recover_updatable_engines must not bring it back yet.
        ray.kill(actor0_before)
        controller.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        await controller.recover_updatable_engines()

        # Slot 0 is still de-allocated; recovery skipped because rollout_id=-1.
        assert not controller.servers["actor"].server_groups[0].all_engines[0].is_allocated

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

        controller = _make_controller(args, pg)
        eal_before = await controller.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        ray.kill(actor0_before)
        controller.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        await controller.prepare_rollout(0)
        await controller.recover_updatable_engines()

        slot0 = controller.servers["actor"].server_groups[0].all_engines[0]
        assert slot0.is_allocated
        assert slot0.actor_handle is not actor0_before
        assert ray.get(slot0.actor_handle.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestRolloutFaultToleranceIsUnsupported:
    async def test_health_monitoring_hooks_are_noops_without_fault_tolerance(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """A plain run never asked for fault tolerance, so the hooks stay out of its way."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = _make_controller(args, pg)

        await controller.health_monitoring_pause()
        await controller.health_monitoring_resume()

    async def test_health_monitoring_hooks_refuse_to_run_under_fault_tolerance(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Asking for fault tolerance must fail loudly, not run unmonitored."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = _make_controller(args, pg)
        controller.args.use_fault_tolerance = True
        controller.args.ft_components = ["rollout"]

        with pytest.raises(NotImplementedError):
            await controller.health_monitoring_pause()
        with pytest.raises(NotImplementedError):
            await controller.health_monitoring_resume()

    async def test_health_monitoring_hooks_are_noops_when_fault_tolerance_skips_rollout(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Fault tolerance limited to training never monitored the engines, so nothing is lost."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = _make_controller(args, pg)
        controller.args.use_fault_tolerance = True
        controller.args.ft_components = ["train"]

        await controller.health_monitoring_pause()
        await controller.health_monitoring_resume()

    async def test_fault_injection_refuses_to_run(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """The injector depended on the deleted monitor to observe the crash."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = _make_controller(args, pg)

        with pytest.raises(NotImplementedError):
            await controller._try_ci_fault_injection()
