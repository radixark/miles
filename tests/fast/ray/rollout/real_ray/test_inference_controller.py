from __future__ import annotations

import asyncio
import textwrap
import time

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.workers.worker_spec import HostAndPort


class _NoopRouterApiClient:
    """The rollout process registers its engines for real; ``sglang_router_ip``
    here is a placeholder that keeps ``wait_router_ready`` short-circuited, and
    no router listens on it."""

    def __init__(self, router_url: str):
        self.router_url = router_url

    async def add_worker(self, **kwargs):
        return None

    async def remove_worker(self, **kwargs):
        return None


@pytest.fixture
def patch_low_level(monkeypatch):
    """Replace, in the test process:
    - ``CommandActor`` → ``MockSGLangEngine`` so created actors are mocks
      (the real addr allocator runs; each mock serves HTTP on its port).
    - ``SGLangRouterApiClient`` → no-op (no router runs at the placeholder address).
    - ``start_session_server`` → no-op (the production default touches network)."""
    import miles.ray.rollout.inference_controller as ictl
    import miles.ray.rollout.rollout_server as rsrv
    import miles.ray.rollout.server_cell as scell
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    monkeypatch.setattr(scell, "CommandActor", MockSGLangEngine.__ray_actor_class__)

    # each model would otherwise wait on a manager-launched router; return a
    # placeholder address nothing listens on instead.
    async def _fake_router_ready(*args, **kwargs):
        return HostAndPort(host="127.0.0.1", port=30000)

    monkeypatch.setattr(rsrv, "wait_router_ready", _fake_router_ready)

    monkeypatch.setattr(rsrv, "SGLangRouterApiClient", _NoopRouterApiClient)
    monkeypatch.setattr(ictl, "start_session_server", lambda args: None)


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
    """Build args that drive ``InferenceController.init`` →
    ``start_rollout_servers`` → N model servers each with 1 group of 2 mock
    engines."""
    cfg = _write_sglang_config(tmp_path, models=models)
    rollout_num_gpus = 2 * len(models)
    return make_args(
        sglang_config=cfg,
        rollout_num_gpus=rollout_num_gpus,
        # disable everything else that would spawn subprocesses or hit network
        use_session_server=False,
        use_fault_tolerance=False,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        sglang_server_concurrency=1,
    )


def _cells(controller, model: str = "actor"):
    return list(controller.servers[model].server_cells.values())


async def _assert_engine_dies(actor_handle, *, deadline_s: float = 15.0, poll_interval_s: float = 0.2) -> None:
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            ray.get(actor_handle.get_calls.remote(), timeout=5.0)
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
        """End-to-end smoke: production ``create`` + ``start_rollout_servers``
        runs against MockSGLangEngine; the resulting engines are addressable over
        http via the public ``get_updatable_engines``, and their launcher
        actors are reachable through the engine slots."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = InferenceController(args, pg)
        await controller.init()
        updatable = await controller.get_updatable_engines()
        assert len(updatable.rollout_engines) == 2
        for api_client in updatable.rollout_engines:
            assert isinstance(api_client, SGLangApiClient)
            assert await api_client.health_generate(timeout=5.0) is True
        for cell in _cells(controller):
            assert isinstance(ray.get(cell.primary_actor_handle.get_calls.remote()), list)


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_stop_cell_kills_target_engine_only(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell`` kills cell 0's actor; cell 1 untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = InferenceController(args, pg)
        await controller.init()
        await controller.get_updatable_engines()
        actor0, actor1 = [cell.primary_actor_handle for cell in _cells(controller)]

        await controller.stop_cell("actor-0")

        await _assert_engine_dies(actor0)
        assert isinstance(ray.get(actor1.get_calls.remote()), list)

    async def test_start_cell_recovers_after_stop_cell(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """stop_cell → start_cell drives a real ``recover()`` that spawns
        a fresh mock actor in place of the killed one."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = InferenceController(args, pg)
        await controller.init()
        updatable_before = await controller.get_updatable_engines()
        actor0_before = _cells(controller)[0].primary_actor_handle
        url_before = updatable_before.rollout_engines[0].server_url

        await controller.stop_cell("actor-0")
        await controller.start_cell("actor-0")

        updatable_after = await controller.get_updatable_engines()
        actor0_after = _cells(controller)[0].primary_actor_handle

        assert actor0_after is not actor0_before, "start_cell must produce a fresh actor"
        assert updatable_after.rollout_engines[0].server_url != url_before, "the recovered engine serves on a new port"
        assert await updatable_after.rollout_engines[0].health_generate(timeout=5.0) is True
        assert isinstance(ray.get(actor0_after.get_calls.remote()), list)

    async def test_stop_cell_targets_high_id_correctly(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell("actor-1")`` (not 0) must kill engine 1, leaving engine 0
        alive — guards against addressing the wrong cell."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = InferenceController(args, pg)
        await controller.init()
        await controller.get_updatable_engines()
        actor0, actor1 = [cell.primary_actor_handle for cell in _cells(controller)]

        await controller.stop_cell("actor-1")

        assert isinstance(ray.get(actor0.get_calls.remote()), list)
        await _assert_engine_dies(actor1)

    async def test_stop_cell_is_idempotent_on_already_stopped(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Calling ``stop_cell`` twice does not raise — production code logs
        and proceeds when the engine is already de-allocated."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = InferenceController(args, pg)
        await controller.init()
        await controller.get_updatable_engines()  # ensure engines are alive

        await controller.stop_cell("actor-0")
        await controller.stop_cell("actor-0")  # must not raise


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

        controller = InferenceController(args, pg)
        await controller.init()
        actor_handles = [cell.primary_actor_handle for cell in _cells(controller, "actor")]
        ref_handles = [cell.primary_actor_handle for cell in _cells(controller, "ref")]

        await controller.stop_cell("ref-0")

        # actor untouched
        for h in actor_handles:
            assert isinstance(ray.get(h.get_calls.remote()), list)
        # ref engine 0 dead, ref engine 1 alive
        await _assert_engine_dies(ref_handles[0])
        assert isinstance(ray.get(ref_handles[1].get_calls.remote()), list)


@pytest.mark.asyncio
class TestGetUpdatableEngines:
    async def test_returns_only_updatable_servers_engines_in_multi_model_setup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned UpdatableEngines contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        controller = InferenceController(args, pg)
        await controller.init()
        updatable = await controller.get_updatable_engines()
        assert len(updatable.rollout_engines) == 2  # actor's 2, not ref's 2
        assert updatable.engine_gpu_counts == [1, 1]
        assert all(isinstance(api_client, SGLangApiClient) for api_client in updatable.rollout_engines)
        assert await updatable.rollout_engines[0].health_generate(timeout=5.0) is True

    async def test_returns_empty_when_no_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """If every model has ``update_weights=False`` (e.g. inference-only
        deployment), the returned UpdatableEngines has an empty engines list."""
        args = _make_test_args(tmp_path, models=[("ref", False)])
        pg = placement_group_factory(2)

        controller = InferenceController(args, pg)
        await controller.init()
        updatable = await controller.get_updatable_engines()
        assert updatable.rollout_engines == []
        assert updatable.engine_gpu_counts == []
        assert updatable.has_new_engines is False

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

        controller = InferenceController(args, pg)
        await controller.init()
        eal_init = await controller.get_updatable_engines()
        assert eal_init.has_new_engines is True

        await controller.clear_updatable_has_new_engines()
        eal_cleared = await controller.get_updatable_engines()
        assert eal_cleared.has_new_engines is False

        await controller.stop_cell("actor-0")
        await controller.start_cell("actor-0")
        eal_recovered = await controller.get_updatable_engines()
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

        controller = InferenceController(args, pg)
        await controller.init()
        # Force ref's flag True so we can detect any erroneous clear.
        controller.servers["ref"].has_new_engines = True

        await controller.clear_updatable_has_new_engines()

        assert controller.servers["ref"].has_new_engines is True
        assert controller.servers["actor"].has_new_engines is False

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

        controller = InferenceController(args, pg)
        await controller.init()
        with pytest.raises(ValueError, match="Multiple servers"):
            await controller.get_updatable_engines()


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

        controller = InferenceController(args, pg)
        await controller.init()
        await controller.get_updatable_engines()  # wait for engines to be alive

        results = await controller.check_weights(action="pre_update")

        # Updatable server only: one flat entry per cell's primary engine.
        assert len(results) == 2
        for engine_result in results:
            assert engine_result == {"mock": True}

        updatable_cells = [
            cell
            for srv in controller.servers.values()
            if srv.update_weights
            for cell in srv.server_cells.values()
            if cell.is_allocated
        ]
        frozen_cells = [
            cell
            for srv in controller.servers.values()
            if not srv.update_weights
            for cell in srv.server_cells.values()
            if cell.is_allocated
        ]
        assert updatable_cells and frozen_cells

        for cell in updatable_cells:
            paths = ray.get(cell.primary_actor_handle.get_http_paths.remote())
            assert "/weights_checker" in paths, f"updatable engine {cell.addr_info.server_url} was not checked"
        for cell in frozen_cells:
            paths = ray.get(cell.primary_actor_handle.get_http_paths.remote())
            assert "/weights_checker" not in paths, f"frozen engine {cell.addr_info.server_url} must not be checked"


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

        controller = InferenceController(args, pg)
        await controller.init()
        await controller.get_updatable_engines()
        actor0_before = _cells(controller)[0].primary_actor_handle

        # Kill engine 0 directly + mark stopped (simulates a fault before any
        # rollout). recover_updatable_engines must not bring it back yet.
        ray.kill(actor0_before)
        controller.servers["actor"].server_cells["actor-0"]._mark_stopped()

        await controller.recover_updatable_engines()

        # Slot 0 is still de-allocated; recovery skipped because rollout_id=-1.
        assert not controller.servers["actor"].server_cells["actor-0"].is_allocated

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

        controller = InferenceController(args, pg)
        await controller.init()
        await controller.get_updatable_engines()
        actor0_before = _cells(controller)[0].primary_actor_handle

        ray.kill(actor0_before)
        controller.servers["actor"].server_cells["actor-0"]._mark_stopped()

        await controller.prepare_rollout(0)
        await controller.recover_updatable_engines()

        slot0 = controller.servers["actor"].server_cells["actor-0"]
        assert slot0.is_allocated
        assert slot0.primary_actor_handle is not actor0_before
        assert isinstance(ray.get(slot0.primary_actor_handle.get_calls.remote()), list)


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

        controller = InferenceController(args, pg)
        await controller.init()

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

        controller = InferenceController(args, pg)
        await controller.init()
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

        controller = InferenceController(args, pg)
        await controller.init()
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

        controller = InferenceController(args, pg)
        await controller.init()

        with pytest.raises(NotImplementedError):
            await controller._try_ci_fault_injection()
