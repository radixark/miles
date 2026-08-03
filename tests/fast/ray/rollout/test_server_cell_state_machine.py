from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata

_ADDR_INFO = CellAddrInfo(
    server_url="http://10.0.0.1:30000",
    bootstrap_port=None,
    gate_url="http://10.0.0.1:13000",
)


def _make_meta(**overrides) -> ServerCellMetadata:
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": "inference-engine-0-0-0",
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": "inference-engine-0-0-0-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


class _RecordingRouterApiClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.add_worker_error: Exception | None = None

    async def add_worker(self, **kwargs):
        self.calls.append(("add_worker", kwargs))
        if self.add_worker_error is not None:
            raise self.add_worker_error

    async def remove_worker(self, **kwargs):
        self.calls.append(("remove_worker", kwargs))


class _RecordingApiClient:
    calls: list[tuple[str, dict]] = []

    def __init__(self, server_url: str):
        self.server_url = server_url

    async def release_memory_occupation(self, tags=None):
        _RecordingApiClient.calls.append(("release", dict(tags=tags)))

    async def resume_memory_occupation(self, tags=None):
        _RecordingApiClient.calls.append(("resume", dict(tags=tags)))

    async def check_weights(self, **kwargs):
        _RecordingApiClient.calls.append(("check_weights", kwargs))


@pytest.fixture
def cell_env(monkeypatch):
    """Stub out everything the cell reaches over the network, and record the calls."""
    _RecordingApiClient.calls = []
    activated: list[str] = []
    health: dict[str, bool] = {"ready": True}

    async def _activate(gate_url: str) -> None:
        activated.append(gate_url)

    async def _compute_addr_info(self) -> CellAddrInfo:
        return _ADDR_INFO

    async def _probe(server_url: str, api_key, timeout: float = 5.0) -> bool:
        return health["ready"]

    monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
    monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)

    return dict(activated=activated, health=health, memory_calls=_RecordingApiClient.calls)


def _make_cell(
    *, router: _RecordingRouterApiClient | None = None, args_overrides=None, **meta_overrides
) -> ServerCell:
    return ServerCell(
        args=make_args(**(args_overrides or {})),
        meta=_make_meta(**meta_overrides),
        router_api_client=router or _RecordingRouterApiClient(),
    )


class TestInit:
    async def test_a_fresh_cell_has_not_been_initialized_yet(self):
        """A constructed cell is only bookkeeping until someone releases its engine."""
        assert isinstance(_make_cell()._state, StateUninitialized)

    async def test_init_releases_the_gate_and_moves_to_initializing(self, cell_env):
        """Initialization is exactly the moment the engine is allowed to claim gpu memory."""
        cell = _make_cell()

        await cell.init()

        assert cell_env["activated"] == ["http://10.0.0.1:13000"]
        assert isinstance(cell._state, StateInitializing)
        assert cell.addr_info == _ADDR_INFO

    async def test_init_does_not_wait_for_the_engine_to_become_ready(self, cell_env):
        """Blocking here would stall the caller for the minutes an engine takes to load."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()

        await cell.init()

        assert cell.is_initializing

    async def test_initializing_twice_is_rejected(self, cell_env):
        """A second release would race the first engine's startup."""
        cell = _make_cell()
        await cell.init()

        with pytest.raises(AssertionError):
            await cell.init()


class TestTick:
    async def test_an_uninitialized_cell_is_not_probed(self, cell_env):
        """It has no address yet, so probing it would be dialing nothing."""
        cell = _make_cell()

        await cell.tick()

        assert isinstance(cell._state, StateUninitialized)

    async def test_a_cell_whose_engine_is_not_ready_stays_initializing(self, cell_env):
        """Readiness is polled, so a not-yet-ready engine must simply be retried later."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        await cell.init()

        await cell.tick()
        await cell.tick()

        assert cell.is_initializing

    async def test_a_ready_engine_moves_on_to_awaiting_its_weights(self, cell_env):
        """A loaded engine still serves nothing until the trainer has pushed weights."""
        cell = _make_cell()
        await cell.init()

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_ticking_a_pending_cell_again_does_not_redo_its_startup(self, cell_env):
        """The sweep runs every second, so a second release would thrash the engine's memory."""
        cell = _make_cell(needs_offload=True)
        await cell.init()
        await cell.tick()
        calls_after_first_tick = list(cell_env["memory_calls"])

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)
        assert cell_env["memory_calls"] == calls_after_first_tick

    async def test_an_engine_sharing_gpus_with_the_trainer_hands_its_memory_back(self, cell_env):
        """Under colocation the engine must give the gpus back until the next rollout."""
        cell = _make_cell(needs_offload=True)
        await cell.init()

        await cell.tick()

        assert cell_env["memory_calls"] == [("release", dict(tags=None)), ("resume", dict(tags=["weights"]))]

    async def test_the_weight_checker_snapshots_before_the_memory_is_handed_back(self, cell_env):
        """Releasing the occupation discards the loaded weights, so a later snapshot records garbage."""
        cell = _make_cell(needs_offload=True, args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        await cell.tick()

        names = [name for name, _kwargs in cell_env["memory_calls"]]
        assert names.index("check_weights") < names.index("release")

    async def test_an_engine_on_its_own_gpus_keeps_its_memory(self, cell_env):
        """Without colocation there is nobody to hand the memory to."""
        cell = _make_cell(needs_offload=False)
        await cell.init()

        await cell.tick()

        assert cell_env["memory_calls"] == []

    async def test_a_frozen_model_starts_serving_without_a_weight_update(self, cell_env):
        """A model the trainer never updates would otherwise wait for a push that never comes."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()

        await cell.tick()

        assert isinstance(cell._state, StateServing)
        assert [name for name, _kwargs in router.calls] == ["add_worker"]

    async def test_a_frozen_cell_whose_router_registration_fails_is_retried_by_a_later_tick(self, cell_env):
        """Nothing else ever revisits a frozen cell, so a transient router error must not strand it."""

        class _FlakyRouter(_RecordingRouterApiClient):
            def __init__(self):
                super().__init__()
                self.failures = 1

            async def add_worker(self, **kwargs):
                if self.failures > 0:
                    self.failures -= 1
                    raise RuntimeError("router rejected the worker")
                await super().add_worker(**kwargs)

        router = _FlakyRouter()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()

        with pytest.raises(RuntimeError):
            await cell.tick()
        assert cell.is_initializing

        await cell.tick()

        assert isinstance(cell._state, StateServing)
        assert [name for name, _kwargs in router.calls] == ["add_worker"]

    async def test_debug_rollout_only_starts_serving_without_a_weight_update(self, cell_env):
        """With no trainer running, the engine must serve the weights it loaded from disk."""
        cell = _make_cell(args_overrides=dict(debug_rollout_only=True))
        await cell.init()

        await cell.tick()

        assert isinstance(cell._state, StateServing)

    async def test_an_engine_that_becomes_ready_later_is_picked_up_by_a_later_tick(self, cell_env):
        """Polling is only worth anything if the transition actually happens once readiness flips."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        await cell.init()
        await cell.tick()

        cell_env["health"]["ready"] = True
        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_a_cell_awaiting_a_real_weight_update_is_not_published_by_the_sweep(self, cell_env):
        """Publishing before the trainer pushes weights would serve the checkpoint's stale weights."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)
        assert router.calls == []

    async def test_a_serving_cell_is_inert_under_the_sweep(self, cell_env):
        """Re-publishing a live engine would duplicate it in the router."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()
        await cell.tick()

        await cell.tick()

        assert isinstance(cell._state, StateServing)
        assert [name for name, _kwargs in router.calls] == ["add_worker"]

    async def test_the_weight_baseline_is_snapshotted_before_memory_is_released(self, cell_env):
        """The checker's baseline must be the freshly loaded checkpoint, not the remapped weight storage."""
        cell = _make_cell(needs_offload=True, args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        await cell.tick()

        names = [
            f"check_weights:{kwargs['action']}" if name == "check_weights" else name
            for name, kwargs in cell_env["memory_calls"]
        ]
        assert names == ["check_weights:snapshot", "release", "resume", "check_weights:reset_tensors"]

    async def test_the_snapshot_is_taken_over_the_whole_model_without_a_skip_list(self, cell_env):
        """The baseline must match what the controller-side reset and comparison later cover."""
        cell = _make_cell(
            args_overrides=dict(check_weight_update_equal=True, check_weight_update_skip_list=["lm_head"])
        )
        await cell.init()

        await cell.tick()

        snapshot_calls = [
            kwargs
            for name, kwargs in cell_env["memory_calls"]
            if name == "check_weights" and kwargs["action"] == "snapshot"
        ]
        assert snapshot_calls == [dict(action="snapshot", allow_quant_error=False, selector="all", skip_list=None)]

    async def test_a_cell_primes_the_weight_update_checker_before_its_first_update(self, cell_env):
        """The checker scrambles the engine tensors so a no-op weight update cannot pass unnoticed."""
        cell = _make_cell(args_overrides=dict(check_weight_update_equal=True, check_weight_update_skip_list=["a"]))
        await cell.init()

        await cell.tick()

        checker_calls = [kwargs for name, kwargs in cell_env["memory_calls"] if name == "check_weights"]
        assert [kwargs["action"] for kwargs in checker_calls] == ["snapshot", "reset_tensors"]
        assert checker_calls[1]["skip_list"] == ["a"]

    async def test_the_weight_update_checker_stays_out_of_the_way_when_it_is_disabled(self, cell_env):
        """The checker is a ci-only tool and must not touch engines in a normal run."""
        cell = _make_cell()
        await cell.init()

        await cell.tick()

        assert [name for name, _kwargs in cell_env["memory_calls"] if name == "check_weights"] == []

    async def test_a_frozen_cell_is_not_scrambled_because_nothing_will_rewrite_it(self, cell_env):
        """Resetting tensors nobody updates would leave the frozen model serving garbage."""
        cell = _make_cell(update_weights=False, args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        await cell.tick()

        assert [name for name, _kwargs in cell_env["memory_calls"] if name == "check_weights"] == []

    async def test_an_uninitialized_cell_has_no_address_to_offer(self, cell_env):
        """Reading an address before the gate opens is what silently dialed nothing before."""
        cell = _make_cell()

        with pytest.raises(AssertionError):
            _ = cell.addr_info


class TestMarkWeightsReady:
    async def test_it_publishes_the_cell_to_the_router_and_starts_serving(self, cell_env):
        """Registering earlier would route requests at an engine holding stale weights."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()

        await cell.mark_weights_ready()

        assert router.calls == [
            (
                "add_worker",
                dict(
                    worker_url="http://10.0.0.1:30000",
                    worker_type="regular",
                    use_legacy_api=False,
                    bootstrap_port=None,
                ),
            )
        ]
        assert isinstance(cell._state, StateServing)

    async def test_a_cell_stays_pending_weights_when_the_router_rejects_the_registration(self, cell_env):
        """Marking it serving on a failed add_worker would strand the cell: never registered, never retried."""
        router = _RecordingRouterApiClient()
        router.add_worker_error = RuntimeError("router returned 503")
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()

        with pytest.raises(RuntimeError, match="router returned 503"):
            await cell.mark_weights_ready()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_a_failed_registration_can_be_retried_by_a_later_end_update_weights(self, cell_env):
        """Because the cell is still pending, the next weight window registers it instead of skipping it."""
        router = _RecordingRouterApiClient()
        router.add_worker_error = RuntimeError("router returned 503")
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()
        with pytest.raises(RuntimeError):
            await cell.mark_weights_ready()

        router.add_worker_error = None
        await cell.mark_weights_ready()

        assert len([name for name, _kwargs in router.calls if name == "add_worker"]) == 2
        assert isinstance(cell._state, StateServing)

    async def test_a_cell_that_is_still_initializing_cannot_be_marked_ready(self, cell_env):
        """Its engine may not even be loaded, so publishing it would break routing."""
        cell = _make_cell()
        await cell.init()

        with pytest.raises(AssertionError):
            await cell.mark_weights_ready()

    async def test_a_prefill_cell_publishes_its_bootstrap_port(self, cell_env, monkeypatch):
        """PD disaggregation needs the decode side to dial this port."""

        async def _compute_addr_info(self) -> CellAddrInfo:
            return CellAddrInfo(
                server_url="http://10.0.0.1:30000", bootstrap_port=8998, gate_url="http://10.0.0.1:13000"
            )

        monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, worker_type="prefill")
        await cell.init()
        await cell.tick()

        await cell.mark_weights_ready()

        assert router.calls[0][1]["bootstrap_port"] == 8998


class TestDispose:
    async def test_a_serving_cell_is_withdrawn_from_the_router(self, cell_env):
        """Leaving it registered would route requests at an engine that is going away."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()
        await cell.mark_weights_ready()

        await cell.dispose()

        assert [name for name, _kwargs in router.calls] == ["add_worker", "remove_worker"]
        assert isinstance(cell._state, StateDisposed)

    @pytest.mark.parametrize("stage", ["uninitialized", "initializing", "pending_weights"])
    async def test_a_cell_can_be_disposed_from_any_earlier_stage(self, cell_env, stage: str):
        """Reconcile removes cells at arbitrary moments, including mid-startup."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        if stage != "uninitialized":
            await cell.init()
        if stage == "pending_weights":
            await cell.tick()

        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)
        assert router.calls == []

    async def test_disposing_twice_is_harmless(self, cell_env):
        """Teardown paths overlap, so a second dispose must not raise."""
        cell = _make_cell()

        await cell.dispose()
        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)

    async def test_a_router_that_rejects_the_removal_still_disposes_the_cell(self, cell_env):
        """Teardown is how a wedged engine is reclaimed, so a router error must not abort it."""

        class _RejectingRouter(_RecordingRouterApiClient):
            async def remove_worker(self, **kwargs):
                raise RuntimeError("router rejected the removal")

        cell = _make_cell(router=_RejectingRouter())
        await cell.init()
        await cell.tick()
        await cell.mark_weights_ready()

        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)

    async def test_a_disposed_cell_is_inert_under_the_tick_sweep(self, cell_env):
        """The sweep may still hold a reference to it for one iteration."""
        cell = _make_cell()
        await cell.init()
        await cell.dispose()

        await cell.tick()

        assert isinstance(cell._state, StateDisposed)
