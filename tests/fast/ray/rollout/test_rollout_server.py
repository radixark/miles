from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_config import (
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
from miles.ray.rollout.cell_state import CellAddrInfo, StateServing
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock


class TestRolloutServerPureFunctions:
    def test_resolve_sglang_config_yaml_gpu_mismatch_asserts(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="total GPUs"):
            resolve_sglang_config(args)

    def test_eval_fleet_inherits_rollout_engine_settings(self):
        """The eval model carries only what makes it an eval fleet; the rest is inherited."""
        args = make_args(eval_num_gpus=2, eval_num_gpus_per_engine=2)
        config = resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        assert eval_model.update_weights is False
        [group] = eval_model.server_groups
        assert (group.num_gpus, group.num_gpus_per_engine) == (2, 2)
        # Eval samples never feed training, so the replay side-channels are forced off.
        assert group.overrides["enable_return_routed_experts"] is False
        assert group.overrides["enable_return_indexer_topk"] is False

        # The fleet boots on --hf-checkpoint; every eval overwrites those weights anyway.
        assert group.model_path == args.hf_checkpoint

    def test_tp_coupled_sizes_are_not_inherited_across_a_different_eval_tp(self):
        """Inheriting these across a different eval tp gives an engine that will not boot."""
        args = make_args(rollout_num_gpus_per_engine=8, eval_num_gpus=1, eval_num_gpus_per_engine=1)
        [group] = [m for m in resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert {k: group.overrides[k] for k in ("dp_size", "pp_size", "ep_size", "attn_cp_size")} == dict.fromkeys(
            ("dp_size", "pp_size", "ep_size", "attn_cp_size"), 1
        )

    def test_tp_coupled_sizes_are_inherited_when_the_tp_matches(self):
        args = make_args(rollout_num_gpus_per_engine=2, eval_num_gpus=2, eval_num_gpus_per_engine=2)
        [group] = [m for m in resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert "ep_size" not in group.overrides  # left to the shared --sglang-* fill-in

    def test_eval_sglang_overrides_win_over_the_tp_coupled_defaults(self):
        args = make_args(
            rollout_num_gpus_per_engine=8,
            eval_num_gpus=2,
            eval_num_gpus_per_engine=2,
            eval_sglang_ep_size=2,
        )
        [group] = [m for m in resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert group.overrides["ep_size"] == 2

    def test_eval_sglang_overrides_reach_the_eval_group_only(self):
        args = make_args(eval_num_gpus=2, eval_sglang_mem_fraction_static=0.95)
        config = resolve_sglang_config(args)

        by_name = {m.name: m for m in config.models}
        assert by_name["eval"].server_groups[0].overrides["mem_fraction_static"] == 0.95
        assert "mem_fraction_static" not in by_name["default"].server_groups[0].overrides

    def test_yaml_eval_model_is_filled_from_cli_without_clobbering(self, tmp_path):
        """Anything the YAML leaves unset falls through to the eval CLI args."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n"
        )
        args = make_args(
            sglang_config=str(cfg_path),
            rollout_num_gpus=8,
            eval_num_gpus=2,
            eval_num_gpus_per_engine=2,
            eval_sglang_mem_fraction_static=0.95,
        )
        config = resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        # Auto-inference would give True here and put the fleet in the broadcast group.
        assert eval_model.update_weights is False
        [group] = eval_model.server_groups
        assert group.num_gpus_per_engine == 2
        assert group.overrides["mem_fraction_static"] == 0.95

    def test_yaml_group_overrides_win_over_eval_cli(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n"
            "        num_gpus_per_engine: 1\n"
            "        overrides:\n"
            "          mem_fraction_static: 0.5\n"
        )
        args = make_args(
            sglang_config=str(cfg_path), rollout_num_gpus=8, eval_num_gpus=2, eval_sglang_mem_fraction_static=0.95
        )
        config = resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        assert eval_model.server_groups[0].overrides["mem_fraction_static"] == 0.5

    async def test_probe_and_mark_dead(self) -> None:
        """recover() only restarts engines already marked stopped, so something has to mark them."""

        class _Cell:
            def __init__(self, alive: bool) -> None:
                self.is_allocated, self._alive = True, alive

            async def probe_and_mark_dead(self) -> None:
                if not self._alive:
                    self.is_allocated = False

        alive, dead = _Cell(True), _Cell(False)
        srv = RolloutServer(
            server_cells={"alive": alive, "dead": dead},
            args=SimpleNamespace(),
            context_lock=ContextLock("test"),
        )

        await srv.probe_and_mark_dead()

        assert alive.is_allocated and not dead.is_allocated

    def test_compute_rollout_offset_colocate_returns_zero(self):
        args = make_args(
            colocate=True,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=False,
        )
        assert _compute_rollout_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 4

    def test_compute_rollout_offset_shared_actor_critic(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 8

    def test_compute_megatron_num_gpus_for_actor_only(self):
        args = make_args(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            use_critic=False,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 16

    def test_compute_megatron_num_gpus_with_shared_critic(self):
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 8

    def test_compute_megatron_num_gpus_zero_when_debug_rollout_only(self):
        args = make_args(debug_rollout_only=True)
        assert _compute_megatron_num_gpus(args) == 0


class TestEngineListOrdering:
    def _server_with_cells(self, num_cells: int) -> RolloutServer:
        cells = {}
        for index in sorted(range(num_cells), key=lambda i: f"inference-engine-0-0-{i}"):
            meta = SimpleNamespace(num_gpus_per_engine=index + 1, gpu_offset=index)
            cells[f"inference-engine-0-0-{index}"] = SimpleNamespace(meta=meta, api_client=f"client-{index}")
        return RolloutServer(server_cells=cells, args=SimpleNamespace(), context_lock=_make_lock())

    @pytest.mark.asyncio
    async def test_engine_lists_are_ordered_by_gpu_offset_not_insertion(self):
        """With 12 cells inserted in string-sorted id order all three derived lists come out offset-ordered."""
        srv = self._server_with_cells(12)
        async with srv.context_lock:
            assert srv.engine_gpu_offsets == list(range(12))
            assert srv.api_clients == [f"client-{i}" for i in range(12)]
            assert srv.engine_gpu_counts == [i + 1 for i in range(12)]


class TestAddCellRollback:
    def _make_meta(self) -> ServerCellMetadata:
        return ServerCellMetadata(
            model_id="default",
            worker_type="regular",
            cell_id="inference-engine-0-0-0",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            worker_name="inference-engine-0-0-0-0",
            needs_offload=False,
            update_weights=True,
            workers_hash="pseudo-hash-0",
        )

    @pytest.mark.asyncio
    async def test_a_failed_add_still_tracks_the_cell_so_nothing_leaks(self, monkeypatch):
        """The cell is committed before init runs, so a failing init cannot orphan its health checker task."""
        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=False, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _raise_async)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta())

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_disposing_the_server_removes_every_cell_it_tracks(self, monkeypatch):
        """Controller teardown must reach each cell so its health checker task stops with it."""
        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=True, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())
            await srv.dispose()

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_successful_add_commits_the_cell(self, monkeypatch):
        """After the failure is gone the same cell id can be added normally."""
        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=False, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()


class TestAddCellInitTiming:
    @pytest.mark.asyncio
    async def test_a_disaggregated_cell_is_initialized_as_soon_as_it_appears(self, monkeypatch):
        """Nothing competes for its gpus, so waiting would only delay it becoming servable."""
        initialized: list[str] = []

        async def _record(self) -> None:
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=False, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())
            await srv.dispose()

        assert initialized == ["inference-engine-0-0-0"]

    @pytest.mark.asyncio
    async def test_a_colocated_cell_is_only_tracked_until_the_weight_update_window(self, monkeypatch):
        """Its engine may not claim gpu memory while the trainer still holds it."""
        initialized: list[str] = []

        async def _record(self) -> None:
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=True, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())

            assert initialized == []
            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()


async def _make_serving_server(monkeypatch, *, num_cells: int) -> RolloutServer:
    """A server whose cells are all registered and serving, without dialling anything."""
    router = _RecordingRouterApiClient()
    monkeypatch.setattr(RolloutServer, "_router_api_client", property(lambda self: router))

    srv = RolloutServer(
        server_cells={},
        args=SimpleNamespace(colocate=True, ft_components=[], use_miles_router=False),
        context_lock=_make_lock(),
    )
    async with srv.context_lock:
        for cell_index in range(num_cells):
            await srv.add_cell(
                ServerCellMetadata(
                    model_id="default",
                    worker_type="regular",
                    cell_id=f"default-{cell_index}",
                    num_gpus_per_engine=1,
                    gpu_offset=cell_index,
                    sglang_api_key=None,
                    worker_name=f"default-{cell_index}-0",
                    needs_offload=False,
                    update_weights=True,
                    workers_hash=f"pseudo-hash-{cell_index}",
                )
            )
            cell = srv.server_cells[f"default-{cell_index}"]
            addr_info = CellAddrInfo(
                server_url=f"http://10.0.0.{cell_index + 1}:3000{cell_index}", bootstrap_port=None, gate_url=None
            )
            await cell._register_with_router(addr_info=addr_info)
            cell._state = StateServing(addr_info=addr_info)
    return srv


class _RecordingRouterApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def add_worker(self, **kwargs) -> None:
        self.calls.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs) -> None:
        self.calls.append(("remove_worker", kwargs))


def _make_lock() -> ContextLock:
    return ContextLock("InferenceController")


async def _raise_async(self):
    raise RuntimeError("injected init failure")


async def _noop_async(self):
    return None
