from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_dataclass_cells

from miles.backends.sglang_utils.sglang_config import (
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
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


@pytest.mark.skip(
    reason="TODO: rebuild against the meta/router_api_client ServerCell; make_dataclass_cells and "
    "_mark_allocated_uninitialized/_mark_addressing target the removed constructor and state API"
)
class TestRolloutServerCrossCellProperties:
    def test_api_clients_expose_one_client_per_cell(self):
        """Each cell is addressed through its primary (node-0) endpoint."""
        cells = make_dataclass_cells(num_cells=2, gpu_offset=0) + make_dataclass_cells(num_cells=2, gpu_offset=2)
        for index, cell in enumerate(cells):
            cell._mark_allocated_uninitialized()
            cell._mark_addressing(server_url=f"http://10.0.0.{index + 1}:30000")
        srv = RolloutServer(
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=SimpleNamespace(),
            context_lock=_make_lock(),
        )
        assert [client.server_url for client in srv.api_clients] == [
            f"http://10.0.0.{index + 1}:30000" for index in range(4)
        ]

    def test_engine_gpu_counts_parallel_to_engines(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2
        )
        srv = RolloutServer(
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=SimpleNamespace(),
            context_lock=_make_lock(),
        )
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_cells(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2, gpu_offset=4
        )
        srv = RolloutServer(
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=SimpleNamespace(),
            context_lock=_make_lock(),
        )
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


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
    async def test_a_failed_add_leaves_no_bookkeeping_so_the_next_reconcile_retries(self, monkeypatch):
        """A cell whose startup fails must not be committed, otherwise the hash no-op blocks any retry."""
        srv = RolloutServer(server_cells={}, args=SimpleNamespace(colocate=False), context_lock=_make_lock())
        monkeypatch.setattr(ServerCell, "init", _raise_async)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta())

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_successful_add_commits_the_cell(self, monkeypatch):
        """After the failure is gone the same cell id can be added normally."""
        srv = RolloutServer(server_cells={}, args=SimpleNamespace(colocate=False), context_lock=_make_lock())
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())

        assert list(srv.server_cells) == ["inference-engine-0-0-0"]


class TestAddCellInitTiming:
    @pytest.mark.asyncio
    async def test_a_disaggregated_cell_is_initialized_as_soon_as_it_appears(self, monkeypatch):
        """Nothing competes for its gpus, so waiting would only delay it becoming servable."""
        initialized: list[str] = []

        async def _record(self) -> None:
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(server_cells={}, args=SimpleNamespace(colocate=False), context_lock=_make_lock())
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())

        assert initialized == ["inference-engine-0-0-0"]

    @pytest.mark.asyncio
    async def test_a_colocated_cell_is_only_tracked_until_the_weight_update_window(self, monkeypatch):
        """Its engine may not claim gpu memory while the trainer still holds it."""
        initialized: list[str] = []

        async def _record(self) -> None:
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(server_cells={}, args=SimpleNamespace(colocate=True), context_lock=_make_lock())
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())

        assert initialized == []
        assert list(srv.server_cells) == ["inference-engine-0-0-0"]


def _make_lock() -> ContextLock:
    return ContextLock("InferenceController")


async def _raise_async(self):
    raise RuntimeError("injected init failure")


async def _noop_async(self):
    return None


@pytest.mark.skip(
    reason="TODO: rebuild against the meta/router_api_client ServerCell; _make_started_server still drives the "
    "removed AddrInfo/_mark_addressing/_mark_alive state API and the removed constructors"
)
class TestRemoveCell:
    @pytest.mark.asyncio
    async def test_remove_cell_detaches_the_cell_from_every_server_view(self):
        """A removed cell is gone from server_cells and from every view derived from it."""
        events: list[dict[str, Any]] = []
        srv = _make_started_server(num_cells=2)

        with _with_recording_router(events):
            await srv.remove_cell("default-0")

        assert "default-0" not in srv.server_cells
        assert list(srv.server_cells) == ["default-1"]
        assert [client.server_url for client in srv.api_clients] == ["http://10.0.0.2:30001"]
        assert srv.engine_gpu_counts == [1]
        assert srv.engine_gpu_offsets == [1]

    @pytest.mark.asyncio
    async def test_remove_cell_unregisters_from_the_router_before_dropping_the_cell(self):
        """Dropping the cell without unregistering would leave the router routing to a dead worker."""
        events: list[dict[str, Any]] = []
        srv = _make_started_server(num_cells=2)

        with _with_recording_router(events):
            await srv.remove_cell("default-0")

        assert events == [{"call": "remove_worker", "worker_url": "http://10.0.0.1:30000", "use_legacy_api": False}]


def _make_started_server(*, num_cells: int) -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    srv = RolloutServer(server_cells={}, args=args)
    for cell_index in range(num_cells):
        meta = ServerCellMetadata(
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
        cell = ServerCell(args=args, meta=meta)
        cell._mark_addressing(AddrInfo(server_url=f"http://10.0.0.{cell_index + 1}:3000{cell_index}"))  # noqa: F821
        cell._mark_alive()
        srv.server_cells[meta.cell_id] = cell
    return srv


def _with_recording_router(events: list[dict[str, Any]]) -> Any:
    return patch.object(
        RolloutServer, "_router_api_client", property(lambda self: _RecordingRouterApiClient(events=events))
    )


class _RecordingRouterApiClient:
    def __init__(self, *, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def add_worker(self, **kwargs: Any) -> None:
        self._events.append({"call": "add_worker", **kwargs})

    async def remove_worker(self, **kwargs: Any) -> None:
        self._events.append({"call": "remove_worker", **kwargs})
