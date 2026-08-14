import asyncio
import dataclasses
import logging
from typing import Any, NamedTuple

import ray

from miles.backends.sglang_utils.arguments import collect_eval_sglang_overrides
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.router_manager import start_router
from miles.ray.rollout.server_cell import ServerCell, compute_nodes_per_engine
from miles.ray.rollout.server_engine import ServerEngine
from miles.utils import async_utils

logger = logging.getLogger(__name__)


def start_rollout_servers(args, pg) -> dict[str, "RolloutServer"]:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_cells: list[ServerCell] = []
        port_allocator = PortAllocator()

        for group_cfg in model_cfg.server_groups:
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local
            nodes_per_engine = compute_nodes_per_engine(
                num_gpus_per_engine=gpus_per_engine, num_gpus_per_node=args.num_gpus_per_node
            )

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            if group_cfg.worker_type != "placeholder":
                assert num_engines % nodes_per_engine == 0, (
                    f"group '{group_cfg.worker_type}' has {num_engines=} which is not a whole number of "
                    f"{nodes_per_engine=} engines; the trailing engine would have no node to run its remaining ranks"
                )
                assert engine_offset % nodes_per_engine == 0, (
                    f"group '{group_cfg.worker_type}' starts at {engine_offset=}, which is not aligned to "
                    f"{nodes_per_engine=}: sglang derives each engine's node_rank from its global rank, so a "
                    f"misaligned start would make the cell's primary a worker node"
                )

                for cell_start in range(0, num_engines, nodes_per_engine):
                    server_cells.append(
                        ServerCell(
                            args=args,
                            worker_type=group_cfg.worker_type,
                            engines=[ServerEngine() for _ in range(nodes_per_engine)],
                            pg=pg,
                            num_gpus_per_engine=gpus_per_engine,
                            rank_offset=engine_offset + cell_start,
                            gpu_offset=gpu_offset + cell_start * num_gpu_per_engine_local,
                            sglang_overrides=overrides,
                            needs_offload=needs_offload,
                            model_path=overrides.get("model_path", args.hf_checkpoint),
                            update_weights=model_cfg.update_weights,
                        )
                    )

            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus

        srv = RolloutServer(
            server_cells=server_cells,
            args=args,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )
        async_utils.run(srv.start_all_cells(port_allocator))
        servers[model_cfg.name] = srv

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _eval_sglang_overrides(args) -> dict:
    """Eval-fleet engine settings; anything absent is inherited from the rollout engines."""
    overrides = {
        # Eval samples never feed training, so the replay side-channels are pure overhead.
        "enable_return_routed_experts": False,
        "enable_return_indexer_topk": False,
    }
    if args.eval_num_gpus_per_engine != args.rollout_num_gpus_per_engine:
        # Inheriting these across a different tp gives an engine SGLang refuses to boot.
        tp_coupled = ("dp_size", "pp_size", "ep_size", "attn_cp_size")
        overrides |= dict.fromkeys(tp_coupled, 1)
        logger.info(
            f"Eval tp={args.eval_num_gpus_per_engine} != rollout tp={args.rollout_num_gpus_per_engine}; "
            f"{', '.join(tp_coupled)} default to 1. Override with --eval-sglang-*."
        )
    return overrides | collect_eval_sglang_overrides(args)


def _apply_eval_model_config(model_cfg: ModelConfig, args) -> None:
    """Fill the eval model from the ``--eval-*`` args: YAML > ``--eval-sglang-*`` > ``--sglang-*``."""
    if model_cfg.update_weights is None:
        # Never joins the training broadcast group; the fleet is synced by snapshot only.
        model_cfg.update_weights = False
    overrides = _eval_sglang_overrides(args)
    for group in model_cfg.server_groups:
        if group.num_gpus_per_engine is None:
            group.num_gpus_per_engine = args.eval_num_gpus_per_engine
        group.overrides = overrides | group.overrides


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    eval_num_gpus = args.eval_num_gpus

    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        expected = args.rollout_num_gpus + eval_num_gpus
        actual = config.total_num_gpus
        assert (
            actual == expected
        ), f"sglang_config total GPUs ({actual}) != rollout_num_gpus + eval_num_gpus ({expected})"
        if eval_num_gpus > 0:
            eval_models = [m for m in config.models if m.name == "eval"]
            assert len(eval_models) == 1 and eval_models[0].total_num_gpus == eval_num_gpus, (
                f"--eval-num-gpus {eval_num_gpus} requires the sglang_config YAML to contain "
                f"exactly one model named 'eval' with that many GPUs."
            )
            _apply_eval_model_config(eval_models[0], args)
        return config

    if args.prefill_num_servers is not None:
        config = SglangConfig.from_prefill_num_servers(args)
    else:
        config = SglangConfig(
            models=[
                ModelConfig(
                    name="default",
                    server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
                )
            ]
        )

    if eval_num_gpus > 0:
        eval_model = ModelConfig(
            name="eval",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=eval_num_gpus)],
        )
        _apply_eval_model_config(eval_model, args)
        config.models.append(eval_model)
    return config


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if getattr(args, "debug_rollout_only", False):
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, as a list of engine cells.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: list[ServerCell]
    args: Any
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool = False
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    _port_allocator: PortAllocator = dataclasses.field(default_factory=PortAllocator)

    @property
    def engines(self) -> list[ServerEngine]:
        """All node-0 engines across all cells."""
        return [cell.primary_engine for cell in self.server_cells]

    def clear_has_new_engines(self):
        self.has_new_engines = False

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.num_gpus_per_engine for cell in self.server_cells]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.gpu_offset for cell in self.server_cells]

    async def probe_and_mark_dead(self):
        """Mark unreachable engines stopped so ``recover`` restarts them.

        For servers without a ``RolloutHealthMonitor``, which does the same job.
        """
        for cell in self.server_cells:
            for engine in cell.engines:
                if not engine.is_allocated:
                    continue
                try:
                    await asyncio.wait_for(engine.actor_handle.get_weight_version.remote(), timeout=60)
                except Exception as e:
                    logger.warning(f"Engine unreachable ({e!r}); marking stopped for recovery")
                    try:
                        ray.kill(engine.actor_handle)
                    except Exception:
                        pass
                    engine.mark_stopped()

    async def start_all_cells(self, port_allocator: PortAllocator):
        if self.args.debug_train_only:
            return

        self._port_allocator = port_allocator
        cell_indices = [cell_index for cell_index, cell in enumerate(self.server_cells) if not cell.is_allocated]
        await asyncio.gather(
            *[
                self.server_cells[cell_index].start(port_allocator, self._router_api_client)
                for cell_index in cell_indices
            ]
        )
        self.has_new_engines |= bool(cell_indices)

    async def recover(self, cell_indices: list[int] | None = None):
        """Recover dead cells, overlapping init across cells.

        Reuses the startup allocator so its per-node cursors still sit past the
        ports the live engines hold, instead of rescanning from the base port.
        """
        port_allocator = self._port_allocator
        if cell_indices is None:
            cell_indices = list(range(len(self.server_cells)))
        cell_indices = [cell_index for cell_index in cell_indices if not self.server_cells[cell_index].is_allocated]

        await asyncio.gather(
            *[
                self.server_cells[cell_index].start(port_allocator, self._router_api_client, recover=True)
                for cell_index in cell_indices
            ]
        )
        self.has_new_engines |= bool(cell_indices)

        logger.info(f"Recovered {len(cell_indices)} dead rollout cells")

    async def stop_cells(self, cell_indices: list[int]):
        logger.info(f"Killing server {cell_indices=}...")
        for cell_index in sorted(set(cell_indices)):
            await self.server_cells[cell_index].stop(self._router_api_client)

    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._allocated_cells_of() if cell.needs_offload]
        )

    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._allocated_cells_of() if cell.needs_offload]
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self._allocated_cells_of()
            ]
        )

    async def wait_all_engines_alive(self, timeout: float = 600):
        # TODO: 600s default is hardcoded; make it configurable (e.g. via args) once we have a clearer
        # picture of init/recovery upper bounds across model sizes
        sleep_time = 2
        for _ in range(int(timeout // sleep_time)):
            if all(e.is_alive for cell in self.server_cells for e in cell.engines):
                return
            await asyncio.sleep(sleep_time)
            logger.info("wait_all_engines_alive looping...")
        raise TimeoutError(f"Timed out after {timeout}s waiting for engines to become ready")

    def _allocated_cells_of(self, cell_indices: list[int] | None = None) -> list[ServerCell]:
        if cell_indices is None:
            cell_indices = range(len(self.server_cells))
        return [
            self.server_cells[cell_index] for cell_index in cell_indices if self.server_cells[cell_index].is_allocated
        ]

    @property
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


class CellIndexer(NamedTuple):
    srv_key: str
    cell_index: int


def get_cell_indexer_of_id_map(servers: dict[str, RolloutServer]) -> list[CellIndexer]:
    """Flatten ``servers`` into a list whose position is the cell id.

    ``cell_index`` is the cell's position within its server. Order is sorted by
    ``srv_key``, so cell ids are stable across calls when the topology is
    unchanged.
    """
    result: list[CellIndexer] = []
    for srv_key in sorted(servers):
        srv = servers[srv_key]
        for cell_index in range(len(srv.server_cells)):
            result.append(CellIndexer(srv_key=srv_key, cell_index=cell_index))
    return result
