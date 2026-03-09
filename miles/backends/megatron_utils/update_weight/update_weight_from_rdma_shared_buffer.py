import dataclasses
import logging
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import ray
import torch
from mooncake.engine import TransferEngine
from ray.actor import ActorHandle
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.server_args import ServerArgs
from tqdm import tqdm

from miles.utils.memory_utils import print_memory
from miles.utils.timer import timer

from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def register_cpu_memory_region(params_dict: dict, transfer_engine: TransferEngine) -> dict:
    """Register CPU pinned memory with the transfer engine.

    Returns:
        dict: name -> (data_ptr, numel, element_size) for each registered tensor.
    """
    start_tic = time.time()
    weight_mr_dict = {}

    for name, cpu_tensor in params_dict.items():
        addr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel() * cpu_tensor.element_size()
        ret = transfer_engine.register_memory(addr, size)
        if ret != 0:
            raise RuntimeError(f"register CPU memory failed for weight {name}, error: {ret}")
        weight_mr_dict[name] = (addr, cpu_tensor.numel(), cpu_tensor.element_size())

    elapsed = time.time() - start_tic
    logger.info(f"[RDMA] Registered {len(weight_mr_dict)} CPU tensors in {elapsed:.2f}s")
    return weight_mr_dict


def create_transfer_engine() -> TransferEngine:
    transfer_engine = TransferEngine()
    local_ip = ray._private.services.get_node_ip_address()
    transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
    logger.info(f"[RDMA] Transfer Engine initialized at port {transfer_engine.get_rpc_port()}")
    return transfer_engine


def create_cpu_replica(
    parallelism_config: RankParallelismConfig,
    model_path: str,
    server_args: ServerArgs,
) -> torch.nn.Module:
    """Create model on GPU (required by sglang), then move to CPU pinned memory."""
    load_config = LoadConfig(
        load_format="dummy",
        model_loader_extra_config=None,
        rl_quant_profile=server_args.rl_quant_profile,
    )
    server_args_module._global_server_args = server_args
    with ParallelismContext(parallelism_config):
        model = get_model(
            model_config=ModelConfig(model_path),
            load_config=load_config,
            device_config=DeviceConfig(),
        )

    gpu_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[RDMA] GPU model created: {gpu_params} params")

    # Move all parameters to CPU pinned memory
    with timer("rdma_move_replica_to_cpu"):
        for param in model.parameters():
            cpu_data = param.data.to("cpu", non_blocking=True).pin_memory()
            param.data = cpu_data
        torch.cuda.synchronize()

    torch.cuda.empty_cache()

    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f"[RDMA] CPU pinned replica: {gpu_params} params, " f"{total_bytes / (1024**3):.2f} GB")
    print_memory("[RDMA] After moving replica to CPU and freeing GPU")
    return model


def query_remote_weight_infos(
    rollout_engines: Sequence[ActorHandle],
    targets,
) -> tuple[dict, dict, dict]:
    """Query remote rollout engines for weight info, session IDs, and server args.

    Returns:
        (remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args)
    """
    remote_weight_infos_by_session_id = {}
    targets_to_session_id = {}
    session_id_to_server_args = {}
    targets_to_query = set((target.engine_ind, target.engine_rank) for target in targets)

    for engine_ind, engine_rank in targets_to_query:
        session_id, weights_info = ray.get(
            rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
        )
        parallelism_info = ray.get(rollout_engines[engine_ind].get_parallelism_info.remote(rank=engine_rank))

        session_id_to_server_args[session_id] = create_server_args_from_dict(
            ray.get(rollout_engines[engine_ind].get_server_info.remote())
        )
        assert session_id is not None, f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
        logger.info(f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}")
        logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
        remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
        targets_to_session_id[(engine_ind, engine_rank)] = session_id

    return remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args


@dataclasses.dataclass
class RemoteWeightInfo:
    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


class RDMATransferManager:
    """Generic async task manager for RDMA writes.

    Accepts arbitrary callables via submit(), runs them in a thread pool,
    and tracks futures for bulk waiting.
    """

    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.executor: ThreadPoolExecutor | None = None
        self.transfer_futures: list[Future] = []

    def ensure_started(self) -> None:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def submit(self, fn: Callable, *args) -> None:
        """Submit a callable to the thread pool."""
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)

    def submit_returning_future(self, fn: Callable, *args) -> Future:
        """Submit a callable and return its future (also tracked for bulk waiting)."""
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)
        return future

    def wait_transfers(self) -> None:
        """Wait for all submitted tasks to complete."""
        for future in self.transfer_futures:
            try:
                future.result(timeout=30.0)
            except Exception as e:
                logger.error(f"[RDMA] Transfer future failed: {e}")

        self.transfer_futures.clear()


@dataclasses.dataclass
class EngineRankInfo:
    """Per-engine-rank metadata: unique model replica (weight_loaders), shared CPU pinned buffers."""

    engine_rank: int
    model_replica: torch.nn.Module  # shares CPU pinned buffers, has unique weight_loaders
    remote_weight_infos: list[RemoteWeightInfo]

    def add_remote_session(self, remote_info: RemoteWeightInfo) -> None:
        self.remote_weight_infos.append(remote_info)



class UpdateWeightFromRDMASharedBuffer(UpdateWeightFromRemote):
    """RDMA weight transfer using a single set of shared CPU pinned buffers.

    Architecture:
    - One set of CPU pinned buffers shared across all engine ranks (saves ~N× CPU memory)
    - One TransferEngine for all engine ranks
    - Lightweight replicas per engine rank share the pinned storage but have unique
      weight_loaders (so load_weights slices the correct shard for each rank)
    - Transfers are serialized across engine ranks because the shared buffer is
      overwritten for each rank's shard, EXCEPT the last engine rank's write which
      runs in a background thread (overlaps with the next bucket's all-gather)

    Per-bucket flow:
      1. compute transfer_ready_params once (same for all engine ranks)
      2. for each engine rank:
           load_weights(shared buffer) → RDMA write
         where the last rank's write is submitted to a background thread
      3. wait_transfers() at finish to collect all background writes
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            weight_update_mode="rdma",
        )

        self._registered = False
        self._update_pending: dict[str, int] = {}
        # Staging buffer: accumulate HF tensors across buckets until all shards
        # for a sglang param are collected. Key = sglang param name, value = list
        # of (hf_name, tensor) tuples.
        self._staged_tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
        num_workers = getattr(args, "rdma_transfer_workers", 4)
        self.transfer_manager = RDMATransferManager(num_workers=num_workers)

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            targets = self.transfer_plan.plan_p2p()
            (
                self.remote_weight_infos_by_session_id,
                targets_to_session_id,
                self.session_id_to_server_args,
            ) = query_remote_weight_infos(rollout_engines, targets)

            print_memory("[RDMA-Shared] After obtaining remote weight info")

            # Group targets by engine rank
            engine_rank_targets: dict[int, list] = {}
            for target in targets:
                engine_rank_targets.setdefault(target.engine_rank, []).append(target)

            # Create ONE transfer engine for all engine ranks
            self._engine = create_transfer_engine()
            self._engine_rank_infos: dict[int, EngineRankInfo] = {}
            self._shared_params_dict: dict[str, torch.Tensor] = {}
            self._shared_param_mapper: ParameterMapper | None = None

            first_engine_rank = True
            for engine_rank, rank_targets in engine_rank_targets.items():
                # Get parallelism config from first target of this engine rank
                first_target = rank_targets[0]
                session_id = targets_to_session_id[(first_target.engine_ind, first_target.engine_rank)]
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                server_args = self.session_id_to_server_args[session_id]

                if first_engine_rank:
                    # First engine rank: create full CPU pinned replica → shared buffers
                    logger.info(f"[RDMA-Shared] Creating shared CPU pinned replica from engine rank {engine_rank}")
                    model_replica = create_cpu_replica(parallelism_config, self.args.hf_checkpoint, server_args)
                    self._shared_params_dict = dict(model_replica.named_parameters())
                    self._shared_param_mapper = ParameterMapper.from_model(model_replica)
                    print_memory(f"[RDMA-Shared] After shared CPU pinned replica for engine rank {engine_rank}")
                    first_engine_rank = False
                else:
                    # Subsequent engine ranks: create lightweight replica sharing pinned buffers
                    logger.info(f"[RDMA-Shared] Creating lightweight replica for engine rank {engine_rank}")
                    model_replica = self._create_lightweight_replica(
                        parallelism_config, self.args.hf_checkpoint, server_args
                    )
                    print_memory(f"[RDMA-Shared] After lightweight replica for engine rank {engine_rank}")

                # Collect remote sessions for this engine rank
                remote_infos = []
                for target in rank_targets:
                    sid = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                    remote_infos.append(RemoteWeightInfo(sid, self.remote_weight_infos_by_session_id[sid][0]))

                self._engine_rank_infos[engine_rank] = EngineRankInfo(
                    engine_rank=engine_rank,
                    model_replica=model_replica,
                    remote_weight_infos=remote_infos,
                )

            # Materialize as list for indexed access (last-rank detection)
            self._engine_rank_list = list(self._engine_rank_infos.values())

            print_memory("[RDMA-Shared] After all replicas and engine creation")

    def _create_lightweight_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args,
    ) -> torch.nn.Module:
        """Create model on GPU (different weight_loaders), then point params to shared CPU pinned buffers.

        The model object (layers, weight_loaders) stays alive but shares the underlying
        storage with the first replica's CPU pinned buffers. No new CPU allocation.
        """
        load_config = LoadConfig(
            load_format="dummy",
            model_loader_extra_config=None,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args_module._global_server_args = server_args
        with ParallelismContext(parallelism_config):
            model = get_model(
                model_config=ModelConfig(model_path),
                load_config=load_config,
                device_config=DeviceConfig(),
            )

        gpu_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[RDMA-Shared] Lightweight replica GPU model created: {gpu_params} params")

        # Point all params to shared pinned buffers (no new CPU allocation)
        for name, param in model.named_parameters():
            if name not in self._shared_params_dict:
                logger.warning(f"[RDMA-Shared] Parameter {name} not found in shared buffers, skipping")
                continue
            param.data = self._shared_params_dict[name]

        torch.cuda.empty_cache()
        logger.info(
            f"[RDMA-Shared] Lightweight replica: {gpu_params} params, "
            f"sharing {len(self._shared_params_dict)} CPU pinned buffers"
        )
        print_memory("[RDMA-Shared] After lightweight replica (shared buffers, GPU freed)")
        return model

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=str(self.weight_version))
                for engine in self.rollout_engines
            ]
        )

    def on_transfer_start(self) -> None:
        """Register shared CPU pinned memory with RDMA on first call."""
        if not self._is_source:
            return

        if not self._registered:
            with timer("rdma_cpu_registration"):
                self._weight_memory_registry = register_cpu_memory_region(self._shared_params_dict, self._engine)
            self._registered = True

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Stage incoming tensors; when all shards for a param are collected,
        load into shared buffer and RDMA-write per engine rank.

        Only calls load_weights() with complete accumulated tensors, preventing
        partial writes that would corrupt the shared buffer when different engine
        ranks have different EP expert-to-local mappings.
        """
        if not self._is_source or not converted_named_tensors:
            return

        # Stage tensors and check which params are now complete
        with timer("get_transfer_ready_params", log_info=False):
            transfer_ready_params, ready_hf_tensors = self._get_transfer_ready_params(converted_named_tensors)

        # Only proceed if we have fully-collected params to transfer
        if transfer_ready_params and ready_hf_tensors:
            last_idx = len(self._engine_rank_list) - 1
            for i, info in enumerate(self._engine_rank_list):
                with timer("load_weights_to_cpu_replica", log_info=False):
                    info.model_replica.load_weights(ready_hf_tensors)

                is_last = i == last_idx
                if is_last:
                    # Last engine rank: fire-and-forget all sessions to background
                    with timer("rdma_async_write", log_info=False):
                        for remote_session in info.remote_weight_infos:
                            self.transfer_manager.submit(
                                self._do_rdma_write_one_session,
                                info,
                                remote_session,
                                transfer_ready_params,
                            )
                else:
                    # Non-last rank: fan out sessions in parallel, then wait
                    # (must complete before next load_weights overwrites buffer)
                    with timer("rdma_sync_write", log_info=False):
                        futures = [
                            self.transfer_manager.submit_returning_future(
                                self._do_rdma_write_one_session,
                                info,
                                remote_session,
                                transfer_ready_params,
                            )
                            for remote_session in info.remote_weight_infos
                        ]
                        for f in futures:
                            f.result()

        # Clear the input list (caller convention)
        converted_named_tensors.clear()

    def _get_transfer_ready_params(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[str], list[tuple[str, torch.Tensor]]]:
        """Determine which sglang params have all shards present, returning their accumulated tensors.

        Stages incoming HF tensors in self._staged_tensors until all shards for a
        sglang param are collected. Only returns tensors for fully-ready params,
        preventing partial load_weights() calls that would corrupt the shared buffer.

        Returns:
            (transfer_ready_param_names, ready_hf_tensors):
            - transfer_ready_param_names: sglang param names ready for RDMA transfer
            - ready_hf_tensors: complete list of (hf_name, tensor) tuples for load_weights()
        """
        transfer_ready_params = []
        params_dict = self._shared_params_dict

        for name, tensor in converted_named_tensors:
            mapped_result = self._shared_param_mapper.map(name)
            mapped, num_shards, num_experts = (
                mapped_result.sglang_name,
                mapped_result.num_shards,
                mapped_result.num_local_experts,
            )
            if mapped not in params_dict:
                logger.warning(f"Parameter {mapped} not found in shared model replica.")
                continue

            if num_experts is not None and num_experts > 0:
                total_expected = num_experts * num_shards
            else:
                total_expected = num_shards

            # Stage the tensor
            self._staged_tensors.setdefault(mapped, []).append((name, tensor))

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                if mapped not in self._update_pending:
                    self._update_pending[mapped] = total_expected - 1
                else:
                    self._update_pending[mapped] -= 1
                if self._update_pending[mapped] == 0:
                    transfer_ready_params.append(mapped)

        # Collect all staged HF tensors for ready params
        ready_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for param_name in transfer_ready_params:
            staged = self._staged_tensors.pop(param_name, [])
            ready_hf_tensors.extend(staged)
            self._update_pending.pop(param_name, None)

        return transfer_ready_params, ready_hf_tensors

    def _do_rdma_write_one_session(
        self, info: EngineRankInfo, remote_session: RemoteWeightInfo, names: list[str]
    ) -> None:
        """RDMA write from shared CPU pinned buffers to a single remote session.

        Used by the parallelized submission path where each session within an
        engine rank is submitted as a separate task to RDMATransferManager.
        """
        source_ptrs, source_lens = [], []
        valid_names = []

        for name in names:
            cpu_reg = self._weight_memory_registry.get(name)
            if cpu_reg is None:
                continue

            data_ptr, numel, ele_size = cpu_reg
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)
            valid_names.append(name)

        if not source_ptrs:
            return

        session_id = remote_session.session_id
        target_ptrs = []
        for name in valid_names:
            if name in remote_session.weights_info:
                target_ptrs.append(remote_session.weights_info[name][0])

        if len(target_ptrs) != len(source_ptrs):
            logger.warning(f"[RDMA-Shared] Pointer count mismatch for session {session_id}")
            return

        ret = self._engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            logger.error(f"[RDMA-Shared] Transfer failed for session {session_id}, error: {ret}")

    def finish_transfer_task(self) -> None:
        """Wait for all background RDMA writes to complete."""
        if not self._is_source:
            return
        self.transfer_manager.wait_transfers()
        self._update_pending = {}
        if self._staged_tensors:
            logger.warning(
                f"[RDMA-Shared] Staging buffer not empty at end of transfer: "
                f"{len(self._staged_tensors)} params with incomplete shards: "
                f"{list(self._staged_tensors.keys())}"
            )
            self._staged_tensors.clear()
        logger.info("[RDMA-Shared] All transfers complete")
