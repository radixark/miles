import dataclasses
import logging
import queue
import threading
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

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
from sglang.srt.model_loader.remote_instance_weight_loader_utils import register_memory_region_v2
from sglang.srt.server_args import ServerArgs
from tqdm import tqdm

from miles.utils.memory_utils import print_memory

from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    # Reconstruct Sglang ServerArgs from sglang Http query.
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


@dataclasses.dataclass
class RemoteWeightInfo:
    # Remote session and weight registration info.
    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


@dataclasses.dataclass
class TransferTask:
    """Represents a queued RDMA transfer task."""

    session_id: str
    source_ptrs: list[int]
    target_ptrs: list[int]
    source_lens: list[int]
    engine: TransferEngine


class ExecutableQueue:
    """
    Asynchronous queue for executing transfer_bundle.execute_each() operations.
    Allows overlapping weight loading with RDMA transfer execution.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._background_thread = None
        self._shutdown_event = threading.Event()
        self._tasks_completed = threading.Event()
        self._active_tasks = 0
        self._lock = threading.Lock()
        self._cleanup_requested = threading.Event()
        self._cleanup_completed = threading.Event()
        self._active_transferring_engine_batch_ids = {}

    def _background_worker(self):
        """Background thread worker that processes queued transfer tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Check if cleanup is requested
                # It's necessary to `get_batch_transfer_status` in the same thread.
                if self._cleanup_requested.is_set():
                    logger.info("[RDMA Worker Thread] Cleanup requested, freeing batch_ids...")
                    self._perform_cleanup()
                    self._cleanup_requested.clear()
                    self._cleanup_completed.set()
                    logger.info("[RDMA Worker Thread] Cleanup completed")
                    continue

                # Get task with timeout to allow periodic shutdown checks
                task = self._queue.get(timeout=0.1)
                try:
                    # Execute the RDMA transfer
                    logger.info(f"[RDMA] Executing transfer task for session {task.session_id}...")
                    ret = task.engine.batch_transfer_async_write(
                        task.session_id, task.source_ptrs, task.target_ptrs, task.source_lens
                    )
                    logger.info(f"[RDMA] Executing transfer task for session {task.session_id} done")
                    self._active_transferring_engine_batch_ids[task.engine] = (
                        self._active_transferring_engine_batch_ids.get(task.engine, []) + [ret]
                    )
                    logger.info(f"[RDMA] saving batch id {ret} for task {task.session_id} ")
                    if ret < 0:
                        logging.error(f"RDMA transfer failed with error code {ret} for session {task.session_id}")
                finally:
                    self._queue.task_done()
                    with self._lock:
                        self._active_tasks -= 1
                        if self._active_tasks == 0:
                            self._tasks_completed.set()

            except queue.Empty:
                continue

    def _perform_cleanup(self):
        """Cleanup batch_ids in the same thread that allocated them (critical for thread-local cache)."""
        all_freed = True
        for e in list(self._active_transferring_engine_batch_ids.keys()):
            batch_ids = self._active_transferring_engine_batch_ids[e]
            if len(batch_ids) > 0:
                logger.info(f"[RDMA Worker Thread] Attempting to free {len(batch_ids)} batch_ids ")
                result = e.get_batch_transfer_status(batch_ids)
                if result >= 0:
                    # Successfully freed
                    self._active_transferring_engine_batch_ids[e] = []
                    logger.info(f"[RDMA Worker Thread] Successfully freed {len(batch_ids)} batch_ids")
                else:
                    logger.warning(f"[RDMA Worker Thread] get_batch_transfer_status returned {result}, will retry...")
                    all_freed = False

        assert (
            all_freed
        ), "[RDMA Worker Thread] Not all batch_ids successfully freed, the transferred batches may be too big"

    def start(self):
        """Start the background worker thread."""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._shutdown_event.clear()
            self._tasks_completed.clear()
            self._background_thread = threading.Thread(target=self._background_worker, daemon=True)
            self._background_thread.start()

    def enqueue_task(self, task: TransferTask):
        """Add a transfer task to the queue."""
        with self._lock:
            self._active_tasks += 1
            self._tasks_completed.clear()
        self._queue.put(task)

    def wait_all_complete(self, timeout=30.0):
        """Wait for all queued tasks to complete before proceeding."""
        if self._active_tasks == 0:
            return True

        # Wait for the completion event first
        if not self._tasks_completed.wait(timeout):
            return False

        # Wait for queue to be fully processed
        try:
            self._queue.join()
        except Exception as e:
            logging.error(f"Error during queue join: {e}")
            return False

        # Request cleanup from the background worker thread
        # This is CRITICAL because ThreadLocalSliceCache in TransferEngine is thread-local.Slices allocated in the worker thread MUST be freed in the same thread.
        logger.info("[RDMA] Requesting batch_id cleanup from worker thread...")

        self._cleanup_completed.clear()
        self._cleanup_requested.set()

        # Wait for worker thread to complete cleanup
        if not self._cleanup_completed.wait(timeout=60.0):
            logger.error("[RDMA] Cleanup timeout! Worker thread did not respond.")
            raise RuntimeError("[RDMA] Worker thread failed to complete batch_id cleanup")

        logger.info("[RDMA] Batch_id cleanup completed successfully")
        return True

    def shutdown(self):
        """Shutdown the background worker thread."""
        self._shutdown_event.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)


@dataclasses.dataclass
class TransferBundle:
    model_replica: Sequence[torch.nn.Module]
    engine: TransferEngine
    weight_memory_registry: dict
    remote_weight_infos: list[RemoteWeightInfo]
    _cached_params_dict: dict = dataclasses.field(default_factory=dict)
    # Local buffer to check for parameter readiness before transfer
    _update_pending: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def params_dict(self):
        if not self._cached_params_dict:
            self._cached_params_dict = dict(self.model_replica.named_parameters())
            # logger.info("Full param list: " + str(list(self._cached_params_dict.keys())))
        return self._cached_params_dict

    def reset(self):
        self._update_pending = {}

    def add_remote_session(self, remote_info: RemoteWeightInfo) -> None:
        self.remote_weight_infos.append(remote_info)

    def get_transfer_ready_params(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> list[str]:
        transfer_ready_params = []
        for name, _ in converted_named_tensors:
            mapped, shard, num_shards, expert, num_experts = self.model_replica.map_weight_name(name)
            if mapped not in self.params_dict:
                logger.warning(f"Parameter {mapped} not found in model replica.")
                continue

            # Calculate total expected contributions for this parameter
            if num_experts > 0:
                # Expert weight: need all experts * shard types
                # For w13_weight (gate+up): shard is "w1" or "w3", multiplier = 2
                # For w2_weight (down): shard is "w2", multiplier = 1
                if shard in ("w1", "w3"):
                    total_expected = num_experts * 2  # Both gate and up projections
                else:  # "w2"
                    total_expected = num_experts
            else:
                # Non-expert weight: just count shards
                total_expected = num_shards

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                # logger.info(f"Sharded param {name} mapped to {mapped} shard {shard}, expert {expert}, expecting {total_expected}")
                if mapped not in self._update_pending:
                    self._update_pending[mapped] = total_expected - 1
                else:
                    self._update_pending[mapped] -= 1
                if self._update_pending[mapped] == 0:
                    transfer_ready_params.append(mapped)
        return transfer_ready_params

    def execute_each(self, names: Sequence[str], executable_queue: ExecutableQueue = None) -> None:
        """
        Execute transfer for specific parameter names.
        If executable_queue is provided, tasks are queued for async execution.
        Otherwise, falls back to immediate execution for backward compatibility.
        """
        # Find local pointers and lengths for the given names
        source_ptrs, source_lens = [], []
        for name in names:
            if name in self._update_pending:
                assert self._update_pending[name] == 0, f"Parameter {name} is not ready for transfer."
            tensor_register = self.weight_memory_registry[name]
            data_ptr, numel, ele_size = tensor_register
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)

        # Match with remote sessions and target pointers
        for remote_session in self.remote_weight_infos:
            session_id, remote_weights_info = remote_session.session_id, remote_session.weights_info
            target_ptrs = []
            for name in names:
                target_ptrs.append(remote_weights_info[name][0])  # remote address

            if executable_queue is not None:
                # Queue the transfer task for async execution
                task = TransferTask(
                    session_id=session_id,
                    source_ptrs=source_ptrs.copy(),  # TODO:copy necessary or not?
                    target_ptrs=target_ptrs.copy(),
                    source_lens=source_lens.copy(),
                    engine=self.engine,
                )
                executable_queue.enqueue_task(task)
            else:
                # Immediate execution (backward compatibility)
                _ = self.engine.batch_transfer_async_write(session_id, source_ptrs, target_ptrs, source_lens)

    def execute(self) -> None:
        # Execute transfer for each target session using this replica.
        for remote_session in self.remote_weight_infos:
            session_id, remote_weights_info = remote_session.session_id, remote_session.weights_info
            source_ptrs, target_ptrs, source_lens = [], [], []
            for name, tensor_register in self.weight_memory_registry.items():
                data_ptr, numel, ele_size = tensor_register
                source_ptrs.append(data_ptr)
                target_ptrs.append(remote_weights_info[name][0])  # remote address
                source_lens.append(numel * ele_size)

            # Batch transfer weights through RDMA
            ret = self.engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
            if ret < 0:
                raise RuntimeError(f"Batch transfer weights via RDMA failed with error code {ret}.")


class UpdateWeightFromRDMA(UpdateWeightFromRemote):
    """
    Update weights from RDMA using Transfer Engine.

    Similar to UpdateWeightFromNCCL but uses P2P RDMA transfer engine for the underlying weight transfer. Workflow
    consists of following steps:
    1. Based off the transfer plan, query the target rollout engines for remote session and weight info during connect_rollout_engines.
    2. Construct local model replica according to the plan and attach target session id and weight memory registry
    2. Do TP-EP all-gather for bucketed weights on parameters needing transfer from local just as in NCCL case.
    3. Convert the gathered HF tensor into target shape and register them with Engine.
    4. Call engine to batch transfer weights for each transfer task.
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

        # For torch memory saver tagging
        self.tag = f"Model Replica {self.global_rank}"
        self._model_on_cpu = False
        self.pipelined_transfer = args.rdma_pipelined_transfer

        # Initialize executable queue for async transfer operations
        self.executable_queue = ExecutableQueue()
        self.executable_queue.start()

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Initialize P2PTrainingTransferEngine if serves as a source.
        """
        # Store rollout engines and lock
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            # Query Engine session and weight info from rollout instances according to the transfer plan
            self.remote_weight_infos_by_session_id = {}
            targets = self.transfer_plan.plan_p2p()
            targets_to_query = set((target.engine_ind, target.engine_rank) for target in targets)
            targets_to_session_id, self.session_id_to_engine_rank = {}, {}
            self.session_id_to_server_args = {}
            for engine_ind, engine_rank in targets_to_query:
                session_id, weights_info = ray.get(
                    self.rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
                )
                parallelism_info = ray.get(
                    self.rollout_engines[engine_ind].get_parallelism_info.remote(rank=engine_rank)
                )

                self.session_id_to_engine_rank[session_id] = engine_rank
                self.session_id_to_server_args[session_id] = create_server_args_from_dict(
                    ray.get(self.rollout_engines[engine_ind].get_server_info.remote())
                )
                assert (
                    session_id is not None
                ), f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
                logger.info(
                    f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
                # logger.info(list(weights_info.keys()))
                self.remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
                targets_to_session_id[(engine_ind, engine_rank)] = session_id

            print_memory("[RDMA] After obtaining remote weight info")

            # Create local model replicas and transfer engines for each target rollout shard
            self.engines = {}
            # Associate transfer tasks based on obtained session and weight info
            for target in targets:
                session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                remote_info = RemoteWeightInfo(session_id, self.remote_weight_infos_by_session_id[session_id][0])
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                if target.engine_rank not in self.engines:
                    transfer_engine = self._create_transfer_engine()
                    logger.info(f"[RDMA] Creating model replica for engine rank {target.engine_rank}")
                    model_replica = self._create_inference_replica(
                        parallelism_config, self.args.hf_checkpoint, self.session_id_to_server_args[session_id]
                    )
                    print_memory(f"[RDMA] After model replica at {target.engine_rank}")
                    weight_memory_registry = self._register_replica_memory(
                        model_replica, self.remote_weight_infos_by_session_id[session_id][0], transfer_engine
                    )
                    self.engines[target.engine_rank] = TransferBundle(
                        model_replica,
                        transfer_engine,
                        weight_memory_registry,
                        [remote_info],
                    )
                else:
                    self.engines[target.engine_rank].add_remote_session(remote_info)

            print_memory("[RDMA] After Local Engine Replicas and engine Creation")

    def _register_replica_memory(self, model_replica, remote_weight_info, transfer_engine) -> dict:
        # Verify the 1-to-1 mapping between local replica and remote weights expected.
        for name, tensor in model_replica.named_parameters():
            if name not in remote_weight_info:
                raise RuntimeError(f"Local replica parameter {name} not found in remote replica.")
            remote_numel, remote_ele_size = remote_weight_info[name][1], remote_weight_info[name][2]
            if tensor.numel() != remote_numel or tensor.element_size() != remote_ele_size:
                raise RuntimeError(
                    f"Local replica parameter {name} numel {tensor.numel()} size {tensor.element_size()} does not match remote numel {remote_numel} size {remote_ele_size}."
                )
            if tensor.device.type != "cuda":
                raise RuntimeError(f"Local replica parameter {name} is not on CUDA device.")
        weight_memory_registry = register_memory_region_v2(model_replica, transfer_engine)

        logger.info(
            f"[RDMA] Registered {len(list(model_replica.named_parameters()))} tensors from replica with transfer engine."
        )
        return weight_memory_registry

    def _create_transfer_engine(self) -> TransferEngine:
        transfer_engine = TransferEngine()
        local_ip = ray._private.services.get_node_ip_address()
        transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")

        logger.info(f"[RDMA] Local replica Transfer Engine initialized at port {transfer_engine.get_rpc_port()}")
        return transfer_engine

    def _create_inference_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args: ServerArgs,
    ):
        load_config = LoadConfig(
            load_format="auto",
            model_loader_extra_config=server_args.model_loader_extra_config,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args_module._global_server_args = server_args
        with ParallelismContext(parallelism_config):
            model = get_model(
                model_config=ModelConfig(model_path),
                load_config=load_config,
                device_config=DeviceConfig(),
            )
        device = next(model.parameters()).device
        logger.info(f" Model {device}, params: {sum(p.numel() for p in model.parameters())} ")
        return model

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        # Update weight version as we were write-only.
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=str(self.weight_version))
                for engine in self.rollout_engines
            ]
        )
        return

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        The RDMA P2P weight update is implemented as a single side write, meaning the trainer writes its weights directly to the rollout engines' memory.
        Now uses an executable queue to make transfer_bundle.execute_each() operations asynchronous,
        allowing overlap between weight loading and RDMA transfers.
        """

        if not self._is_source or not converted_named_tensors:
            return
        # TODO(letian): update cpu -> gpu logic here ?
        # if self._model_on_cpu:
        #     torch_memory_saver.resume(self.tag)
        #     self._model_on_cpu = False

        for transfer_bundle in self.engines.values():
            transfer_ready_params = transfer_bundle.get_transfer_ready_params(converted_named_tensors)
            transfer_bundle.model_replica.load_weights(converted_named_tensors)
            if self.pipelined_transfer:
                # Use executable queue for async transfer operations
                transfer_bundle.execute_each(transfer_ready_params, self.executable_queue)

        converted_named_tensors.clear()

    def __del__(self):
        """Cleanup resources when the instance is destroyed."""
        if hasattr(self, "executable_queue"):
            self.executable_queue.shutdown()

    def finish_transfer_task(self) -> None:
        if not self._is_source:
            return

        # Execute transfer for each engine replica.
        if not self.pipelined_transfer:
            for transfer_bundle in self.engines.values():
                transfer_bundle.execute()
        else:
            # Wait for all queued transfer tasks to complete before cpu offloading
            logging.info("[RDMA] Waiting for all queued transfer tasks to complete...")
            assert self.executable_queue.wait_all_complete(
                timeout=300.0  # TODO: here 30 -> 300?
            ), "[RDMA] Some transfer tasks may not have completed within timeout"

            # Add CUDA synchronization to ensure all asynchronous RDMA operations are complete
            # This is critical to prevent race conditions with memory offloading
            logging.info("[RDMA] Synchronizing CUDA to ensure all asynchronous operations complete...")
            torch.cuda.synchronize()
            for transfer_bundle in self.engines.values():
                transfer_bundle.reset()

        # Offload model replicas from memory after transfer.
        # TODO(letian): update gpu -> cpu logic here ?
        # if not self._model_on_cpu:
        #     print_memory("[RDMA] Before offloading model replica")
        #     torch_memory_saver.pause(self.tag)
        #     self._model_on_cpu = True
        #     print_memory("[RDMA] After offloading model replica")

        return
