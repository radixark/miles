"""Weight sync over tpu-sync (Raiden): each trainer rank's FSDP shards transfer directly to rollout engines.

Rank 0 runs the tpu-sync controller; one ``start_transfer`` call reshards the registered trainer blocks into every
rollout engine's staging buffers (registered via ``init_tpu_sync_receiver``), which each engine then installs via
``install_tpu_sync_weights``.

This module is TPU-only for the ``tpu_sync`` library itself, hence the deferred local imports of ``tpu_sync.*`` so
CUDA-only processes never need ``tpu_sync`` installed.
"""

import logging
import time
from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import EngineApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.platforms import current_platform
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)

# tpu-sync's default number of parallel streams per unit
_PARALLELISM = 8
# Wait budget for all trainer and rollout units to register with the Raiden controller
_REGISTRATION_TIMEOUT_S = 60.0

# (HF name, local tensor, (count, index) of an FSDP dim-0 block or None for a whole tensor)
_LocalWeight = tuple[str, torch.Tensor, tuple[int, int] | None]


class UpdateWeightTpuSync(WeightTransferProtocol):
    streams_buckets = False

    def __init__(self, args) -> None:
        super().__init__(args)
        self._synchronizer = None
        self._signature: list[tuple] | None = None
        # Rank 0 owns the controller; its server must stay referenced for the protocol lifetime.
        self._controller_address: str | None = None
        self._controller = None
        self._controller_server = None
        self._registered_receiver_shapes: dict[str, list[int]] | None = None
        self._engine_tp_sizes: dict[str, int] = {}
        self._engine_job_names: dict[str, str] = {}

    def connect(
        self,
        rollout_engines: Sequence[EngineApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        assert engine_gpu_counts is not None and len(engine_gpu_counts) == len(rollout_engines)
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.group_name = "miles-tpu-sync"
        self.is_sender = True
        self._engine_tp_sizes = dict(zip(engine_cell_ids, engine_gpu_counts, strict=True))
        self._engine_job_names = {
            cell_id: _engine_job_name(engine_idx) for engine_idx, cell_id in enumerate(engine_cell_ids)
        }
        # New or restarted engines must register their receivers again.
        self._registered_receiver_shapes = None

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        raise NotImplementedError("tpu-sync moves the whole model in transfer()")

    def transfer(self, hf_weight_iterator) -> None:
        local_weights = list(hf_weight_iterator.iter_local_weights())
        # Collective: a contract violation on one rank must stop every rank before the controller collectives.
        with fail_on_every_rank("[tpu-sync] check local weights"):
            seen_data_ptrs: set[int] = set()
            for name, tensor, block in local_weights:
                _check_bound_tensor(name, tensor, block, seen_data_ptrs=seen_data_ptrs)
        try:
            self._ensure_controller()
            with fail_on_every_rank("[tpu-sync] bind weights"):
                # TorchTPU defers execution; bound buffers must hold their final values before transfer.
                # Never call ws.d2h() before transfer 1 completes (skip_tiling plan arrives on transfer 1).
                if torch.accelerator.is_available():
                    torch.accelerator.synchronize()
                self._bind(local_weights)
            # Non-driver ranks serve the transfer from their synchronizer listener threads.
            try:
                with fail_on_every_rank("[tpu-sync] transfer"):
                    if dist.get_rank() == 0:
                        self._push(_full_shapes(local_weights))
            finally:
                self._unbind()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Shut down the local WeightSynchronizer and stop the rank-0 RaidenControllerServer."""
        self._shutdown_synchronizer()
        self._stop_controller()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_controller(self) -> None:
        if self._controller_address is not None:
            return
        with fail_on_every_rank("[tpu-sync] start controller"):
            if dist.get_rank() == 0:
                module = _controller_module()
                controller = module.RaidenController(port=0)
                server = module.RaidenControllerServer(controller)
                try:
                    port = server.start()
                except Exception:
                    _stop_server(server)
                    raise
                self._controller = controller
                self._controller_server = server
                self._controller_address = f"{_local_ip()}:{port}"
        address = [self._controller_address]
        dist.broadcast_object_list(address, src=0, group=get_gloo_group())
        self._controller_address = address[0]

    def _bind(self, local_weights: list[_LocalWeight]) -> None:
        signature = [(name, tuple(t.shape), t.dtype, block) for name, t, block in local_weights]
        tensors = [[t] for _, t, _ in local_weights]
        if self._synchronizer is not None and signature == self._signature:
            bind_weights = getattr(
                self._synchronizer, "bind_weights", None
            )  # config-access-exempt: older tpu_sync builds may omit bind_weights
            if callable(bind_weights):
                bind_weights(tensors)
                return
            self._shutdown_synchronizer()
        elif self._synchronizer is not None:
            self._shutdown_synchronizer()

        bind_ip = _local_ip()
        self._synchronizer = _new_weight_synchronizer(tensors, bind_ip=bind_ip)
        self._signature = signature
        _controller_module().RaidenControllerClientFacade(self._controller_address).register_work_unit(
            _unit_id("trainer", dist.get_rank()),
            [f"{bind_ip}:{self._synchronizer.local_port}"],
            f"{bind_ip}:{self._synchronizer.listener_port}",
            mesh_shape=[max([1] + [block[0] for _, _, block in local_weights if block is not None]), 1],
            variables=[_trainer_variable(i, name, t, block) for i, (name, t, block) in enumerate(local_weights)],
            mesh_axes=["fsdp", "tp"],
        )

    def _unbind(self) -> None:
        if self._synchronizer is None:
            return
        unbind_weights = getattr(
            self._synchronizer, "unbind_weights", None
        )  # config-access-exempt: older tpu_sync builds may omit unbind_weights
        if callable(unbind_weights):
            unbind_weights()
            return
        self._shutdown_synchronizer()

    def _shutdown_synchronizer(self) -> None:
        synchronizer, self._synchronizer, self._signature = self._synchronizer, None, None
        if synchronizer is None:
            return
        shutdown = getattr(
            synchronizer, "shutdown", None
        )  # config-access-exempt: external WeightSynchronizer cleanup hook
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.warning("[tpu-sync] WeightSynchronizer.shutdown() failed", exc_info=True)

    def _stop_controller(self) -> None:
        server = self._controller_server
        self._controller_server = None
        self._controller = None
        self._controller_address = None
        self._registered_receiver_shapes = None
        if server is not None:
            _stop_server(server)

    def _push(self, weight_shapes: dict[str, list[int]]) -> None:
        updaters = [u for u in self.cell_updaters_of_cell_id.values() if not u.is_errored]
        if not updaters:
            return
        if self._registered_receiver_shapes != weight_shapes:
            _init_engine_receivers(
                updaters,
                controller_address=self._controller_address,
                engine_job_names=self._engine_job_names,
                weight_shapes=weight_shapes,
                parallelism=_PARALLELISM,
            )
            self._registered_receiver_shapes = weight_shapes
            # A cell that failed to register is marked errored under fault tolerance; exclude it.
            updaters = [u for u in updaters if not u.is_errored]
            if not updaters:
                return
        src_units = [_unit_id("trainer", rank) for rank in range(dist.get_world_size())]
        # One transfer for every engine: tpu-sync reshards within each job and broadcasts across jobs.
        dst_units = [
            _unit_id(self._engine_job_names[u.cell_id], rank)
            for u in updaters
            for rank in range(self._engine_tp_sizes[u.cell_id])
        ]
        _wait_for_registration(self._controller, src_units + dst_units)
        transfer = self._controller.start_transfer(
            src_units=src_units,
            dst_units=dst_units,
            dst_mem_type=_controller_module().RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
            expected_block_count=0,
            parallelism=_PARALLELISM,
            req_id=f"miles-{time.time_ns()}",
        )
        # RaidenFuture.wait() is an async coroutine that only drives the transfer while awaited.
        async_utils.run(transfer.wait())
        _install_engine_weights(updaters)


def _init_engine_receivers(
    updaters,
    *,
    controller_address: str,
    engine_job_names: dict[str, str],
    weight_shapes: dict[str, list[int]],
    parallelism: int,
) -> None:
    futures = [
        updater.submit_client_call(
            "init_tpu_sync_receiver",
            controller_address=controller_address,
            job_name=engine_job_names[updater.cell_id],
            weight_shapes=weight_shapes,
            parallelism=parallelism,
        )
        for updater in updaters
    ]
    check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)


def _install_engine_weights(updaters) -> None:
    futures = [updater.submit_client_call("install_tpu_sync_weights") for updater in updaters]
    check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)


def _check_bound_tensor(
    name: str, tensor: torch.Tensor, block: tuple[int, int] | None, *, seen_data_ptrs: set[int]
) -> None:
    """Raise unless ``tensor`` meets the iterator contract (#18) tpu-sync binds against.

    The weight iterator owns producing such tensors; a violation here is an iterator bug, so it fails loudly
    instead of being patched over with a silent copy.
    """
    if tensor.ndim < 2 and block is not None:
        raise ValueError(f"[tpu-sync] 1D/0D tensor {name!r} must be registered whole (got block={block})")
    if torch.accelerator.is_available() and tensor.device.type != current_platform.device_type:
        raise ValueError(f"[tpu-sync] {name!r} is on {tensor.device}, not the {current_platform.device_type} device")
    if not tensor.is_contiguous():
        raise ValueError(f"[tpu-sync] {name!r} is not contiguous")
    if tensor._base is not None or tensor.storage_offset() != 0:
        raise ValueError(f"[tpu-sync] {name!r} is a view; a bound buffer must own its storage")
    if tensor.data_ptr() in seen_data_ptrs:
        raise ValueError(f"[tpu-sync] {name!r} aliases the storage of another bound tensor")
    seen_data_ptrs.add(tensor.data_ptr())


def _full_shapes(local_weights: list[_LocalWeight]) -> dict[str, list[int]]:
    return {name: _full_shape(t, block) for name, t, block in local_weights}


def _full_shape(tensor: torch.Tensor, block: tuple[int, int] | None) -> list[int]:
    shape = list(tensor.shape)
    if block is not None:
        shape[0] *= block[0]
    return shape


def _trainer_variable(index: int, name: str, tensor: torch.Tensor, block: tuple[int, int] | None):
    """Describe a local tensor to tpu-sync: dim-0 block ``index`` of ``count`` (FSDP shard), or the whole tensor."""
    if tensor.ndim < 2 and block is not None:
        raise ValueError(f"[tpu-sync] 1D/0D tensor {name!r} must be registered whole (got block={block})")
    shape = _full_shape(tensor, block)
    if block is None:
        sharding = {
            "mesh_shape": [1] * len(shape),
            "sharding_spec": [""] * len(shape),
            "global_shard_indices": [0],
        }
    else:
        count, block_index = block
        sharding = {
            "mesh_shape": [count] + [1] * (len(shape) - 1),
            "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
            "global_shard_indices": [block_index],
        }
    # Deferred: tpu_sync is a TPU-only dependency absent on CUDA hosts.
    from tpu_sync.rpc import raiden_service_pb2

    return raiden_service_pb2.VariableMetadataProto(
        name=name,
        shape=shape,
        layout=list(range(len(shape) - 1, -1, -1)),
        item_size=tensor.element_size(),
        layer_idx=index,
        **sharding,
    )


def _wait_for_registration(controller, units: list) -> None:
    deadline = time.perf_counter() + _REGISTRATION_TIMEOUT_S
    while True:
        registered = controller.get_registered_units()
        missing = [unit for unit in units if unit not in registered]
        if not missing:
            return
        if time.perf_counter() > deadline:
            raise TimeoutError(f"[tpu-sync] units never registered with the controller: {missing}")
        time.sleep(0.1)


def _new_weight_synchronizer(tensors: list[list[torch.Tensor]], *, bind_ip: str):
    # Deferred: tpu_sync is a TPU-only dependency absent on CUDA hosts.
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        tensors,
        local_port=0,
        parallelism=_PARALLELISM,
        listener_port=0,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        # A trainer only sends.
        auto_h2d=False,
    )


def _controller_module():
    # Deferred: tpu_sync is a TPU-only dependency absent on CUDA hosts.
    from tpu_sync.rpc import raiden_controller

    return raiden_controller


def _unit_id(job_name: str, rank: int):
    return _controller_module().RaidenId(job_name=job_name, job_replica_id=str(rank), data_name="weights")


def _engine_job_name(engine_idx: int) -> str:
    return f"sampler{engine_idx}"


def _local_ip() -> str:
    # Hostnames can resolve to loopback (127.0.1.1), which a peer on another host cannot reach.
    return get_current_node_ip()


def _stop_server(server) -> None:
    stop = getattr(server, "stop", None)  # config-access-exempt: external RaidenControllerServer cleanup hook
    if callable(stop):
        try:
            stop()
        except Exception:
            logger.warning("[tpu-sync] RaidenControllerServer.stop() failed", exc_info=True)
