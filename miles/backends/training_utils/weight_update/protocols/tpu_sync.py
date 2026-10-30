"""Weight sync over tpu-sync: each trainer rank's FSDP shards go straight to every engine TP rank's slice.

Rank 0 runs the tpu-sync controller; one transfer reshards the registered trainer blocks into the engines' staging
buffers (sglang-torchtpu's ``tpu_sync.init_receiver`` handler), which the engines copy into their parameters
(``tpu_sync.install``).
tpu-sync is TPU-only, hence the local imports; its endpoints are unauthenticated.
"""

import time
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group

# tpu-sync's default number of parallel streams per unit
_PARALLELISM = 8
# arbitrary
_REGISTRATION_TIMEOUT_S = 60.0

# (HF name, local tensor, (count, index) of an FSDP dim-0 block or None for a whole tensor)
_LocalWeight = tuple[str, torch.Tensor, tuple[int, int] | None]


class UpdateWeightTpuSync(WeightTransferProtocol):
    streams_buckets = False

    def __init__(self, args) -> None:
        super().__init__(args)
        self._synchronizer = None
        self._signature: list[tuple] | None = None
        # rank 0 owns the controller; its server must stay referenced
        self._controller_address: str | None = None
        self._controller = None
        self._controller_server = None
        self._receivers_registered = False

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.group_name = "miles-tpu-sync"
        self.is_sender = True
        self._engine_tp_sizes = dict(zip(engine_cell_ids, engine_gpu_counts, strict=True))
        # new or restarted engines register again
        self._receivers_registered = False

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        raise NotImplementedError("tpu-sync moves the whole model in transfer()")

    def transfer(self, hf_weight_iterator) -> None:
        local_weights = list(hf_weight_iterator.iter_local_weights())
        self._ensure_controller()
        with fail_on_every_rank("[tpu-sync] bind weights"):
            # TorchTPU defers execution; the bound buffers must hold their final values
            torch.accelerator.synchronize()
            self._bind(local_weights)
        # the other ranks serve the transfer from listener threads
        with fail_on_every_rank("[tpu-sync] transfer"):
            if dist.get_rank() == 0:
                self._push(_full_shapes(local_weights))
        self._synchronizer.unbind_weights()

    def _ensure_controller(self) -> None:
        if self._controller_address is not None:
            return
        with fail_on_every_rank("[tpu-sync] start controller"):
            if dist.get_rank() == 0:
                module = _controller_module()
                self._controller = module.RaidenController(port=0)
                self._controller_server = module.RaidenControllerServer(self._controller)
                self._controller_address = f"{_local_ip()}:{self._controller_server.start()}"
        address = [self._controller_address]
        dist.broadcast_object_list(address, src=0, group=get_gloo_group())
        self._controller_address = address[0]

    def _bind(self, local_weights: list[_LocalWeight]) -> None:
        signature = [(name, tuple(t.shape), t.dtype, block) for name, t, block in local_weights]
        tensors = [[t] for _, t, _ in local_weights]
        if signature == self._signature:
            self._synchronizer.bind_weights(tensors)
            return
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

    def _push(self, weight_shapes: dict[str, list[int]]) -> None:
        updaters = [u for u in self.cell_updaters_of_cell_id.values() if not u.is_errored]
        if not self._receivers_registered:
            _call_engines(
                updaters,
                "tpu_sync.init_receiver",
                lambda updater: dict(
                    controller_address=self._controller_address,
                    job_name=_engine_job_name(updater.cell_id),
                    weight_shapes=weight_shapes,
                    parallelism=_PARALLELISM,
                ),
            )
            self._receivers_registered = True
            # a cell that failed to register is marked errored under fault tolerance; leave it out of the transfer
            updaters = [u for u in updaters if not u.is_errored]
        src_units = [_unit_id("trainer", rank) for rank in range(dist.get_world_size())]
        # one transfer for every engine: tpu-sync shards within a job and broadcasts across jobs
        dst_units = [
            _unit_id(_engine_job_name(u.cell_id), rank)
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
        # the transfer is a coroutine that runs only while wait() is awaited
        async_utils.run(transfer.wait())
        _call_engines(updaters, "tpu_sync.install", lambda updater: {})


def _call_engines(updaters, handler: str, payload_of) -> None:
    futures = [
        updater.submit_client_call("plugin_weight_update", handler=handler, payload=payload_of(updater))
        for updater in updaters
    ]
    check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)


def _full_shapes(local_weights: list[_LocalWeight]) -> dict[str, list[int]]:
    return {name: _full_shape(t, block) for name, t, block in local_weights}


def _full_shape(tensor: torch.Tensor, block: tuple[int, int] | None) -> list[int]:
    shape = list(tensor.shape)
    if block is not None:
        shape[0] *= block[0]
    return shape


def _trainer_variable(index: int, name: str, tensor: torch.Tensor, block: tuple[int, int] | None):
    """Describe a local tensor to tpu-sync: dim-0 block ``index`` of ``count`` (FSDP shard), or the whole tensor."""
    shape = _full_shape(tensor, block)
    if block is None:
        sharding = {"mesh_shape": [1] * len(shape), "sharding_spec": [""] * len(shape)}
    else:
        count, block_index = block
        sharding = {
            "mesh_shape": [count] + [1] * (len(shape) - 1),
            "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
            "global_shard_indices": [block_index],
        }
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
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        tensors,
        local_port=0,
        parallelism=_PARALLELISM,
        listener_port=0,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        # a trainer only sends
        auto_h2d=False,
    )


def _controller_module():
    # the library still names its controller after its former code name
    from tpu_sync.rpc import raiden_controller

    return raiden_controller


def _unit_id(job_name: str, rank: int):
    from tpu_sync.api.common import RaidenId

    return RaidenId(job_name=job_name, job_replica_id=str(rank), data_name="weights")


def _engine_job_name(cell_id: str) -> str:
    return f"engine-{cell_id}"


def _local_ip() -> str:
    # the hostname can resolve to loopback (127.0.1.1), which a peer on another host cannot reach
    return ray.util.get_node_ip_address()
