import functools
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfNameRoute, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P
from miles.backends.training_utils.weight_update.protocols.transports.local_pull import (
    LocalPullTransport,
    get_device_uuid,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group


class UpdateWeightFromColocatedP2P(UpdateWeightP2P):
    """p2p for colocated rollout engines: each trainer rank prepares the param bytes of the engine rank on its own
    GPU, which pulls them out of this rank's transfer buffers (sglang `update_weights_from_colocated_param_bytes`).

    The transfer buffers live for one sync, so rollout gets their memory back once the engines resume.
    """

    # one trainer rank serves every layer of the engine rank on its GPU
    required_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._socket_paths_by_device_uuid: dict[str, str] = {}
        self._pull_ops: list[Future] = []

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
        """Pair each trainer rank with the engine rank on its GPU and collect every sender's socket for the
        driver's pull ops."""
        super().connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_cell_ids,
            parallel_state,
            placement,
            selector,
        )
        # rebuilt at the next sync from the new pairing's mappings
        self.hf_name_route = None
        socket_paths = [None] * dist.get_world_size()
        dist.all_gather_object(
            socket_paths,
            (get_device_uuid(), self._local_pull_transport.socket_path if self.is_sender else None),
            group=get_gloo_group(),
        )
        self._socket_paths_by_device_uuid = {
            device_uuid: socket_path for device_uuid, socket_path in socket_paths if socket_path is not None
        }

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        """After p2p's setup, route each rank only the expert tensors its engine rank loads; every rank joins."""
        super().begin_sync(weight_version, iter_buckets)
        if self.hf_name_route is None:
            self.hf_name_route = self._gather_hf_name_route()
        return True

    def after_session_open(self) -> None:
        """The driver starts one pull op per engine, which runs for the whole stream; each sender answers its
        engine rank's Hello."""
        if dist.get_rank() == 0:
            self._pull_ops = [
                cell_updater.submit_client_call(
                    "update_weights_from_colocated_param_bytes",
                    socket_paths_by_device_uuid=self._socket_paths_by_device_uuid,
                    timeout_s=self.args.p2p_transfer_timeout,
                )
                for cell_updater in self.cell_updaters_of_cell_id.values()
            ]
        if self.is_sender:
            self._local_pull_transport.begin_session(
                expected_names_by_role={
                    runner_sender.runner_role: frozenset(
                        runner_sender.model_replicas.hf_name_mapping.hf_names_by_param_name
                    )
                    for runner_sender in self._runner_senders
                },
                stop_waiting=lambda: any(pull_op.done() for pull_op in self._pull_ops),
            )

    def after_base_weights(self) -> None:
        """Wait for every write, tell the engine rank the stream ended, and on the driver surface every engine's
        result."""
        super().after_base_weights()
        if self.is_sender:
            self._local_pull_transport.end_session()
        if dist.get_rank() == 0:
            check_weight_sync_results(async_utils.wait_futures(self._pull_ops), is_lora=False)
            self._pull_ops = []

    def after_engines_resumed(self) -> None:
        if self._transfer_buffers is not None:
            self._transfer_buffers.release()
            self._transfer_buffers = None
        torch.cuda.empty_cache()

    def _gather_hf_name_route(self) -> HfNameRoute:
        loaded_hf_names = frozenset(
            hf_name
            for runner_sender in (self._runner_senders if self.is_sender else [])
            for hf_name in runner_sender.model_replicas.hf_name_mapping.param_names_by_hf_name
        )
        hf_names_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(hf_names_by_rank, loaded_hf_names, group=get_gloo_group())
        return HfNameRoute(hf_names_by_rank=tuple(hf_names_by_rank))

    def _assign(
        self,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        engine_gpu_counts: Sequence[int],
        engine_gpu_offsets: Sequence[int],
    ) -> list[RolloutEngineRankAssignment]:
        trainer_world_size = dist.get_world_size()
        assert all(
            gpu_offset + gpu_count <= trainer_world_size
            for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True)
        ), (
            f"rollout engines on GPUs {list(zip(engine_gpu_offsets, engine_gpu_counts, strict=True))} reach beyond the "
            f"{trainer_world_size} trainer ranks; a colocated pull needs a trainer rank on every engine GPU"
        )
        return assign_colocated_rollout_engine_ranks(
            trainer_rank=dist.get_rank(), engine_gpu_counts=engine_gpu_counts, engine_gpu_offsets=engine_gpu_offsets
        )

    def _create_native_transport(self) -> LocalPullTransport:
        return self._local_pull_transport

    @functools.cached_property
    def _local_pull_transport(self) -> LocalPullTransport:
        # created on first use, by a sender only: it binds a socket and reads this rank's GPU UUID
        return LocalPullTransport(timeout_s=self.args.p2p_transfer_timeout)


def assign_colocated_rollout_engine_ranks(
    *, trainer_rank: int, engine_gpu_counts: Sequence[int], engine_gpu_offsets: Sequence[int]
) -> list[RolloutEngineRankAssignment]:
    """The engine rank on this trainer rank's GPU, by the GPU ranges the engines were placed on; a trainer rank in
    a placeholder slot sends nothing. The engine's Hello checks the pairing by GPU UUID."""
    for rollout_engine_ind, (gpu_offset, gpu_count) in enumerate(
        zip(engine_gpu_offsets, engine_gpu_counts, strict=True)
    ):
        if gpu_offset <= trainer_rank < gpu_offset + gpu_count:
            return [
                RolloutEngineRankAssignment(
                    rollout_engine_rank=trainer_rank - gpu_offset, rollout_engine_indices=(rollout_engine_ind,)
                )
            ]
    return []
