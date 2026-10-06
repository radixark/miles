import concurrent.futures
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from typing import NamedTuple

import torch
import torch.distributed as dist
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import (
    MooncakeTransport,
    RemoteShard,
    RemoteWeightLocation,
)
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks,
)
from miles.utils.distributed_utils import get_gloo_group

from .p2p_transfer_utils import query_remote_weight_infos


class _ReplicaTarget(NamedTuple):
    model_replica: torch.nn.Module
    remote_shards: list[RemoteShard]
    parallelism_config: RankParallelismConfig


class UpdateWeightP2P(WeightTransferProtocol):
    """P2P weight transfer over the updater's bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all engine ranks)
    For each engine rank:
        wait for the previous rank's writes, load_weights(shared buffer) → P2P write
    after_base_weights waits for every write and fails the update if any failed
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_registered = False
        self._model_param_stager = ModelParamStager()
        self._pending_writes: list[tuple[RemoteShard, Future]] = []

    def after_base_weights(self) -> None:
        """Wait for every write of this update; fail the update if any write failed or is still running."""
        if not self.is_sender:
            return
        pending_writes, self._pending_writes = self._pending_writes, []
        _raise_if_any_write_failed(pending_writes, timeout=self.args.p2p_transfer_timeout)
        self._model_param_stager.assert_all_done()

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        """Register shared CPU pinned memory with P2P on the first sync."""
        if self.is_sender and not self._model_registered:
            for tensor in self._shared_params_dict.values():
                self._transport.register_memory(tensor)
            self._model_registered = True
        return True

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Stage incoming tensors; when all shards for a param are collected,
        load into shared buffer and P2P-write per engine rank.

        Only calls load_weights() with complete accumulated tensors, preventing
        partial writes that would corrupt the shared buffer when different engine
        ranks have different EP expert-to-local mappings.
        """
        if not self.is_sender or not converted_named_tensors:
            return
        # `ready_hf_tensors`` here are the complete tensors ready to be transferred.
        transfer_ready_params, ready_hf_tensors = self._model_param_stager.get_transfer_ready_params(
            converted_named_tensors,
            param_mapper=self._shared_param_mapper,
            params_dict=self._shared_params_dict,
        )

        if transfer_ready_params and ready_hf_tensors:
            tensors_by_name = {name: self._shared_params_dict[name] for name in transfer_ready_params}
            previous_rank_writes: list[Future] = []
            for target in self._replica_targets:
                # loading overwrites the shared buffer the previous rank's writes read from
                concurrent.futures.wait(previous_rank_writes)
                with ParallelismContext(target.parallelism_config):
                    target.model_replica.load_weights(ready_hf_tensors)

                previous_rank_writes = self._transport.write(target.remote_shards, tensors_by_name)
                self._pending_writes += zip(target.remote_shards, previous_rank_writes, strict=True)

        converted_named_tensors.clear()

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """``connect`` here will:

        - Assign this rank its target engine ranks from the engines handed over
          (``engine_gpu_counts``) and the iterator's resolved placement.
        - Query remote rollout engines for their weight memory registration
          info (addresses and sizes for RDMA writes).
        - Query remote parallelism config and construct a local CPU model
          replica that mirrors the target's sharding layout, enabling correct
          weight format conversion before transfer.
        """
        self.rollout_engines = rollout_engines
        assignments = assign_rollout_engine_ranks(parallel_state, placement, engine_gpu_counts)
        self.is_sender = bool(assignments)

        if self.is_sender:
            (
                self.remote_weight_infos_by_session_id,
                targets_to_session_id,
                self.session_id_to_server_args,
            ) = query_remote_weight_infos(rollout_engines, assignments)

            self._transport = MooncakeTransport()
            self._shared_params_dict: dict[str, torch.Tensor] = {}
            self._shared_param_mapper: ParameterMapper | None = None
            self._replica_targets: list[_ReplicaTarget] = []
            first_rollout_engine_rank = True
            for assignment in assignments:
                session_id = targets_to_session_id[
                    (assignment.rollout_engine_indices[0], assignment.rollout_engine_rank)
                ]
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                server_args = self.session_id_to_server_args[session_id]

                model_replica = self._create_cpu_replica(
                    parallelism_config,
                    self.args.hf_checkpoint,
                    server_args,
                    first_rollout_engine_rank=first_rollout_engine_rank,
                )
                if first_rollout_engine_rank:
                    self._shared_params_dict = dict(model_replica.named_parameters())
                    self._shared_param_mapper = ParameterMapper.from_model(model_replica)
                    first_rollout_engine_rank = False

                remote_shards = []
                for rollout_engine_ind in assignment.rollout_engine_indices:
                    shard_session_id = targets_to_session_id[(rollout_engine_ind, assignment.rollout_engine_rank)]
                    weights_info = self.remote_weight_infos_by_session_id[shard_session_id][0]
                    remote_shards.append(
                        RemoteShard(
                            rollout_engine_ind=rollout_engine_ind,
                            rollout_engine_rank=assignment.rollout_engine_rank,
                            session_id=shard_session_id,
                            weight_locations_by_name={
                                name: RemoteWeightLocation(*location) for name, location in weights_info.items()
                            },
                        )
                    )

                self._replica_targets.append(_ReplicaTarget(model_replica, remote_shards, parallelism_config))

    def _create_cpu_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args: ServerArgs,
        first_rollout_engine_rank: bool = False,
    ) -> torch.nn.Module:
        """Create a CPU model replica that loads the right shard and skips post_load_weights."""
        load_config = LoadConfig(
            load_format="dummy",
            model_loader_extra_config=None,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args_module.set_global_server_args_for_scheduler(server_args)
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()

        # Monkey-patch the loader-level post_load_weights to no-op BEFORE get_model,
        # because get_model() calls post_load_weights() internally (loader.py:1310)
        # which may invoke CUDA-only kernels (e.g., per_tensor_quant_fp8 for FP8 models).
        # This is safe because the rollout engine runs post_load_weights on its own GPU
        # after RDMA transfer, at end_weight_update.
        from sglang.srt.model_loader import loader as model_loader_module

        original_post_load_weights = model_loader_module.post_load_weights
        model_loader_module.post_load_weights = lambda *args, **kwargs: None
        try:
            with ParallelismContext(parallelism_config):
                model = get_model(
                    model_config=ModelConfig(model_path),
                    load_config=load_config,
                    device_config=DeviceConfig(device="cpu"),
                )
        finally:
            model_loader_module.post_load_weights = original_post_load_weights

        # Also patch the instance method for subsequent load_weights() calls
        # (deepseek_weight_loader.py:342 calls self.post_load_weights() at the end).
        if hasattr(model, "post_load_weights"):
            model.post_load_weights = lambda *args, **kwargs: None

        if first_rollout_engine_rank:
            for param in model.parameters():
                param.data = param.data.pin_memory()
        else:
            for name, param in model.named_parameters():
                assert name in self._shared_params_dict, f"[P2P-Shared] Parameter {name} not found in shared buffers"
                param.data = self._shared_params_dict[name]

        return model


def _raise_if_any_write_failed(pending_writes: list[tuple[RemoteShard, Future]], timeout: float) -> None:
    _, unfinished_writes = concurrent.futures.wait([write for _, write in pending_writes], timeout=timeout)
    failures = []
    for remote_shard, write in pending_writes:
        if write in unfinished_writes:
            reason = f"still running after {timeout}s"
        elif write.exception() is not None:
            reason = repr(write.exception())
        else:
            continue
        failures.append(
            f"rollout engine {remote_shard.rollout_engine_ind} rank {remote_shard.rollout_engine_rank} "
            f"(session {remote_shard.session_id}): {reason}"
        )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(pending_writes)} p2p writes failed:\n" + "\n".join(failures))
