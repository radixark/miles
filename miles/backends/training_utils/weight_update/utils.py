from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

if TYPE_CHECKING:
    from sglang.srt.model_loader.parameter_mapper import ParameterMapper

logger = logging.getLogger(__name__)


def get_data_replica_rank_and_size(parallel_state: ParallelState, placement: WeightUpdatePlacement) -> tuple[int, int]:
    """(replica_rank, replica_size): this rank's index among the ranks that hold
    identical data after gathering per ``placement``, and their count. Collective."""
    if placement.gather_pp:
        return dist.get_rank(), dist.get_world_size()

    column_id = min(dist.get_process_group_ranks(parallel_state.pp.group))
    all_column_ids: list = [None] * dist.get_world_size()
    dist.all_gather_object(all_column_ids, column_id)
    return sorted(set(all_column_ids)).index(column_id), dist.get_world_size() // parallel_state.pp.size


def record_lora_checksums(bucket, checksums) -> None:
    """Accumulate the sha256 manifest the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        digest = hashlib.sha256(
            tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        checksums[lora_name][hf_key] = digest


class ModelParamStager:
    def __init__(self) -> None:
        self._tensor_update_pending: dict[str, int] = {}
        self._staged_tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}

    def get_transfer_ready_params(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        param_mapper: ParameterMapper,
        params_dict: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[tuple[str, torch.Tensor]]]:
        """Determine which sglang params have all shards present, returning their accumulated tensors.

        Some parameters are trained separately on the training side but fused into a
        single tensor on the rollout side (e.g., Q/K/V projections are separate in
        Megatron but merged into one qkv_proj in sglang). This function stages
        incoming HF tensors in self._staged_tensors until all shards for a
        sglang param are collected. Only returns tensors for fully-ready params,
        preventing partial load_weights() calls that would corrupt the shared buffer.

        Return:
            transfer_ready_params: tensors' names for the ones ready to be transferred.
            ready_hf_tensor: corresponding complete tensors ready to be transferred.
        """
        transfer_ready_params = []

        for name, tensor in converted_named_tensors:
            # map the tensor name of huggingface to the one of sglang.
            mapped_result = param_mapper.map(name)
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

            self._staged_tensors.setdefault(mapped, []).append((name, tensor))

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                if mapped not in self._tensor_update_pending:
                    self._tensor_update_pending[mapped] = total_expected - 1
                else:
                    self._tensor_update_pending[mapped] -= 1
                if self._tensor_update_pending[mapped] == 0:
                    transfer_ready_params.append(mapped)

        ready_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for param_name in transfer_ready_params:
            staged = self._staged_tensors.pop(param_name, [])
            ready_hf_tensors.extend(staged)
            self._tensor_update_pending.pop(param_name, None)

        return transfer_ready_params, ready_hf_tensors

    def assert_all_done(self) -> None:
        assert len(self._tensor_update_pending) == 0 and len(self._staged_tensors) == 0, (
            f"Some tensors were not transferred during P2P weight update. "
            f"Pending: {self._tensor_update_pending}, Staged: {self._staged_tensors}"
        )
