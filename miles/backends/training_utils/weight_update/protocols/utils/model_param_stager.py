from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_loader.parameter_mapper import ParameterMapper

logger = logging.getLogger(__name__)


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
