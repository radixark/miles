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
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Stages `converted_named_tensors` and returns the HF tensors of each sglang param that became complete, by
        sglang param name.

        sglang fuses several HF tensors into one param (q/k/v into `qkv_proj`, every expert's gate and up into
        `w13_weight`), and they can arrive in different buckets; a load of a param missing some of them would leave
        part of its bytes unwritten.
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

        ready_hf_tensors_by_param_name: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for param_name in dict.fromkeys(transfer_ready_params):
            ready_hf_tensors_by_param_name[param_name] = self._staged_tensors.pop(param_name, [])
            self._tensor_update_pending.pop(param_name, None)

        return ready_hf_tensors_by_param_name

    def assert_all_done(self) -> None:
        assert len(self._tensor_update_pending) == 0 and len(self._staged_tensors) == 0, (
            f"Some tensors were not transferred during P2P weight update. "
            f"Pending: {self._tensor_update_pending}, Staged: {self._staged_tensors}"
        )
