from collections.abc import Callable

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm


from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, expert_named_params_and_buffers, non_expert_named_params_and_buffers


class BucketedWeightGatherMixin:
    """Mixin providing bucketed TP/EP all-gather and HF format conversion."""

    def _gather_and_convert_non_expert_weights(
        self,
        bucket_weight_transfer: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Bucketed TP all-gather + HF conversion for non-expert parameters."""
        pbar = tqdm(desc="[Update Weights]", total=0) if self._is_source else None
        buffer_size = 0
        converted_named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in non_expert_named_params_and_buffers(self.args, self.model):
            param = all_gather_param(self.args, name, param)
            if not self._is_source:
                continue

            param_size = param.numel() * param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                bucket_weight_transfer(converted_named_tensors)
                converted_named_tensors = []
                buffer_size = 0
                if pbar:
                    pbar.update(1)

            converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            buffer_size += param_size

        if converted_named_tensors:
            bucket_weight_transfer(converted_named_tensors)
            if pbar:
                pbar.update(1)

    def _gather_and_convert_expert_weights(
        self,
        bucket_weight_transfer: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Bucketed TP + EP all-gather + HF conversion for expert parameters."""
        pbar = tqdm(desc="[Update Expert Weights]", total=0) if self._is_source else None
        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in expert_named_params_and_buffers(self.args, self.model):
            param = all_gather_param(self.args, name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size and named_tensors:
                self._handle_expert_bucket(named_tensors, bucket_weight_transfer, pbar)
                named_tensors = []
                buffer_size = 0

            named_tensors.append((name, param))
            buffer_size += param_size

        if named_tensors:
            self._handle_expert_bucket(named_tensors, bucket_weight_transfer, pbar)

    def _handle_expert_bucket(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        bucket_weight_transfer: Callable[[list[tuple[str, torch.Tensor]]], None],
        pbar: tqdm | None,
    ) -> None:

        names = [name for name, _ in named_tensors]
        all_names: list[list[str] | None] = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for ep_names in all_names:
            assert len(named_tensors) == len(
                ep_names
            ), f"mismatch names length: {len(named_tensors)} != {len(ep_names)}"

        all_gathered_params: list[list[tuple[str, torch.Tensor]]] = [
            [] for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, ep_names in enumerate(all_names):
                all_gathered_params[ep_rank].append((ep_names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_source:
            return

        flat_gathered = sum(all_gathered_params, [])

        converted_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for name, param in flat_gathered:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        bucket_weight_transfer(converted_hf_tensors)
        if pbar:
            pbar.update(1)
