from abc import abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, expert_named_params_and_buffers, non_expert_named_params_and_buffers
from .remote_transfer_plan import RemoteTransferPlan


class UpdateWeightFromRemote:
    """
    Abstract base class for remote bucketed tensor weight update. Weights are all-gathered in TP EP dimension
    for each bucket of predefined size and processed into HF format.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        weight_update_mode: Literal["nccl", "rdma"] = "nccl",
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.transfer_plan = RemoteTransferPlan(args, model, weight_update_mode)
        self.weight_update_mode = weight_update_mode
        self._is_source = self.transfer_plan.is_source()
        self.global_rank = dist.get_rank(group=get_gloo_group())

    @abstractmethod
    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """Establish connection to remote rollout engines."""

    @abstractmethod
    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Implementation of the bucketed parameter update from remote."""

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        For each named parameter in the model, do bucketed weight update by all-gather EP/TP, convert and quantize,
        and relies on underlying implementation to do the transfer.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )

        dist.barrier(group=get_gloo_group())

        self.on_transfer_start()

        non_expert_params = non_expert_named_params_and_buffers(self.args, self.model)
        expert_params = expert_named_params_and_buffers(self.args, self.model)

        self._update_weights(non_expert_params)
        if self.weight_update_mode == "nccl":
            dist.barrier(group=get_gloo_group())

        self._update_expert_weights(expert_params)
        if self.weight_update_mode == "nccl":
            dist.barrier(group=get_gloo_group())

        self.finish_transfer_task()

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            # int4/fp4 post_process, mxfp8 post-process (swizzle MoE scales).
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors",
                "mxfp8",
            ]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            self.leader_post_update()
        dist.barrier(group=get_gloo_group())

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    def on_transfer_start(self) -> None:
        """Hook called at start of weight transfer cycle. Override for setup work like starting background registration."""

    def finish_transfer_task(self) -> None:
        """Hook called after all weight buckets are transferred. Override for cleanup."""

    def _update_weights(self, named_params_and_buffers: Iterable[tuple[str, torch.Tensor]]) -> None:
        pbar = tqdm(desc="[Update Weights]", total=0) if self._is_source else None
        buffer_size = 0
        converted_named_tensors = []
        for name, param in named_params_and_buffers:
            assert ".experts." not in name, "Function intended for non-expert params only."
            buffer_size = self._update_weight_from_remote(name, param, converted_named_tensors, buffer_size, pbar=pbar)

        if converted_named_tensors:
            self._update_bucket_weights_from_remote(converted_named_tensors, pbar=pbar)

    def _update_expert_weights(self, named_params_and_buffers: Iterable[tuple[str, torch.Tensor]]) -> None:
        pbar = tqdm(desc="[Update Expert Weights]", total=0) if self._is_source else None
        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers:
            assert ".experts." in name, "Function intended for expert params only."
            buffer_size = self._update_expert_weight_from_remote(name, param, named_tensors, buffer_size, pbar=pbar)

        if named_tensors:
            self._update_expert_bucket_weights_from_remote(named_tensors, pbar=pbar)

    def _update_weight_from_remote(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full).
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(self.args, name, param)
        if not self._is_source:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_remote(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_remote(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(self.args, name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size and named_tensors:
            self._update_expert_bucket_weights_from_remote(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_remote(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Gather EP → HF → delegate to _update_bucket_weights_from_remote. Clears buffer."""
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(
                params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True
            )
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_source:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(
                self.args, self.model_name, name, param, self.quantization_config
            )

        self._update_bucket_weights_from_remote(converted_hf_tensors, pbar)


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """Trigger post-process for int4/fp4 quantization on all rollout engines."""
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
