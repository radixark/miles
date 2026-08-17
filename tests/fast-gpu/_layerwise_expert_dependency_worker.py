"""Distributed dependency-integration probe for Bridge expert LoRA and MCore LayerWise."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.peft.utils import GroupedExpertLinearAdapter
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import FP32Optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    try:
        config = ModelParallelConfig(
            tensor_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            params_dtype=torch.float32,
        )
        adapter = GroupedExpertLinearAdapter(
            in_features=4,
            out_features=4,
            dim=2,
            num_local_experts=2,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
            activation="identity",
            input_is_parallel=True,
            model_parallel_config=config,
            params_device=torch.device("cuda", local_rank),
            params_dtype=torch.float32,
        )
        params = [adapter.linear_in.weight, adapter.linear_out.weight]
        with torch.no_grad():
            for index, param in enumerate(params, start=1):
                param.fill_(float(index))
        assert all(param.allreduce is False for param in params)
        assert all(param.tensor_model_parallel is True for param in params)

        optimizer_config = OptimizerConfig(
            optimizer="sgd",
            lr=0.1,
            min_lr=0.0,
            weight_decay=0.0,
            sgd_momentum=0.0,
            clip_grad=1.0,
            bf16=False,
            use_distributed_optimizer=False,
            params_dtype=torch.float32,
        )
        base_optimizer = torch.optim.SGD(
            [{"params": params, "is_expert_parallel": True}],
            lr=optimizer_config.lr,
        )
        optimizer = LayerWiseDistributedOptimizer(
            [FP32Optimizer(base_optimizer, optimizer_config, None)],
            optimizer_config,
            ProcessGroupCollection.use_mpu_process_groups(["tp", "expt_tp", "dp_cp", "expt_dp"]),
        )

        assert optimizer.dp_cp_params_list is None
        assert optimizer.expt_dp_params_list is not None
        local_owners = torch.tensor(
            len(optimizer.chained_optimizers[0].get_parameters()),
            device="cuda",
            dtype=torch.int64,
        )
        dist.all_reduce(local_owners)
        assert local_owners.item() == len(params)

        for param in params:
            param.main_grad = torch.full_like(param, 3.0)
        true_norm = (sum(param.numel() * 3.0**2 for param in params)) ** 0.5
        before = [param.detach().clone() for param in params]

        update_successful, grad_norm, _ = optimizer.step()

        assert update_successful
        assert grad_norm == pytest.approx(true_norm, rel=1e-6, abs=1e-6)
        clip_coefficient = 1.0 / (true_norm + 1.0e-6)
        for previous, param in zip(before, params, strict=True):
            torch.testing.assert_close(
                param,
                previous - optimizer_config.lr * 3.0 * clip_coefficient,
                rtol=1e-6,
                atol=1e-6,
            )
            replicas = [torch.empty_like(param) for _ in range(dist.get_world_size())]
            dist.all_gather(replicas, param)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
