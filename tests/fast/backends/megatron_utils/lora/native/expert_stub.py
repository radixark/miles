"""Grouped expert host contracts for native adapter attachment tests."""

import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    def __init__(self, num_experts, out_features, in_features, *, partition_dim=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params_dtype = torch.float32
        for index in range(num_experts):
            parameter = nn.Parameter(torch.randn(out_features, in_features))
            parameter.tensor_model_parallel = True
            parameter.partition_dim = partition_dim
            parameter.partition_stride = 1
            parameter.allreduce = False
            self.register_parameter(f"weight{index}", parameter)
