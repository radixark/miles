"""K3 FSDP, tensor/sequence parallelism, and expert parallelism."""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder

from miles.backends.torchtitan_utils.models.kimi_k3.attention import DeltaAttention
from miles.backends.torchtitan_utils.models.kimi_k3.sharding import shard_kimi_k3


def parallelize_kimi_k3(model, *, parallel_dims, training, parallelism, compile_config, ac_config, dump_folder):
    if parallel_dims.pp_enabled:
        raise ValueError("Kimi K3 currently requires PP=1")
    if parallel_dims.cp_enabled:
        if parallelism.context_parallel_load_balancer != "headtail":
            raise ValueError("Kimi K3 context parallelism requires headtail load balancing")
        model.cp_mesh = parallel_dims.get_mesh("cp")
    if compile_config.enable:
        raise ValueError("Kimi K3 compilation has not been validated")
    replicate_dims = ["dp_replicate"] if parallel_dims.dp_replicate_enabled else []
    mesh = parallel_dims.get_mesh(replicate_dims + ["fsdp"])
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        shard_kimi_k3(model, parallel_dims)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
    for layer in model.layers.values():
        attention = layer.self_attn
        if isinstance(attention, DeltaAttention):
            for module in (
                attention.core,
                attention.q_conv1d,
                attention.k_conv1d,
                attention.v_conv1d,
                attention.o_norm,
            ):
                fully_shard(module, mesh=mesh, mp_policy=fp32)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    apply_fsdp_to_decoder(
        model,
        mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=parallel_dims.get_optional_mesh(replicate_dims + ["efsdp"]),
    )
    return model
