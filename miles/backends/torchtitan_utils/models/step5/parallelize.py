import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.config import TORCH_DTYPE_MAP, CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, get_fsdp_reshard_after_forward_policy


def parallelize_step5(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    if parallelism.spmd_backend != "partial_dtensor":
        raise NotImplementedError(f"Step5 supports spmd_backend partial_dtensor, got {parallelism.spmd_backend}")
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Step5 does not support torch.compile of the model yet")

    if parallel_dims.cp_enabled:
        model.enable_context_parallel(
            parallel_dims.get_mesh("cp"), load_balancer=parallelism.context_parallel_load_balancer
        )
    if parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    _shard_norms_in_fp32(
        model,
        dp_mesh=dp_mesh,
        reshard_after_forward=get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward, pp_enabled=parallel_dims.pp_enabled
        ),
    )
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )
    return model


def _shard_norms_in_fp32(model, *, dp_mesh, reshard_after_forward: bool) -> None:
    # Megatron and SGLang keep every norm weight in fp32 and add it to 1; a bf16 copy would round (1 + w)
    fp32_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
    )
    groups = [
        [block.attention_norm, block.ffn_norm, block.attention.q_norm, block.attention.k_norm]
        for block in model.layers.values()
    ]
    if model.norm is not None:
        groups.append([model.norm.rms])
    for norms in groups:
        fully_shard(norms, mesh=dp_mesh, mp_policy=fp32_policy, reshard_after_forward=reshard_after_forward)
