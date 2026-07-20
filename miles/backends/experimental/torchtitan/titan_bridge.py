"""Connective utils between miles RL and torchtitan (pinned @7e3f2eb, pip-installed).

torchtitan is used as a library: its models (native kernels: flex attention, grouped-MoE,
fused RMSNorm), its parallelize composition (FSDP2+TP via ParallelDims), its streaming
HF-safetensors checkpoint load, its optimizer container. miles keeps the RL orchestration
(rollout, advantages, sglang weight-sync protocol). This module is the seam:

    spec_from_hf()        HF config.json -> programmatic titan ModelSpec (flavors are
                          hardcoded upstream; this mapper replaces them)
    build_model()         PolicyTrainer._build_model sequence: update_from_config ->
                          meta-build -> parallelize_fn -> to_empty -> init_weights
    load_hf_checkpoint()  dcp streaming load of HF safetensors into DTensor shards
    hf_state_dict()       titan-FQN -> HF-FQN live view for per-step sglang weight sync
"""

import json
import os

import torch

from .compat import apply_torchtitan_compat

apply_torchtitan_compat()


def spec_from_hf(hf_checkpoint: str, *, seq_len: int, attn_backend: str = "flex"):
    """Build a titan ModelSpec programmatically from an HF checkpoint's config.json."""
    with open(os.path.join(hf_checkpoint, "config.json")) as f:
        hf = json.load(f)
    model_type = hf.get("model_type")
    if model_type != "qwen3":
        raise NotImplementedError(f"model_type={model_type!r}; dense qwen3 first (M1)")

    import torchtitan.models.qwen3 as q3
    from torchtitan.protocols.model_spec import ModelSpec

    eps = float(hf.get("rms_norm_eps", 1e-6))
    if abs(eps - q3._EPS) > 1e-12:
        raise NotImplementedError(f"rms_norm_eps={eps} != titan qwen3 _EPS={q3._EPS}")

    dim = hf["hidden_size"]
    n_heads = hf["num_attention_heads"]
    head_dim = int(hf.get("head_dim") or dim // n_heads)
    vocab_size = hf["vocab_size"]
    # RoPE cache must cover both the checkpoint's native context and our run length.
    rope_max = max(int(hf.get("max_position_embeddings", seq_len)), seq_len)

    layers = q3._build_qwen3_layers(
        # fuse_qkv=False keeps state_dict() free of the per-call wqkv allgather hook,
        # which matters because weight sync iterates state_dict() every step.
        fuse_qkv=False,
        n_layers=hf["num_hidden_layers"],
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=hf["num_key_value_heads"],
        head_dim=head_dim,
        hidden_dim=hf["intermediate_size"],
        attn_backend=attn_backend,
        rope=q3.CosSinRoPE.Config(
            dim=head_dim,
            max_seq_len=rope_max,
            theta=float(hf.get("rope_theta", 1000000.0)),
        ),
    )
    config = q3.Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=q3._qwen3_norm(dim),
        tok_embeddings=q3.Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=q3._EMBEDDING_INIT
        ),
        lm_head=q3.Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=q3._output_linear_init(dim)
        ),
        layers=layers,
        enable_weight_tying=bool(hf.get("tie_word_embeddings", False)),
    )
    return ModelSpec(
        name="qwen3",
        flavor=f"hf:{os.path.basename(os.path.normpath(hf_checkpoint))}",
        model=config,
        parallelize_fn=q3.parallelize_qwen3,
        pipelining_fn=None,
        post_optimizer_build_fn=q3.register_moe_load_balancing_hook,
        state_dict_adapter=q3.Qwen3StateDictAdapter,
    )


class _RuntimeConfig:
    """Minimal config-like object for Decoder.update_from_config (needs .parallelism only)."""

    def __init__(self, parallelism):
        self.parallelism = parallelism


def build_model(
    spec,
    *,
    parallel_dims,
    dtype: str = "float32",
    mixed_precision_param: str = "bfloat16",
    tp: int = 1,
    enable_cpu_offload: bool = False,
    ac_mode: str = "none",
    device: str = "cuda",
    dump_folder: str = "/tmp/titan_dump",
):
    """Titan build sequence (mirrors torchtitan's own RL PolicyTrainer._build_model).

    fp32 master weights + bf16 FSDP2 MixedPrecisionPolicy: the forward runs in the
    same bf16 the rollout engine uses (torchtitan's bitwise-parity recipe), while
    optimizer updates accumulate in fp32.
    """
    from torchtitan.config import TORCH_DTYPE_MAP, CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
    from torchtitan.tools.utils import set_default_dtype

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=parallel_dims.dp_shard,
        tensor_parallel_degree=tp,
    )
    training = TrainingConfig(
        seq_len=spec.model.max_seq_len,
        dtype=dtype,
        mixed_precision_param=mixed_precision_param,
        enable_cpu_offload=enable_cpu_offload,
    )

    spec.model.update_from_config(config=_RuntimeConfig(parallelism))

    with torch.device("meta"):
        with set_default_dtype(TORCH_DTYPE_MAP[dtype]):
            model = spec.model.build()

    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=CompileConfig(enable=False),
        ac_config=ActivationCheckpointingConfig(mode=ac_mode),
        dump_folder=dump_folder,
    )
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    return model


def make_adapter(spec, hf_checkpoint: str):
    """State-dict adapter instance (titan-FQN <-> HF-FQN, incl. grouped-expert split)."""
    return spec.state_dict_adapter(spec.model, hf_checkpoint)


def load_hf_checkpoint(model, adapter, hf_checkpoint: str) -> None:
    """Stream HF safetensors directly into the (possibly sharded) titan model via DCP."""
    import torch.distributed.checkpoint as dcp

    hf_sd = adapter.to_hf(model.state_dict())
    dcp.load(hf_sd, storage_reader=adapter.get_hf_storage_reader(hf_checkpoint))
    model.load_state_dict(adapter.from_hf(hf_sd))


def hf_state_dict(model, adapter):
    """Live titan->HF-named view of model weights for sglang weight sync.

    For dense qwen3 this is pure key renaming over the sharded DTensors (zero copy);
    tied-embedding checkpoints correctly omit lm_head (sglang re-ties from embeddings).
    """
    return adapter.to_hf(model.state_dict())
