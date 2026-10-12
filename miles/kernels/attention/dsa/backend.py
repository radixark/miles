"""Backend selection for the DSA (DeepSeek Sparse Attention) indexer and sparse-attention kernels.

``--dsa-attention-backend`` picks one kernel family for both DSA model plugins (GLM-5 / DeepSeek-V3.2
``thd`` MLA and DeepSeek-V4 ``bshd`` MQA):

* ``tilelang`` (default): the fused TileLang kernels in ``miles/kernels/attention/dsa``.
* ``cake``: the generated deterministic kernels in ``miles/kernels/attention/dsa/cake`` (SM100a /
  SM103a): one launch for the batched DeepSeek-V4 indexer, bit-deterministic backward for both
  operators, FP32 attention sink and shared-latent gradients reduced without atomics.
* ``megatron``: Megatron-core's portable DSA path (bridge / LoRA only; not a plugin-spec choice).
"""

DSA_ATTENTION_BACKENDS = ("megatron", "tilelang", "cake")


def resolve_dsa_attention_backend(backend: str | None) -> str:
    """Validate the plugin-side kernel backend; ``None`` and ``megatron`` fall back to ``tilelang``."""
    if backend is None or backend == "megatron":
        # The plugin specs always run the fused miles kernels; ``megatron`` only selects Megatron-core's
        # own attention on the bridge path, where this module is not involved.
        return "tilelang"
    if backend not in ("tilelang", "cake"):
        raise ValueError(f"Unsupported DSA attention backend: {backend!r} (expected one of {DSA_ATTENTION_BACKENDS})")
    return backend


def use_cake_dsa(backend: str | None) -> bool:
    return resolve_dsa_attention_backend(backend) == "cake"


def cake_dsa_ops():
    """Import the generated deterministic DSA operators (raises a clear error when they are missing)."""
    try:
        from miles.kernels.attention.dsa import cake
    except ImportError as exc:
        raise ImportError(
            "DSA attention backend 'cake' requires the generated kernels in miles/kernels/attention/dsa/cake."
        ) from exc
    return cake
