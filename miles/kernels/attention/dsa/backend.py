"""Backend selection for the DSA (DeepSeek Sparse Attention) indexer and sparse-attention kernels.

``--dsa-attention-backend`` picks one kernel family for both DSA model plugins (GLM-5 / DeepSeek-V3.2
``thd`` MLA and DeepSeek-V4 ``bshd`` MQA):

* ``tilelang`` (default): the fused TileLang kernels in ``miles/kernels/attention/dsa``.
* ``cake``: the generated deterministic kernels in ``miles/kernels/attention/dsa/cake`` (SM100a /
  SM103a): one launch for the batched DeepSeek-V4 indexer, bit-deterministic backward for both
  operators, FP32 attention sink and shared-latent gradients reduced without atomics.
* ``megatron``: Megatron-core's portable DSA path (bridge / LoRA only; not a plugin-spec choice).

``--miles-dsa-topk-backend cake`` independently selects the generated deterministic indexer top-k
(``cake.cake_topk``); :func:`check_cake_topk_backend` rejects it at startup for models whose
top-k exceeds the kernel's ``CAKE_TOPK_MAX`` slots (GLM-5 / DeepSeek-V3.2 select 2048) or on
non-Blackwell devices.
"""

DSA_ATTENTION_BACKENDS = ("megatron", "tilelang", "cake")

# Limits of the generated top-k kernel (``cake._kernels.TOPK_K`` / ``TOPK_CAPACITIES[-1]``).
CAKE_TOPK_MAX = 512
CAKE_TOPK_MAX_KEYS = 32768


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


def cake_topk_key_bound(seq_length: int | None, compress_ratios=()) -> int | None:
    """Static bound on the key-row length the indexer scores: ``ceil(seq_length / r)`` for the smallest nonzero
    compress ratio ``r`` in ``compress_ratios`` (``seq_length`` itself without compression); ``None`` when
    ``seq_length`` is unknown.  ``seq_length`` is the global sequence length (CP ranks all-gather the keys)."""
    if seq_length is None:
        return None
    ratios = [int(ratio) for ratio in compress_ratios if ratio]
    return -(-int(seq_length) // (min(ratios) if ratios else 1))


def check_cake_topk_backend(
    topk_backend: str | None, index_topk: int | None, *, model: str = "DSA", max_keys: int | None = None
) -> None:
    """Fail at startup when ``--miles-dsa-topk-backend cake`` cannot serve ``model``; other backends pass.

    The generated top-k kernel fills at most ``CAKE_TOPK_MAX`` slots per query row, scores rows of at most
    ``CAKE_TOPK_MAX_KEYS`` keys and runs on SM100a / SM103a only.  ``index_topk`` is the model's per-query
    selection width and ``max_keys`` its static bound on the key-row length (:func:`cake_topk_key_bound`);
    either is ``None`` when the path has no static value, in which case the per-call ``ValueError`` in
    ``cake.cake_topk`` is the final guard.
    """
    if topk_backend != "cake":
        return
    if index_topk is not None and int(index_topk) > CAKE_TOPK_MAX:
        raise ValueError(
            f"--miles-dsa-topk-backend cake supports index_topk <= {CAKE_TOPK_MAX}, but {model} selects "
            f"{int(index_topk)} keys per query; use --miles-dsa-topk-backend torch or flashinfer."
        )
    if max_keys is not None and int(max_keys) > CAKE_TOPK_MAX_KEYS:
        raise ValueError(
            f"--miles-dsa-topk-backend cake scores key rows of at most {CAKE_TOPK_MAX_KEYS} compressed keys, but "
            f"{model} can produce rows of {int(max_keys)} keys at this --seq-length / compress ratio; shorten the "
            "sequence or use --miles-dsa-topk-backend torch or flashinfer."
        )
    import torch

    if torch.cuda.is_available():
        from miles.kernels.attention.dsa.cake._jit import SUPPORTED_CAPABILITIES

        capability = torch.cuda.get_device_capability()
        if capability not in SUPPORTED_CAPABILITIES:
            raise ValueError(
                "--miles-dsa-topk-backend cake requires a Blackwell SM100a / SM103a device (compute capability "
                f"10.0 / 10.3), got {capability[0]}.{capability[1]}; use --miles-dsa-topk-backend torch or flashinfer."
            )


def cake_dsa_ops():
    """Import the generated deterministic DSA operators (raises a clear error when they are missing)."""
    try:
        from miles.kernels.attention.dsa import cake
    except ImportError as exc:
        raise ImportError(
            "DSA attention backend 'cake' requires the generated kernels in miles/kernels/attention/dsa/cake."
        ) from exc
    return cake
