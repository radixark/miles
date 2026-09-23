"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic DSA (DeepSeek Sparse Attention) training kernels for Blackwell (SM100a / SM103a).

One maintained path for the GLM-5 / DeepSeek-V3.2 (``thd``, absorbed MLA, no sink) and DeepSeek-V4
(``bshd``, MQA, batched, compressed KV, FP32 attention sink) lightning indexer and sparse-attention
kernels, forward and backward.  Generated register-MMA kernels (``mma.sync`` / ``ldmatrix`` /
``cp.async``) without floating-point atomics: key-side gradients are reduced from per-slot FP32
partials in an order fixed by the indices, so repeated passes on identical inputs are bit-identical.

Public surface (see :mod:`.ops`): :func:`sparse_attention`, :func:`lightning_indexer`,
:func:`indexer_logits`, :func:`select_topk`, :func:`indexer_topk_scores`, plus the non-autograd
:func:`sparse_attention_forward` / :func:`sparse_attention_backward` / :func:`indexer_backward`.
"""

from ..cake_native import SUPPORTED_CAPABILITIES, device_arch
from .ops import (
    extract_topk_scores,
    flashinfer_topk,
    get_topk_fn,
    indexer_backward,
    indexer_logits,
    indexer_topk_scores,
    lightning_indexer,
    register_topk_backend,
    select_topk,
    sparse_attention,
    sparse_attention_backward,
    sparse_attention_forward,
    torch_topk,
)

__all__ = [
    "SUPPORTED_CAPABILITIES",
    "device_arch",
    "extract_topk_scores",
    "flashinfer_topk",
    "get_topk_fn",
    "indexer_backward",
    "indexer_logits",
    "indexer_topk_scores",
    "lightning_indexer",
    "register_topk_backend",
    "select_topk",
    "sparse_attention",
    "sparse_attention_backward",
    "sparse_attention_forward",
    "torch_topk",
]
