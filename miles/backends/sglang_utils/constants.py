"""SGLang GPU memory tag constants, importable even when sglang is not installed."""

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
except ImportError:
    GPU_MEMORY_TYPE_WEIGHTS = "weights"
    GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"

__all__ = [
    "GPU_MEMORY_TYPE_CUDA_GRAPH",
    "GPU_MEMORY_TYPE_KV_CACHE",
    "GPU_MEMORY_TYPE_WEIGHTS",
]
