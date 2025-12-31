"""
Device and hardware utilities for Miles RL framework.

This module provides utilities for GPU device management, hardware
monitoring, and multi-GPU coordination for large-scale MoE training.

These utilities support Miles' goals of:
- New hardware support (GB300 and beyond)
- Production-grade reliability
- Memory robustness and efficiency
"""

from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple, Union
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Cache for device properties to avoid repeated API calls
_device_properties_cache: Dict[int, torch.cuda.device_prop] = {}


def get_device_property(device: int, property_name: str):
    """Get a property of a CUDA device.
    
    This function provides a unified interface to query various
    properties of CUDA devices, with caching to reduce overhead.
    
    Args:
        device: The CUDA device ID (0-indexed)
        property_name: One of 'name', 'total_memory', 'compute_cap',
                      'multiprocessor_count', 'max_threads_per_block',
                      'max_threads_per_multiprocessor', 'total_memory_gb',
                      'warp_size', 'arch'
    
    Returns:
        The requested property value, or None if the property doesn't
        exist or CUDA is unavailable.
    
    Raises:
        ValueError: If an unknown property_name is requested.
    
    Example:
        >>> name = get_device_property(0, 'name')
        >>> memory_gb = get_device_property(0, 'total_memory_gb')
        >>> compute_cap = get_device_property(0, 'compute_cap')
        >>> print(f"GPU 0: {name} with {compute_cap[0]}.{compute_cap[1]} compute capability")
    """
    if not torch.cuda.is_available():
        return None
    
    if device < 0 or device >= torch.cuda.device_count():
        return None
    
    # Use cached properties
    if device not in _device_properties_cache:
        _device_properties_cache[device] = torch.cuda.get_device_properties(device)
    
    props = _device_properties_cache[device]
    
    # Map property names to attributes
    property_map = {
        'name': props.name,
        'total_memory': props.total_memory,
        'compute_cap': (props.major, props.minor),
        'multiprocessor_count': props.multi_processor_count,
        'max_threads_per_block': props.max_threads_per_block,
        'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
        'total_memory_gb': props.total_memory / (1024**3),
        'warp_size': props.warp_size,
        'arch': f"sm_{props.major}{props.minor}",
        'bernoulli': props.has_bernoulli,
        'cuda': props.is_cuda,
    }
    
    if property_name not in property_map:
        valid_options = [k for k in property_map.keys() if k != 'arch']
        raise ValueError(f"Unknown property '{property_name}'. Valid options: {valid_options}")
    
    return property_map[property_name]


def get_device_info(device: int = 0) -> Dict:
    """Get comprehensive information about a CUDA device.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Dictionary containing comprehensive device information:
        - 'name': GPU name
        - 'compute_cap': Compute capability tuple
        - 'total_memory_gb': Total memory in GB
        - 'multiprocessor_count': Number of SMs
        - 'max_threads_per_block': Max threads per block
        - 'arch': Architecture string (e.g., 'sm_80')
    
    Example:
        >>> info = get_device_info(0)
        >>> print(f"Using {info['name']} with {info['multiprocessor_count']} SMs")
    """
    if not torch.cuda.is_available():
        return {
            'name': 'CPU',
            'compute_cap': None,
            'total_memory_gb': 0.0,
            'multiprocessor_count': 0,
            'max_threads_per_block': 0,
            'arch': 'cpu',
        }
    
    if device < 0 or device >= torch.cuda.device_count():
        return {
            'name': 'Invalid Device',
            'compute_cap': None,
            'total_memory_gb': 0.0,
            'multiprocessor_count': 0,
            'max_threads_per_block': 0,
            'arch': 'invalid',
        }
    
    torch.cuda.synchronize(device)
    
    return {
        'name': get_device_property(device, 'name'),
        'compute_cap': get_device_property(device, 'compute_cap'),
        'total_memory_gb': get_device_property(device, 'total_memory_gb'),
        'multiprocessor_count': get_device_property(device, 'multiprocessor_count'),
        'max_threads_per_block': get_device_property(device, 'max_threads_per_block'),
        'arch': get_device_property(device, 'arch'),
    }


def get_all_device_info() -> List[Dict]:
    """Get information about all available CUDA devices.
    
    Returns:
        List of device info dictionaries for all GPUs.
    
    Example:
        >>> for i, info in enumerate(get_all_device_info()):
        ...     print(f"GPU {i}: {info['name']} ({info['total_memory_gb']:.1f} GB)")
    """
    if not torch.cuda.is_available():
        return []
    
    return [get_device_info(i) for i in range(torch.cuda.device_count())]


def get_gpu_memory_info(device: int = 0) -> Dict[str, float]:
    """Get comprehensive memory information for a GPU device.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Dictionary containing:
        - 'total_gb': Total GPU memory in GB
        - 'allocated_gb': Currently allocated memory in GB
        - 'reserved_gb': Currently reserved memory in GB
        - 'free_gb': Available (free) memory in GB
        - 'utilization_percent': Memory utilization percentage
    
    Example:
        >>> info = get_gpu_memory_info(0)
        >>> print(f"Used: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB")
    """
    if not torch.cuda.is_available():
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'free_gb': 0.0,
            'utilization_percent': 0.0,
        }
    
    if device < 0 or device >= torch.cuda.device_count():
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'free_gb': 0.0,
            'utilization_percent': 0.0,
        }
    
    torch.cuda.synchronize(device)
    
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total_memory - reserved
    
    return {
        'total_gb': total_memory / (1024**3),
        'allocated_gb': allocated / (1024**3),
        'reserved_gb': reserved / (1024**3),
        'free_gb': free / (1024**3),
        'utilization_percent': (allocated / total_memory * 100) if total_memory > 0 else 0.0,
    }


def get_all_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """Get memory information for all available GPUs.
    
    Returns:
        Dictionary mapping device ID to memory info dictionaries.
    
    Example:
        >>> all_mem = get_all_gpu_memory_info()
        >>> for device_id, info in all_mem.items():
        ...     print(f"GPU {device_id}: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB")
    """
    if not torch.cuda.is_available():
        return {}
    
    return {i: get_gpu_memory_info(i) for i in range(torch.cuda.device_count())}


def get_gpu_utilization(device: int = 0) -> float:
    """Get the current GPU memory utilization percentage.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Memory utilization as a percentage (0.0 to 100.0),
        or -1.0 if CUDA is unavailable.
    
    Example:
        >>> util = get_gpu_utilization()
        >>> if util > 90:
        ...     print("Warning: High GPU memory utilization!")
    """
    info = get_gpu_memory_info(device)
    return info['utilization_percent']


def get_available_gpu_memory(device: int = 0) -> float:
    """Get the available (free) GPU memory in GB.
    
    Args:
        device: The CUDA device ID (default: 0)
    
    Returns:
        Available memory in GB, or -1.0 if CUDA is unavailable
        or the device doesn't exist.
    
    Example:
        >>> free_memory = get_available_gpu_memory()
        >>> if free_memory > 20:
        ...     print("Sufficient memory for large batch")
    """
    if not torch.cuda.is_available():
        return -1.0
    
    if device < 0 or device >= torch.cuda.device_count():
        return -1.0
    
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    reserved = torch.cuda.memory_reserved(device)
    free = props.total_memory - reserved
    
    return free / (1024**3)


def clear_gpu_caches(device: Optional[int] = None) -> None:
    """Clear GPU memory caches for specified device(s).
    
    This function releases cached memory back to the GPU allocator,
    which can help when switching between different workloads or
    when troubleshooting memory issues.
    
    Args:
        device: Specific GPU ID, or None to clear all devices
    
    Example:
        >>> # Clear cache on current device
        >>> clear_gpu_caches()
        >>> # Clear cache on specific device
        >>> clear_gpu_caches(device=1)
        >>> # Clear cache on all devices
        >>> clear_gpu_caches(device=None)
    """
    if not torch.cuda.is_available():
        return
    
    if device is not None:
        if 0 <= device < torch.cuda.device_count():
            torch.cuda.empty_cache()
    else:
        for _ in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()


def print_device_info(device: int = 0, log_rank: bool = True) -> Dict:
    """Print and return comprehensive device information.
    
    Args:
        device: The CUDA device ID (default: 0)
        log_rank: Whether to include rank in log output
    
    Returns:
        Device info dictionary.
    
    Example:
        >>> info = print_device_info(0)
    """
    info = get_device_info(device)
    
    rank = dist.get_rank() if log_rank else 0
    prefix = f"[Rank {rank}]" if log_rank else ""
    
    logger.info(f"{prefix} Device {device}: {info['name']}")
    logger.info(f"{prefix}   Architecture: {info['arch']}")
    logger.info(f"{prefix}   Compute Capability: {info['compute_cap']}")
    logger.info(f"{prefix}   Total Memory: {info['total_memory_gb']:.2f} GB")
    logger.info(f"{prefix}   Multiprocessors: {info['multiprocessor_count']}")
    
    return info


def print_memory_usage(msg: str = "", device: Optional[int] = None, log_rank: bool = True) -> Dict[str, float]:
    """Print current GPU memory usage.
    
    This is a production-grade utility for monitoring memory during
    large-scale MoE training.
    
    Args:
        msg: Optional message to include in log output
        device: The CUDA device ID, or None to use current device
        log_rank: Whether to include rank in log output
    
    Returns:
        Memory info dictionary.
    
    Example:
        >>> print_memory_usage("Before forward pass")
    """
    if device is None:
        device = torch.cuda.current_device()
    
    mem_info = get_gpu_memory_info(device)
    
    rank = dist.get_rank() if log_rank else 0
    prefix = f"[Rank {rank}]" if log_rank else ""
    
    logger.info(f"{prefix} Memory Usage {msg}: "
                f"Allocated={mem_info['allocated_gb']:.2f}GB, "
                f"Reserved={mem_info['reserved_gb']:.2f}GB, "
                f"Free={mem_info['free_gb']:.2f}GB, "
                f"Util={mem_info['utilization_percent']:.1f}%")
    
    return mem_info


@contextmanager
def device_memory_tracing(device: int = 0) -> Generator[Dict[str, float], None, None]:
    """Context manager to trace memory usage before and after a code block.
    
    This provides a convenient way to measure the memory impact of
    specific operations or code sections in RL training loops.
    
    Args:
        device: The CUDA device ID to trace (default: 0)
    
    Yields:
        A dictionary containing memory statistics that gets updated
        with delta information after the context exits.
    
    Example:
        >>> with device_memory_tracing() as mem:
        ...     outputs = model(inputs)
        >>> print(f"Memory delta: {mem.get('delta_gb', 0):.3f} GB")
    """
    if not torch.cuda.is_available():
        yield {
            'device': device,
            'before_allocated_gb': 0.0,
            'after_allocated_gb': 0.0,
            'delta_gb': 0.0,
        }
        return
    
    torch.cuda.synchronize(device)
    before_allocated = torch.cuda.memory_allocated(device)
    
    yield {
        'device': device,
        'before_allocated_gb': before_allocated / (1024**3),
    }
    
    # After the context, calculate delta
    torch.cuda.synchronize(device)
    after_allocated = torch.cuda.memory_allocated(device)
    
    yield {
        'device': device,
        'before_allocated_gb': before_allocated / (1024**3),
        'after_allocated_gb': after_allocated / (1024**3),
        'delta_gb': (after_allocated - before_allocated) / (1024**3),
    }


def get_device_count() -> int:
    """Get the number of available CUDA devices.
    
    Returns:
        The number of CUDA devices, or 0 if CUDA is unavailable.
    
    Example:
        >>> n_gpus = get_device_count()
        >>> if n_gpus > 1:
        ...     print(f"Running on {n_gpus} GPUs")
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def is_using_gpu() -> bool:
    """Check if the system has CUDA available.
    
    Returns:
        True if CUDA is available and at least one GPU exists.
    
    Example:
        >>> if is_using_gpu():
        ...     print("GPU acceleration available")
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_current_device() -> int:
    """Get the index of the current CUDA device.
    
    Returns:
        The index of the current device, or -1 if no CUDA device.
    
    Example:
        >>> current = get_current_device()
        >>> print(f"Currently using GPU {current}")
    """
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.current_device()


def get_device_name(device: Optional[int] = None) -> str:
    """Get the name of a CUDA device by index or current device.
    
    Args:
        device: The device index, or None to use current device
    
    Returns:
        The GPU name, or "CPU" if no GPU available.
    
    Example:
        >>> print(f"Running on: {get_device_name()}")
    """
    if device is None:
        device = get_current_device()
    
    if device < 0:
        return "CPU"
    
    return get_device_property(device, 'name') or "Unknown"


def is_hardware_supported(device: Optional[int] = None) -> Tuple[bool, str]:
    """Check if the hardware meets minimum requirements for Miles.
    
    This function checks for compute capability and memory requirements
    to ensure optimal performance for large-scale MoE training.
    
    Args:
        device: The device to check, or None for current device
    
    Returns:
        Tuple of (is_supported: bool, reason: str)
    
    Example:
        >>> supported, reason = is_hardware_supported()
        >>> if not supported:
        ...     print(f"Warning: {reason}")
    """
    if not is_using_gpu():
        return False, "CUDA is not available"
    
    if device is None:
        device = get_current_device()
    
    compute_cap = get_device_property(device, 'compute_cap')
    if compute_cap is None:
        return False, "Could not determine compute capability"
    
    major, minor = compute_cap
    
    # Minimum: Ampere (sm_80) or newer for FlashAttention-3 and DeepGEMM
    if major < 8:
        return False, f"Compute capability {major}.{minor} is below minimum (8.0)"
    
    # Check memory (minimum 40GB for production workloads)
    total_memory_gb = get_device_property(device, 'total_memory_gb')
    if total_memory_gb < 40:
        return False, f"Only {total_memory_gb:.1f}GB memory, minimum 40GB recommended"
    
    return True, "Hardware meets requirements"


def get_nccl_backend_info() -> Dict:
    """Get information about the NCCL backend for distributed training.
    
    Returns:
        Dictionary with NCCL configuration and debug info.
    
    Example:
        >>> nccl_info = get_nccl_backend_info()
        >>> print(f"NCCL Version: {nccl_info['nccl_version']}")
    """
    try:
        nccl_version = torch.distributed.NCCL_VERSION
    except AttributeError:
        nccl_version = "Unknown"
    
    try:
        nccl_debug = os.environ.get('NCCL_DEBUG', 'INFO')
    except:
        nccl_debug = "Unknown"
    
    return {
        'nccl_version': nccl_version,
        'debug_level': nccl_debug,
        'backend': 'nccl',
        'distributed': dist.is_initialized(),
    }


def estimate_model_memory_requirements(
    num_parameters: int,
    precision: str = "fp16",
    num_layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
    vocab_size: Optional[int] = None,
    num_experts: Optional[int] = None,
    moe_layer_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """Estimate memory requirements for a model based on parameters.
    
    This is particularly useful for planning large MoE model training
    and understanding memory requirements for production workloads.
    
    Args:
        num_parameters: Total number of parameters in the model
        precision: Data precision ("fp32", "fp16", "bf16", "fp8")
        num_layers: Number of transformer layers (for better estimates)
        hidden_size: Hidden size dimension (for KV cache estimates)
        vocab_size: Vocabulary size (for embedding estimates)
        num_experts: Number of experts in MoE models
        moe_layer_ratio: Fraction of layers that are MoE (e.g., 0.25)
    
    Returns:
        Dictionary with estimated memory requirements in GB:
        - 'weights_gb': Estimated weight memory
        - 'activations_gb': Estimated activation memory
        - 'kv_cache_per_token_gb': KV cache per token
        - 'moe_overhead_gb': Additional overhead for MoE routing
        - 'total_estimate_gb': Total estimated memory
    
    Example:
        >>> est = estimate_model_memory_requirements(
        ...     num_parameters=400_000_000_000,  # 400B model
        ...     precision="fp16",
        ...     num_layers=128,
        ...     hidden_size=2048,
        ...     num_experts=64,
        ...     moe_layer_ratio=0.25
        ... )
        >>> print(f"Weights: {est['weights_gb']:.1f} GB")
        >>> print(f"Total: {est['total_estimate_gb']:.1f} GB")
    """
    # Bytes per parameter based on precision
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
    }
    
    bytes_per_param = precision_bytes.get(precision.lower(), 2)
    
    # Weight memory
    weight_memory = (num_parameters * bytes_per_param) / (1024**3)
    
    estimates: Dict[str, float] = {
        "weights_gb": weight_memory,
    }
    
    # Activation memory estimate
    if num_layers is not None:
        # Rough estimate: ~2x params for full activation during training
        activation_memory = (num_parameters * bytes_per_param * 2) / (1024**3)
        estimates["activations_gb"] = activation_memory
    
    # KV cache estimate
    if num_layers is not None and hidden_size is not None:
        # For each layer: 2 * hidden_size * bytes_per_param
        kv_per_token = (2 * num_layers * hidden_size * bytes_per_param) / (1024**3)
        estimates["kv_cache_per_token_gb"] = kv_per_token
    
    # Embedding memory
    if vocab_size is not None and hidden_size is not None:
        embedding_memory = (vocab_size * hidden_size * bytes_per_param) / (1024**3)
        estimates["embeddings_gb"] = embedding_memory
    
    # MoE-specific overhead
    if num_experts is not None and moe_layer_ratio is not None:
        # MoE routing adds overhead for expert selection and load balancing
        moe_overhead = weight_memory * moe_layer_ratio * 0.1  # ~10% overhead
        estimates["moe_overhead_gb"] = moe_overhead
    
    # Total estimate
    total = weight_memory
    for key, value in estimates.items():
        if key != "weights_gb":
            total += value
    estimates["total_estimate_gb"] = total
    
    return estimates
