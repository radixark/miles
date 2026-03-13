"""Standalone GPU health-check script, executed as a subprocess.

Runs pynvml extended checks and a deterministic compute fingerprint on all
visible GPUs, then prints a JSON array of per-GPU results to stdout.

Each GPU result includes nvml health status and a SHA256 hash of a
deterministic computation.  Healthy GPUs of the same architecture should
produce identical hashes; the caller compares hashes across nodes to find
outliers (cf. ByteRobust bit-wise alignment test).

Usage::

    python -m miles.utils.ft.agents.diagnostics.utils.gpu_check_script

The caller (GpuNodeExecutor) launches this via asyncio.create_subprocess_exec
so that pynvml init/shutdown and torch computation happen in an isolated
process and never block the NodeAgent event loop.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _NvmlCheckResult:
    ecc_errors_uncorrectable: int
    retired_pages_count: int
    power_state_abnormal: bool
    row_remap_failure: bool


@dataclass
class GpuCheckResult:
    gpu_index: int
    nvml_passed: bool = False
    ecc_errors_uncorrectable: int = 0
    retired_pages_count: int = 0
    power_state_abnormal: bool = False
    row_remap_failure: bool = False
    compute_hash: str = ""
    compute_error: str = ""
    details: str = ""


_ABNORMAL_POWER_STATES = frozenset({8, 15})

_COMPUTE_SEED = 42
_HIDDEN_DIM = 512
_NUM_HEADS = 8
_FFN_DIM = 2048
_NUM_LAYERS = 3
_SEQ_LEN = 128
_BATCH_SIZE = 4


def main() -> None:
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import pynvml

    logger.info("diagnostics: gpu check script starting, initializing nvml")
    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info("diagnostics: gpu check found devices: count=%s", device_count)
        model_and_input = _build_deterministic_model_and_input()

        results: list[GpuCheckResult] = []
        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                result = _check_single_gpu(
                    gpu_index=i,
                    handle=handle,
                    model_and_input=model_and_input,
                )
                logger.debug("diagnostics: gpu check result: gpu=%s, nvml_passed=%s, details=%s", i, result.nvml_passed, result.details)
            except Exception as exc:
                msg = f"check failed: {exc}"
                logger.error("diagnostics: gpu check failed: gpu=%s", i, exc_info=True)
                result = GpuCheckResult(gpu_index=i, compute_error=msg, details=msg)
            results.append(result)
    finally:
        pynvml.nvmlShutdown()
        logger.info("diagnostics: gpu check script complete, nvml shut down")

    json.dump([asdict(r) for r in results], sys.stdout)


def _check_single_gpu(
    gpu_index: int,
    handle: object,
    model_and_input: tuple[Any, Any, Any],
) -> GpuCheckResult:
    """Run all checks on one GPU and produce a GpuCheckResult."""
    failures: list[str] = []

    nvml = _check_nvml(handle)

    if nvml.ecc_errors_uncorrectable > 0:
        failures.append(f"uncorrectable ECC errors: {nvml.ecc_errors_uncorrectable}")
    if nvml.retired_pages_count > 0:
        failures.append(f"retired pages: {nvml.retired_pages_count}")
    if nvml.power_state_abnormal:
        failures.append("abnormal power state")
    if nvml.row_remap_failure:
        failures.append("row remap failure")

    nvml_passed = len(failures) == 0

    compute_hash = ""
    compute_error = ""
    try:
        compute_hash = _compute_fingerprint(gpu_index, *model_and_input)
        logger.debug("diagnostics: compute fingerprint: gpu=%s, hash=%s", gpu_index, compute_hash)
    except Exception as exc:
        compute_error = str(exc)
        logger.error("diagnostics: compute fingerprint failed: gpu=%s", gpu_index, exc_info=True)
        failures.append(f"compute fingerprint failed: {exc}")

    details = "; ".join(failures) if failures else "all checks passed"

    return GpuCheckResult(
        gpu_index=gpu_index,
        nvml_passed=nvml_passed,
        **asdict(nvml),
        compute_hash=compute_hash,
        compute_error=compute_error,
        details=details,
    )


def _check_nvml(handle: object) -> _NvmlCheckResult:
    """Run pynvml extended checks on a single GPU handle.

    Individual NVML queries may raise NVMLError_NotSupported on GPUs that
    lack the feature (e.g. ECC on consumer cards, row-remap on some SKUs).
    Treat "not supported" as healthy for that check.
    """
    import pynvml

    try:
        ecc_uncorrectable: int = pynvml.nvmlDeviceGetTotalEccErrors(
            handle,
            pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            pynvml.NVML_VOLATILE_ECC,
        )
    except pynvml.NVMLError_NotSupported:
        ecc_uncorrectable = 0

    try:
        retired_double_bit = pynvml.nvmlDeviceGetRetiredPages(
            handle,
            pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
        )
        retired_pages_count = len(retired_double_bit)
    except pynvml.NVMLError_NotSupported:
        retired_pages_count = 0

    try:
        power_state: int = pynvml.nvmlDeviceGetPowerState(handle)
        power_state_abnormal = power_state in _ABNORMAL_POWER_STATES
    except pynvml.NVMLError_NotSupported:
        power_state_abnormal = False

    try:
        remap_info = pynvml.nvmlDeviceGetRemappedRows(handle)
        row_remap_failure = bool(remap_info[3])
    except pynvml.NVMLError_NotSupported:
        row_remap_failure = False

    return _NvmlCheckResult(
        ecc_errors_uncorrectable=ecc_uncorrectable,
        retired_pages_count=retired_pages_count,
        power_state_abnormal=power_state_abnormal,
        row_remap_failure=row_remap_failure,
    )


def _build_deterministic_model_and_input() -> tuple[Any, Any, Any]:
    """Build a small transformer decoder and fixed input on CPU in bfloat16.

    Uses a causal (autoregressive) decoder to mirror actual LLM workloads.
    The model exercises: matmul (attention QKV projections, FFN), causal
    masked attention (triangular mask + softmax), layer normalization
    (reduction + elementwise), and GELU activation.

    Returns (model, input_tensor, causal_mask), all on CPU in bfloat16.
    Called once; reused for all GPUs.
    """
    import torch
    import torch.nn as nn

    gen = torch.Generator(device="cpu").manual_seed(_COMPUTE_SEED)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=_HIDDEN_DIM,
        nhead=_NUM_HEADS,
        dim_feedforward=_FFN_DIM,
        batch_first=True,
        dropout=0.0,
    )
    model = nn.TransformerDecoder(decoder_layer, num_layers=_NUM_LAYERS)

    for param in model.parameters():
        param.data = torch.randn(param.shape, generator=gen) * 0.02

    model.bfloat16().eval()

    x = torch.randn(_BATCH_SIZE, _SEQ_LEN, _HIDDEN_DIM, generator=gen).bfloat16()
    causal_mask = nn.Transformer.generate_square_subsequent_mask(_SEQ_LEN)

    return model, x, causal_mask


def _compute_fingerprint(
    gpu_index: int,
    model: Any,
    x: Any,
    causal_mask: Any,
) -> str:
    """Run deterministic forward pass on a single GPU and return SHA256 hash.

    Uses a deep copy of the model so each GPU gets an independent instance.
    """
    import torch

    device = torch.device(f"cuda:{gpu_index}")

    model_gpu = copy.deepcopy(model).to(device)
    x_gpu = x.to(device)
    mask_gpu = causal_mask.to(device)

    with torch.no_grad():
        output = model_gpu(tgt=x_gpu, memory=x_gpu, tgt_mask=mask_gpu)

    output_bytes = output.cpu().float().contiguous().numpy().tobytes()
    digest = hashlib.sha256(output_bytes).hexdigest()

    del model_gpu, x_gpu, mask_gpu, output
    torch.cuda.empty_cache()

    return digest


if __name__ == "__main__":
    main()
