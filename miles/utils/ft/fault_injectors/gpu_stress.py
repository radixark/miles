"""Standalone GPU stress workload for E2E fault injection.

Saturates all visible GPUs with continuous matmul operations.
Runs until killed or --duration timeout is reached.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

import typer

logger = logging.getLogger(__name__)

_DEFAULT_MATRIX_SIZE = 4096

app = typer.Typer()


@app.command()
def main(
    duration: Annotated[float, typer.Option(help="Max duration in seconds")] = 3600.0,
    matrix_size: Annotated[int, typer.Option(help="Square matrix dimension for matmul")] = _DEFAULT_MATRIX_SIZE,
) -> None:
    """GPU stress workload: saturates all visible GPUs with continuous matmul."""
    _stress_loop(duration=duration, matrix_size=matrix_size)


def _stress_loop(duration: float, matrix_size: int = _DEFAULT_MATRIX_SIZE) -> None:
    import torch

    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.error("fault_injector: no CUDA devices available for gpu_stress")
        raise RuntimeError("No CUDA devices available")

    logger.info(
        "fault_injector: gpu_stress starting device_count=%d, matrix_size=%d, duration=%s",
        device_count,
        matrix_size,
        duration,
    )
    tensors = []
    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        tensors.append((a, b, device))

    start = time.monotonic()
    while time.monotonic() - start < duration:
        for a, b, _ in tensors:
            torch.mm(a, b)
    logger.info("fault_injector: gpu_stress finished elapsed=%.1fs", time.monotonic() - start)


if __name__ == "__main__":
    app()
