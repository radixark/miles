"""Standalone GPU stress workload for E2E fault injection.

Saturates all visible GPUs with continuous matmul operations.
Runs until killed or --duration timeout is reached.
"""

from __future__ import annotations

import time
from typing import Annotated

import typer

_DEFAULT_MATRIX_SIZE = 4096

app = typer.Typer()


def _stress_loop(duration: float, matrix_size: int = _DEFAULT_MATRIX_SIZE) -> None:
    import torch

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices available")

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


@app.command()
def main(
    duration: Annotated[float, typer.Option(help="Max duration in seconds")] = 3600.0,
    matrix_size: Annotated[int, typer.Option(help="Square matrix dimension for matmul")] = _DEFAULT_MATRIX_SIZE,
) -> None:
    """GPU stress workload: saturates all visible GPUs with continuous matmul."""
    _stress_loop(duration=duration, matrix_size=matrix_size)


if __name__ == "__main__":
    app()
