"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Native loading for the generated deterministic DSA (sparse attention + lightning indexer) training kernels (SM100a / SM103a).

``registry_<arch>.json`` next to this file is written mechanically by the kernel generator: one
record per stage with the argument plan, launch block, dynamic shared memory, the device/launcher
source pair under ``csrc/<arch>/`` and the compile flags. Each stage builds on first use as a torch
CUDA extension (``nvcc`` + ``ninja``; cached under ``TORCH_EXTENSIONS_DIR``).
"""

import json
import os
from functools import cache
from pathlib import Path

import torch

_PACKAGE = Path(__file__).resolve().parent
_CSRC = _PACKAGE / "csrc"
_ARCH_FLAGS = {
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
    "sm_103a": "-gencode=arch=compute_103a,code=sm_103a",
}
SUPPORTED_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# torch's extension build disables the half / bf16 conversion operators; the generated device code uses them.
_CONVERSION_FLAGS = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)


def _load_registries() -> dict:
    modules: dict = {}
    for arch in _ARCH_FLAGS:
        path = _PACKAGE / f"registry_{arch}.json"
        if not path.is_file():
            continue
        for stage, record in json.loads(path.read_text()).items():
            modules.setdefault(stage, {})[arch] = record
    return modules


# MODULES[stage][arch] -> generated module record (from registry_<arch>.json).
MODULES = _load_registries()


def device_arch(device=None):
    """Map the current CUDA device to the exported architecture tag."""
    capability = torch.cuda.get_device_capability(device)
    try:
        return SUPPORTED_CAPABILITIES[capability]
    except KeyError:
        raise NotImplementedError(
            f"the deterministic DSA (sparse attention + lightning indexer) training kernels require SM100a or SM103a, got {capability}"
        ) from None


@cache
def load(stage, arch):
    """Build (once per extension cache) and import the native module for ``stage`` on ``arch``."""
    from torch.utils.cpp_extension import load as load_extension

    record = MODULES[stage][arch]
    return load_extension(
        name=record["cache_name"],
        sources=[str(_CSRC / relative) for relative in record["sources"]],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=[_ARCH_FLAGS[arch], "-O3", *_CONVERSION_FLAGS, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("MILES_DSA_TRAIN_VERBOSE_BUILD", "0") == "1",
    )


@cache
def prebuild(arch, max_workers=None):
    """Build every stage for ``arch`` up front, in parallel, and return the wall time in seconds.

    The stages build lazily on first use otherwise; with the attention and indexer variants selected
    by head count, a new head count seen mid-training would compile a stage inside a training step
    (about a minute per stage, serially).  Called once from the operators on their first call on a
    device; ``MILES_DSA_TRAIN_PREBUILD=0`` disables it.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    stages = [stage for stage, per_arch in MODULES.items() if arch in per_arch]
    t0 = time.perf_counter()
    workers = max_workers or min(len(stages), max(1, (os.cpu_count() or 8) // 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda stage: load(stage, arch), stages))
    return time.perf_counter() - t0


class NativeKernel:
    """One generated stage; named bindings are mapped onto the exported argument plan.

    ``launch(grid=(x, y, z), **bindings)`` binds tensors and scalars by their exported names;
    the grid is filled in automatically.
    """

    def __init__(self, stage, arch=None, device=None):
        self.stage = stage
        self.arch = arch or device_arch(device)
        record = MODULES[stage][self.arch]
        self.record = record
        native = load(stage, self.arch)
        self._call = getattr(native, record["ffi_entry"])
        self._arg_plan = tuple(tuple(item) for item in record["arg_plan"])
        workspace_bytes = int(record["tma_workspace_bytes"])
        self.descriptor_storage = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=device if device is not None else "cuda")
            if workspace_bytes
            else None
        )

    def launch(self, *, grid, **bindings):
        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        args = []
        used = set()
        for kind, name in self._arg_plan:
            if kind == "grid":
                args.append(grid[("grid_x", "grid_y", "grid_z").index(name)])
            elif kind == "workspace":
                args.append(self.descriptor_storage)
            else:
                if name not in bindings:
                    raise KeyError(f"{self.stage}: missing binding {name!r}")
                args.append(bindings[name])
                used.add(name)
        unexpected = set(bindings) - used
        if unexpected:
            raise KeyError(f"{self.stage}: unexpected bindings {sorted(unexpected)!r}")
        return self._call(*args)


@cache
def kernel(stage, arch):
    """Return the process-wide NativeKernel for ``stage`` on ``arch``."""
    return NativeKernel(stage, arch)
