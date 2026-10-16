"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Native loading for the generated deterministic chunked GDN training kernels (SM100a / SM103a).

``registry.json`` next to this file is written mechanically by the kernel generator: one record per
stage with the argument plan, launch block, dynamic shared memory, the device/launcher source pair
under ``csrc/`` and the compile flags. The generated device code is the same text for SM100a and
SM103a (``__CUDA_ARCH__`` guards where they differ), so one source pair serves both architectures
(``arches``); a stage exported with a different schedule per architecture carries its pairs under
``per_arch`` (``dqkwg``: eight warps on SM103a). Each stage builds on first use as a torch CUDA
extension for the current device's architecture (``nvcc`` + ``ninja``; cached per architecture under
``TORCH_EXTENSIONS_DIR``).
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


def _load_registry() -> dict:
    modules: dict = {}
    path = _PACKAGE / "registry.json"
    if not path.is_file():
        return modules
    for stage, record in json.loads(path.read_text()).items():
        per_arch = record.pop("per_arch", None)
        for arch in record["arches"]:
            modules.setdefault(stage, {})[arch] = record if per_arch is None else {**record, **per_arch[arch]}
    return modules


# MODULES[stage][arch] -> generated module record (from registry.json); a stage exported once for every
# architecture shares one record object across them.
MODULES = _load_registry()


def device_arch(device=None):
    """Map the current CUDA device to the exported architecture tag."""
    capability = torch.cuda.get_device_capability(device)
    try:
        return SUPPORTED_CAPABILITIES[capability]
    except KeyError:
        raise NotImplementedError(
            f"the deterministic chunked GDN training kernels require SM100a or SM103a, got {capability}"
        ) from None


@cache
def load(stage, arch):
    """Build (once per extension cache) and import the native module for ``stage`` on ``arch``."""
    from torch.utils.cpp_extension import load as load_extension

    record = MODULES[stage][arch]
    return load_extension(
        name=f"{record['name']}_{arch}",
        sources=[str(_CSRC / relative) for relative in record["sources"]],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=[_ARCH_FLAGS[arch], "-O3", *_CONVERSION_FLAGS, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("MILES_GDN_CHUNK_TRAIN_VERBOSE_BUILD", "0") == "1",
    )


@cache
def prebuild(arch, max_workers=None):
    """Build every stage for ``arch`` up front, in parallel, and return the wall time in seconds.

    The stages build lazily on first use otherwise; with the register-scan variants selected by shape,
    a new (num_seqs, heads) combination seen mid-training would compile a stage inside a training step
    (about a minute per stage, serially).  Called once from ``chunk_gated_delta_rule`` on its first call
    on a device; ``MILES_GDN_CHUNK_TRAIN_PREBUILD=0`` disables it.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    stages = [stage for stage, per_arch in MODULES.items() if arch in per_arch]
    t0 = time.perf_counter()
    workers = max_workers or min(len(stages), max(1, (os.cpu_count() or 8) // 2))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda stage: load(stage, arch), stages))
    except Exception:  # a concurrent build failure is not fatal: fall back to the serial, cached path
        for stage in stages:
            load(stage, arch)
    return time.perf_counter() - t0


_GRID_INDEX = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


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
        # The thin launchers encode their TMA descriptors inside the binding (no host workspace), so the plan only
        # carries buffers, parameters and the grid.
        unknown = sorted({kind for kind, _ in self._arg_plan} - {"buffer", "parameter", "grid"})
        if unknown:
            raise ValueError(f"{stage}: unsupported argument kinds {unknown!r} in the exported plan")
        self._binding_names = frozenset(name for kind, name in self._arg_plan if kind != "grid")

    def launch(self, *, grid, **bindings):
        # Hot path (one call per kernel launch): a single set comparison validates the bindings; the ordered
        # argument list follows the exported plan.
        if len(bindings) != len(self._binding_names) or not self._binding_names.issuperset(bindings):
            missing = sorted(self._binding_names.difference(bindings))
            unexpected = sorted(set(bindings).difference(self._binding_names))
            raise KeyError(f"{self.stage}: missing bindings {missing!r}, unexpected bindings {unexpected!r}")
        if len(grid) < 3:
            grid = tuple(grid) + (1,) * (3 - len(grid))
        args = []
        for kind, name in self._arg_plan:
            if kind == "grid":
                args.append(int(grid[_GRID_INDEX[name]]))
            else:
                args.append(bindings[name])
        return self._call(*args)

    def launcher(self, names):
        """Positional launcher for one call site (host-path hot loop, round R6-7).

        ``names`` lists every buffer and parameter name of the exported plan exactly once, in the order the call site
        passes its values.  The order is resolved against the plan ONCE here; the returned ``run(grid_x, grid_y, grid_z,
        *values)`` forwards the values to the native entry in plan order with the grid appended -- no per-launch keyword
        dictionary, set comparison or plan walk (``launch`` keeps the keyword form for tests and one-off callers).  A name
        set that does not match the plan raises here, never at launch time; a value count that does not match raises a
        TypeError naming the stage.  Launch geometry and argument values are untouched: the same ints and tensors reach
        ``cudaLaunchKernelExC`` as through ``launch``.
        """
        names = tuple(names)
        plan = tuple(name for kind, name in self._arg_plan if kind != "grid")
        grid_names = tuple(name for kind, name in self._arg_plan if kind == "grid")
        if sorted(names) != sorted(plan) or len(set(names)) != len(names):
            raise KeyError(
                f"{self.stage}: launcher names {sorted(set(names) - set(plan))!r} unknown / {sorted(set(plan) - set(names))!r} missing"
            )
        if (
            grid_names != ("grid_x", "grid_y", "grid_z")
            or tuple(kind for kind, _ in self._arg_plan[-3:]) != ("grid",) * 3
        ):
            raise ValueError(
                f"{self.stage}: the exported plan does not end with grid_x, grid_y, grid_z: {self._arg_plan!r}"
            )
        call, stage, count = self._call, self.stage, len(plan)
        if names == plan:

            def run(grid_x, grid_y, grid_z, *values):
                if len(values) != count:
                    raise TypeError(f"{stage}: launcher expects {count} values, got {len(values)}")
                return call(*values, grid_x, grid_y, grid_z)

        else:
            order = tuple(names.index(name) for name in plan)  # call-site position of every plan slot

            def run(grid_x, grid_y, grid_z, *values):
                if len(values) != count:
                    raise TypeError(f"{stage}: launcher expects {count} values, got {len(values)}")
                return call(*[values[i] for i in order], grid_x, grid_y, grid_z)

        run.stage, run.arch, run.names, run.plan = self.stage, self.arch, names, plan
        return run


@cache
def kernel(stage, arch):
    """Return the process-wide NativeKernel for ``stage`` on ``arch``."""
    return NativeKernel(stage, arch)


@cache
def launcher(stage, arch, names):
    """Process-wide positional launcher of ``stage`` on ``arch`` for the call-site order ``names`` (see ``NativeKernel.launcher``)."""
    return kernel(stage, arch).launcher(names)
