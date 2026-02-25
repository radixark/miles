"""
Patch fla 0.4.0 + Triton 3.5.1 for Blackwell (sm103a / GB300).

Applied at Docker build time when ENABLE_CUDA_13=1.

Bug 1 — fla wy_fast.py backward kernel (Triton issue #8695, fla PR #687):
  Triton 3.5.1 lowers `x += tl.dot(a, b)` → `tt.dot(a, b, x)` creating SSA
  self-reference. TritonGPUHoistTMEMAlloc then fails with MLIR dominance error.
  Fix: safe_dot() wraps tl.dot result in mov.f32 inline ASM to prevent fusion.

Bug 2 — Triton jit.py gluon_ir import crash (arm64 cu130 build-specific):
  triton/runtime/jit.py create_specialize_impl() unconditionally imports
  triton.experimental.gluon which requires gluon_ir, not compiled in this build.
  Crashes every @triton.autotune kernel's first run on a cache miss.
  Fix: wrap the import in try/except so autotuning falls back gracefully.

Bug 3 — Triton gluon _semantic.py gluon_ir import (same root cause as Bug 2):
  Triggered via compiler.py -> code_generator import chain on kernel compilation.
  Fix: same try/except guard.
"""

import os
import shutil
import sys

FLA_WY_FAST = "/usr/local/lib/python3.12/dist-packages/fla/ops/gated_delta_rule/wy_fast.py"
TRITON_JIT = "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py"
TRITON_GLUON_SEMANTIC = "/usr/local/lib/python3.12/dist-packages/triton/experimental/gluon/language/_semantic.py"
TRITON_CACHE = os.path.expanduser("~/.triton/cache")

# safe_dot block to insert after `import triton.language as tl`
SAFE_DOT_CODE = r"""

import torch as _fla_torch
_FLA_IS_NVIDIA_BLACKWELL = (
    _fla_torch.cuda.is_available() and _fla_torch.cuda.get_device_capability()[0] == 10
)

if _FLA_IS_NVIDIA_BLACKWELL:
    @triton.jit
    def safe_dot(a, b):
        # Blackwell workaround: prevent TritonGPUHoistTMEMAlloc from fusing dot+add.
        # See: triton-lang/triton#8695, fla-org/flash-linear-attention#638 PR#687
        return tl.inline_asm_elementwise(
            asm="mov.f32 $0, $1;",
            constraints="=r,r",
            args=[tl.dot(a, b)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
else:
    @triton.jit
    def safe_dot(a, b):
        return tl.dot(a, b)
"""


def patch_wy_fast():
    print("=== Patching fla wy_fast.py (Bug 1: safe_dot) ===")
    if not os.path.exists(FLA_WY_FAST):
        print(f"  SKIP: {FLA_WY_FAST} not found")
        return True

    with open(FLA_WY_FAST) as f:
        content = f.read()

    if "safe_dot" in content:
        print(f"  SKIP: already patched")
        return True

    TARGET_IMPORT = "import triton.language as tl"
    if TARGET_IMPORT not in content:
        print(f"  ERROR: could not find '{TARGET_IMPORT}' in wy_fast.py")
        return False

    content = content.replace(TARGET_IMPORT, TARGET_IMPORT + SAFE_DOT_CODE, 1)

    OLD = "b_dk += tl.dot(tl.trans(b_dA), b_kb)"
    NEW = "b_dk += safe_dot(tl.trans(b_dA), b_kb)"
    n = content.count(OLD)
    if n == 0:
        print(f"  ERROR: target line not found: {OLD!r}")
        return False

    content = content.replace(OLD, NEW)
    print(f"  Replaced {n} occurrence(s): tl.dot -> safe_dot")

    with open(FLA_WY_FAST, "w") as f:
        f.write(content)
    print(f"  Patched: {FLA_WY_FAST}")
    return True


def patch_triton_jit():
    print("=== Patching triton jit.py (Bug 2: gluon_ir import) ===")
    if not os.path.exists(TRITON_JIT):
        print(f"  SKIP: {TRITON_JIT} not found")
        return True

    with open(TRITON_JIT) as f:
        content = f.read()

    if "gluon_ir not compiled" in content:
        print(f"  SKIP: already patched")
        return True

    OLD = "    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor"
    NEW = """    try:
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor
    except (ImportError, ModuleNotFoundError):
        GluonTensorDescriptor = type(None)  # gluon_ir not compiled on this build"""

    if OLD not in content:
        print(f"  ERROR: target line not found in jit.py — Triton version may differ")
        return False

    content = content.replace(OLD, NEW, 1)
    with open(TRITON_JIT, "w") as f:
        f.write(content)
    print(f"  Patched: {TRITON_JIT}")
    return True


def patch_triton_gluon_semantic():
    print("=== Patching triton gluon _semantic.py (Bug 3: gluon_ir root cause) ===")
    if not os.path.exists(TRITON_GLUON_SEMANTIC):
        print(f"  SKIP: {TRITON_GLUON_SEMANTIC} not found")
        return True

    with open(TRITON_GLUON_SEMANTIC) as f:
        content = f.read()

    if "gluon_ir not compiled" in content:
        print(f"  SKIP: already patched")
        return True

    OLD = "from triton._C.libtriton.gluon_ir import GluonOpBuilder"
    NEW = """try:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
except (ImportError, ModuleNotFoundError):
    class GluonOpBuilder:  # gluon_ir not compiled in this build
        def __getattr__(self, name):
            raise NotImplementedError(f"GluonOpBuilder.{name}: gluon_ir not compiled")"""

    if OLD not in content:
        print(f"  ERROR: target line not found in _semantic.py")
        return False

    content = content.replace(OLD, NEW, 1)
    with open(TRITON_GLUON_SEMANTIC, "w") as f:
        f.write(content)
    print(f"  Patched: {TRITON_GLUON_SEMANTIC}")
    return True


def clear_triton_cache():
    if os.path.exists(TRITON_CACHE):
        shutil.rmtree(TRITON_CACHE)
        print(f"  Cleared Triton cache: {TRITON_CACHE}")
    else:
        print(f"  Triton cache not found (nothing to clear)")


if __name__ == "__main__":
    ok1 = patch_wy_fast()
    ok2 = patch_triton_jit()
    ok3 = patch_triton_gluon_semantic()
    print("=== Clearing Triton cache ===")
    clear_triton_cache()
    if not (ok1 and ok2 and ok3):
        print("FAILED. Check errors above.")
        sys.exit(1)
    print("Done.")
