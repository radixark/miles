"""Prebuild the aiter JIT modules that sglang's aiter attention backend compiles on first use.

The base image builds aiter with PREBUILD_KERNELS=1, which leaves out the mha_batch_prefill
(extend) modules, and the pa_ragged (decode) kernels are only ever compiled at runtime. So every
fresh container spends minutes in hipcc before its first rollout. This covers bf16 models with a
bf16 KV cache and no ALiBi, logits soft-cap, sliding window or sinks; anything else still JITs.

The build host has no GPU, so GPU_ARCHS must name the target. `import aiter` probes the GPU, so
this loads aiter's JIT core by path, the way aiter's own setup.py does.
"""

import ast
import concurrent.futures
import importlib.util
import os
import shutil
import sys
import types

import torch

AITER_DIR = os.path.dirname(importlib.util.find_spec("aiter").submodule_search_locations[0])
sys.path[:0] = [os.path.join(AITER_DIR, "aiter"), AITER_DIR]

from csrc.cpp_itfs import utils as itfs_utils  # noqa: E402
from csrc.cpp_itfs.pa import pa_ragged  # noqa: E402
from jit import core  # noqa: E402

# Decode shapes: GQA ratio x head size x ceil(context / (256 * 64)). npar_loops <= 4 covers
# contexts up to 64K, which every model the nightly runs fits in.
GQA_RATIOS = range(1, 17)
HEAD_SIZES = (64, 128)
NPAR_LOOPS = range(1, 5)


def _check_archs(so_path, gfx):
    # aiter rebuilds a module whose device code does not include the running GPU's arch.
    archs = core._so_offload_archs(so_path)
    assert archs == {gfx}, f"{so_path} targets {sorted(archs)}, not {gfx}"


def _mha_batch_prefill_gen():
    """aiter's own cmdGenFunc_mha_batch_prefill, lifted out of aiter/ops/mha.py, which imports aiter."""
    path = os.path.join(AITER_DIR, "aiter", "ops", "mha.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmdGenFunc_mha_batch_prefill"]
    fn.decorator_list = []
    scope = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "Generator": torch.Generator,
        "dtypes": types.SimpleNamespace(bf16=torch.bfloat16, fp8=torch.float8_e4m3fnuz),
        "CK_DIR": core.CK_DIR,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), scope)
    return scope[fn.name]


def build_mha_batch_prefill(gfx):
    gen = _mha_batch_prefill_gen()
    q = torch.empty(0, dtype=torch.bfloat16)
    # sglang's extend call; a one-token extend (max_seqlen_q=1) drops the causal mask.
    for max_seqlen_q in (2, 1):
        args = core.get_args_of_build("module_mha_batch_prefill")
        args.update(
            gen(
                q, q, q, None, None, None,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=4096, dropout_p=0.0, softmax_scale=1.0,
                logits_soft_cap=0.0, zero_tensors=False, is_causal=True, window_size_left=-1,
                window_size_right=-1, sink_size=0, return_softmax_lse=False, return_dropout_randval=False,
            )
        )  # fmt: skip
        md_name = args["md_name"]
        core.build_module(
            md_name,
            args["srcs"],
            args["flags_extra_cc"],
            args["flags_extra_hip"],
            args["blob_gen_cmd"],
            args["extra_include"],
            args["extra_ldflags"],
            args["verbose"],
            args["is_python_module"],
            args["is_standalone"],
            args["torch_exclude"],
            args.get("third_party", []),
            args.get("hipify", False),
            flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
        )
        # The build tree holds the generated blobs and objects; only the .so is loaded.
        shutil.rmtree(os.path.join(core.bd_dir, md_name))
        so_path = os.path.join(core.get_user_jit_dir(), f"{md_name}.so")
        _check_archs(so_path, gfx)
        print(f"[aiter] {md_name}.so: {os.path.getsize(so_path) / 2**20:.0f} MiB", flush=True)


def _mp_lock(lock_path, main_func, final_func=None, wait_func=None):
    # The stock lock imports the aiter package. Every build here has its own folder, so none contend.
    main_func()
    if final_func is not None:
        final_func()


def _run_lib(func_name, folder=None):
    # Stands in for the dlopen of the result, which needs the GPU runtime.
    return folder or func_name


def build_pa_ragged(gfx, jobs):
    itfs_utils.mp_lock = _mp_lock
    itfs_utils.run_lib = _run_lib

    def build(shape):
        gqa_ratio, head_size, npar_loops = shape
        # The argument values and types match sglang's paged_attention_ragged call, which keys the build.
        folder = pa_ragged.compile(
            gqa_ratio, head_size, npar_loops,
            "__hip_bfloat16", "__hip_bfloat16", "auto", "__hip_bfloat16",
            1, False, 256, 1, False,
        )  # fmt: skip
        build_dir = os.path.join(itfs_utils.BUILD_DIR, folder)
        # Each build copies the CK headers into its folder; only lib.so is loaded.
        for entry in os.listdir(build_dir):
            path = os.path.join(build_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif entry != "lib.so":
                os.remove(path)
        return folder

    shapes = [(g, h, n) for g in GQA_RATIOS for h in HEAD_SIZES for n in NPAR_LOOPS]
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        folders = list(pool.map(build, shapes))

    total = 0
    for folder in folders:
        lib = os.path.join(itfs_utils.BUILD_DIR, folder, "lib.so")
        _check_archs(lib, gfx)
        total += os.path.getsize(lib)
    print(f"[aiter] {len(folders)} pa_ragged kernels in {itfs_utils.BUILD_DIR}: {total / 2**20:.0f} MiB", flush=True)


def main():
    gfx = os.environ.get("GPU_ARCHS", "")
    if not gfx.startswith("gfx") or ";" in gfx:
        sys.exit(f"set GPU_ARCHS to the one target arch (got {gfx!r})")
    build_mha_batch_prefill(gfx)
    # Sizes MAX_JOBS to the host's cores and free memory.
    core.check_and_set_ninja_worker()
    build_pa_ragged(gfx, int(os.environ["MAX_JOBS"]))


if __name__ == "__main__":
    main()
