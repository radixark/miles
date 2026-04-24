#!/usr/bin/env python3
# /// script
# dependencies = ["typer"]
# ///
"""Miles artifact CLI — build Miles Docker images at pinned commits.

Layout philosophy:
    Part 1 = NightFall sglang preset (verbatim from NightFall-nda/preset.py
             `flavor_sglang_sunrise`, H200/B200 branch).
    Part 2 = Miles-specific layers stacked on top (from .rcli/preset.py
             `flavor_sunrise_miles`).
    Part 3 = COPY pinned source trees + editable installs.
    Part 4 = DeepGEMM (the NightFall preset's setup-time step, placed last
             because install.sh pulls tvm-ffi 0.1.10 which we then repin).

Two environments: H200 (x86 cu129, the focus) and GB300 (arm64 cu130, stub).

Usage:
    uv run miles_artifact.py build-h200
    uv run miles_artifact.py build-h200 --tag myrepo/miles:test --push
    uv run miles_artifact.py build-gb300      # stub
"""

import os
import random
import subprocess
import time
from pathlib import Path

import typer

# ================================ Pinned commits ================================
SGLANG_COMMIT = "3c485ff302310e73dad450b5889648402b3e561c"    # NightFall/optimize
MILES_COMMIT = "aa55ae379c121cc0c95f46ea45a347ad205afe36"     # miles-sunrise/sunrise_202603
MEGATRON_COMMIT = "fa613058c27066952b2164e5aed5e068634dc281"  # megatron-sunrise/sunrise_202603

REPO_SGLANG = "DarkSharpness/NightFall"
REPO_MILES = "fzyzcjy/miles-sunrise"
REPO_MEGATRON = "fzyzcjy/megatron-sunrise"

# ====================== Third-party pins (miles-only layer) =====================
MILES_WHEELS_URL = "https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64"
TORCH_MEMORY_SAVER_COMMIT = "d64a6394d1e09c613fab90260054cecc2684586d"
MBRIDGE_COMMIT = "89eb10887887bc74853f89a4de258c0702932a1c"
TRANSFORMERS_COMMIT = "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc"
NCCL_TESTS_COMMIT = "ae98985f5599617be94042f4aa3637d10014ce89"

# ====================== Third-party pins (NightFall preset) =====================
# Source: DarkSharpness/NightFall origin/rcli-config preset.py flavor_sglang_sunrise
DEEPGEMM_COMMIT = "7f2a70"
FLASHINFER_JIT_CACHE_VER = "0.6.8"
# tilelang 0.1.8: sglang mhc.py calls T.gemm(wg_wait=0), which was removed from the
# public T.gemm signature in 0.1.9 (moved to T.wgmma_gemm). Must pin 0.1.8.
TILELANG_VER = "0.1.8"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

app = typer.Typer(help="Build Miles Docker images at pinned commits")


# ================================== Dockerfiles =================================

DOCKERFILE_H200 = f"""\
FROM lmsysorg/sglang:v0.5.7

WORKDIR /root/

# ================================================================================
# PART 1 — NightFall sglang preset (flavor_sglang_sunrise, H200/B200)
# Source: NightFall-nda/preset.py
# ================================================================================

# --- preset build_commands ---
RUN pip install tilelang=={TILELANG_VER}
RUN pip install flashinfer-jit-cache=={FLASHINFER_JIT_CACHE_VER} --index-url https://flashinfer.ai/whl/cu129
RUN cd /tmp && git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && \\
    cd flash-mla && git submodule update --init --recursive && \\
    pip install --no-build-isolation -v . && \\
    cd /tmp && rm -rf flash-mla

# preset setup_command `pip install -e sglang/python/` is deferred to Part 3
# (after COPY). DeepGEMM step is deferred to Part 4 (last — tvm-ffi repin).

# ================================================================================
# PART 2 — Miles-specific additions (on top of the sglang preset)
# Source: sunrise/.rcli/preset.py flavor_sunrise_miles
# ================================================================================

# --- apt: miles needs ethtool + nccl-tests binaries that NightFall preset skips ---
RUN apt update && apt install -y nvtop rsync dnsutils ethtool
RUN apt remove -y libgtest-dev || true

RUN git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests && \\
    cd /tmp/nccl-tests && git checkout {NCCL_TESTS_COMMIT} && \\
    make -j$(nproc) CUDA_HOME=/usr/local/cuda && \\
    cp /tmp/nccl-tests/build/*_perf /usr/local/bin/ && \\
    rm -rf /tmp/nccl-tests

# --- prebuilt wheels (flash_attn, FA3, apex) — rebuilding takes 30+ min, reuse ---
RUN mkdir -p /tmp/wheels && cd /tmp/wheels && \\
    curl -fSL -O {MILES_WHEELS_URL}/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl && \\
    curl -fSL -O {MILES_WHEELS_URL}/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl && \\
    curl -fSL -O {MILES_WHEELS_URL}/apex-0.1-cp312-cp312-linux_x86_64.whl && \\
    pip install --no-deps /tmp/wheels/flash_attn-*.whl /tmp/wheels/flash_attn_3-*.whl /tmp/wheels/apex-*.whl && \\
    rm -rf /tmp/wheels
# FA3 wheel installs flash_attn_interface.py at site-packages root;
# TE 2.10 imports it as flash_attn_3.flash_attn_interface — mirror it.
RUN python -c "import site, shutil; sp = site.getsitepackages()[0]; \\
shutil.copy(f'{{sp}}/flash_attn_interface.py', f'{{sp}}/flash_attn_3/flash_attn_interface.py')"

# --- fast-hadamard-transform: used by Miles DSv4 rotate_activation plugin ---
RUN cd /tmp && git clone https://github.com/Dao-AILab/fast-hadamard-transform.git && \\
    cd fast-hadamard-transform && pip install -e . -v --no-build-isolation

# --- TE + Miles numerics stack ---
RUN pip install nvidia-mathdx==25.6.0
RUN pip install --no-build-isolation transformer_engine[core_cu12,pytorch]==2.10.0
RUN pip install --no-deps git+https://github.com/ISEEKYAN/mbridge.git@{MBRIDGE_COMMIT}
RUN pip install flash-linear-attention==0.4.0
# torch_memory_saver @d64a639 (NOT @dc68769 — the newer commit regresses on B200
# with tms_get_interesting_region AssertionError in MegatronTrainRayActor init).
RUN pip install --no-cache-dir --force-reinstall \\
    git+https://github.com/fzyzcjy/torch_memory_saver.git@{TORCH_MEMORY_SAVER_COMMIT}
RUN pip install --no-build-isolation git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl
RUN pip install --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
RUN pip install "numpy<2"

# tvm-ffi 0.1.9 — tilelang compat; DeepGEMM's install.sh will upgrade to 0.1.10,
# we re-pin in Part 4.
RUN pip install apache-tvm-ffi==0.1.9

# cudnn pin (pytorch/pytorch#168167)
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# ================================================================================
# PART 3 — Pinned source trees + editable installs
# ================================================================================

RUN pip uninstall -y miles megatron sglang transformers deep-gemm deep_gemm || true
RUN rm -rf /sgl-workspace/sglang /root/miles /root/Megatron-LM

COPY sglang /root/sglang
COPY megatron /root/Megatron-LM
COPY miles /root/miles

# transformers + miles deepseekv32 patch (must come before miles install)
RUN cd /tmp && git clone https://github.com/huggingface/transformers.git && \\
    cd transformers && git checkout {TRANSFORMERS_COMMIT} && \\
    git apply /root/miles/docker/deepseekv32/transformers.patch && \\
    pip install -e .

# sglang without --no-deps so pyproject.toml pulls sgl-kernel==0.3.21,
# quack-kernels, cuda-python, nvidia-cutlass-dsl — the versions NightFall was
# built against.  This is the NightFall preset's `setup_command`.
RUN pip install -e /root/sglang/python

RUN pip install -r /root/miles/requirements.txt
RUN pip install -e /root/miles
RUN pip install --no-deps -e /root/Megatron-LM

# ================================================================================
# PART 4 — DeepGEMM (NightFall preset setup_command, deferred)
# install.sh pulls tvm-ffi 0.1.10; pin back to 0.1.9 for tilelang.
# ================================================================================

RUN cd /tmp && git clone https://github.com/sgl-project/DeepGEMM.git -b release && \\
    cd DeepGEMM && git checkout {DEEPGEMM_COMMIT} && \\
    git submodule update --init --recursive && bash install.sh
RUN pip install --no-deps apache-tvm-ffi==0.1.9

# ================================================================================
# Cleanup
# ================================================================================
# Note: /tmp/transformers and /tmp/fast-hadamard-transform were `pip install -e .`
# (editable) — their .dist-info points at the source dir, so deleting the source
# breaks import. Only rm the truly wheel-installed leftovers.
RUN rm -rf /root/.cache/pip /tmp/DeepGEMM
"""

DOCKERFILE_GB300 = f"""\
# GB300 is a thin relayer on top of the prebuilt miles-sunrise image.
# Heavy deps (flash-attn, FA3, apex, TE[core_cu13], mbridge, flash-linear-attention,
# Megatron-Bridge, modelopt, fast-hadamard-transform, numpy) are already baked into
# yuemingy/miles-sunrise:gb300 (built from NightFall-nda artifact_cli.py
# DOCKERFILE_MILES_GB300). We upgrade what the base image ships stale:
#
#   Upgrade          | Base image state          | Target (NightFall GB300 preset)
#   -----------------|---------------------------|----------------------------------
#   cuda-python      | baseline                  | --upgrade
#   flashinfer-python| 0.6.2                     | {FLASHINFER_JIT_CACHE_VER}
#   flashinfer-cubin | 0.6.2                     | {FLASHINFER_JIT_CACHE_VER}
#   flashinfer-      | 0.6.2 --index-url cu129   | {FLASHINFER_JIT_CACHE_VER} --index-url cu130
#     jit-cache      |   (cu129 is a bug on cu130)|
#     (jit-cache, python, and cubin must match version; flashinfer's runtime
#      version check errors out if they differ — RuntimeError at JIT import time.)
#   DeepGEMM         | not installed             | @{DEEPGEMM_COMMIT} + cutlass symlinks + install.sh
#   torch_           | @dc6876905 (regresses on   | @{TORCH_MEMORY_SAVER_COMMIT[:7]}
#     memory_        |   B200: tms_get_            |
#     saver          |   interesting_region)      |
#
# Everything else (apt packages, TE, mbridge, etc.) stays as baked.
FROM yuemingy/miles-sunrise:gb300

WORKDIR /root/

# ---------- PART 1: NightFall sglang preset (GB300 branch) upgrades ----------
RUN pip install cuda-python --upgrade
# flashinfer {{python,cubin,jit-cache}} must all match 0.6.8 — else JIT import raises
# `RuntimeError: flashinfer-jit-cache version does not match flashinfer version`.
RUN pip install flashinfer-python=={FLASHINFER_JIT_CACHE_VER} flashinfer-cubin=={FLASHINFER_JIT_CACHE_VER}
RUN pip install flashinfer-jit-cache=={FLASHINFER_JIT_CACHE_VER} --index-url https://flashinfer.ai/whl/cu130
# DeepGEMM (base image lacks it; NightFall preset installs with cutlass symlinks).
RUN cd /tmp && git clone https://github.com/sgl-project/DeepGEMM.git -b release && \\
    cd DeepGEMM && git checkout {DEEPGEMM_COMMIT} && \\
    git submodule update --init --recursive && \\
    ln -sf $(pwd)/third-party/cutlass/include/cutlass $(pwd)/deep_gemm/include/cutlass && \\
    ln -sf $(pwd)/third-party/cutlass/include/cute $(pwd)/deep_gemm/include/cute && \\
    bash install.sh && \\
    cd / && rm -rf /tmp/DeepGEMM

# ---------- PART 2: Miles-side upgrades ----------
# torch_memory_saver swap (see table above).
RUN pip install --no-cache-dir --force-reinstall \\
    git+https://github.com/fzyzcjy/torch_memory_saver.git@{TORCH_MEMORY_SAVER_COMMIT}

# ---------- PART 3: Pinned source trees + editable installs ----------
RUN pip uninstall -y miles megatron sglang || true
RUN rm -rf /sgl-workspace/sglang /root/miles /root/Megatron-LM

COPY sglang /root/sglang
COPY megatron /root/Megatron-LM
COPY miles /root/miles

# --no-deps: base image already resolves all cu130/arm64 deps (sgl-kernel,
# cuda-python, etc.). If pyproject pins drift, drop --no-deps on the affected line.
RUN pip install --no-deps -e /root/sglang/python
RUN pip install --no-deps -e /root/miles
RUN pip install --no-deps -e /root/Megatron-LM

RUN rm -rf /root/.cache/pip
"""


# =================================== Commands ===================================


@app.command("build-h200")
def build_h200(
    sglang_commit: str = typer.Option(SGLANG_COMMIT),
    miles_commit: str = typer.Option(MILES_COMMIT),
    megatron_commit: str = typer.Option(MEGATRON_COMMIT),
    tag: str | None = typer.Option(None, help="Override image tag."),
    push: bool = typer.Option(False, help="docker push after build."),
):
    """Build the H200 (x86, CUDA 12.9) Miles artifact."""
    if tag is None:
        tag = (
            f"radixark/miles-sunrise:h200-"
            f"sgl-{sglang_commit[:7]}-"
            f"miles-{miles_commit[:7]}-"
            f"mega-{megatron_commit[:7]}"
        )
    repos = [
        (REPO_SGLANG, "sglang", sglang_commit),
        (REPO_MILES, "miles", miles_commit),
        (REPO_MEGATRON, "megatron", megatron_commit),
    ]
    _build_image(DOCKERFILE_H200, repos, tag, push=push)


@app.command("build-gb300")
def build_gb300(
    sglang_commit: str = typer.Option(SGLANG_COMMIT),
    miles_commit: str = typer.Option(MILES_COMMIT),
    megatron_commit: str = typer.Option(MEGATRON_COMMIT),
    tag: str | None = typer.Option(None),
    push: bool = typer.Option(False),
):
    """Build the GB300 (arm64, CUDA 13) Miles artifact — thin relayer over yuemingy/miles-sunrise:gb300."""
    if tag is None:
        tag = (
            f"radixark/miles-sunrise:gb300-"
            f"sgl-{sglang_commit[:7]}-"
            f"miles-{miles_commit[:7]}-"
            f"mega-{megatron_commit[:7]}"
        )
    repos = [
        (REPO_SGLANG, "sglang", sglang_commit),
        (REPO_MILES, "miles", miles_commit),
        (REPO_MEGATRON, "megatron", megatron_commit),
    ]
    _build_image(DOCKERFILE_GB300, repos, tag, push=push)


# ==================================== Helpers ===================================


def _build_image(
    dockerfile_content: str,
    repos: list[tuple[str, str, str]],
    tag: str,
    push: bool,
) -> None:
    dir_tmp = Path(f"/tmp/miles-artifact-{int(time.time())}-{random.randint(1000, 9999)}")
    dir_tmp.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Working directory: {dir_tmp}")

    (dir_tmp / "Dockerfile").write_text(dockerfile_content)
    typer.echo(f"Dockerfile written to {dir_tmp / 'Dockerfile'}")

    for repo_name, local_name, commit in repos:
        typer.echo(f"\n=== Cloning {repo_name} -> {local_name} @ {commit} ===")
        _clone_repo(repo_name, dir_tmp / local_name, commit)

    typer.echo(f"\n=== Building {tag} ===")
    _exec(f"cd {dir_tmp} && docker build . -t {tag}")

    if push:
        typer.echo(f"\n=== Pushing {tag} ===")
        _exec(f"docker push {tag}")

    typer.echo(f"\nDone! Image: {tag}")


def _clone_repo(repo_name: str, repo_dir: Path, commit: str) -> None:
    url = f"https://{GITHUB_TOKEN}@github.com/{repo_name}" if GITHUB_TOKEN else f"https://github.com/{repo_name}"
    log_url = f"https://github.com/{repo_name}"
    cmd = (
        f"rm -rf {repo_dir} && "
        f"git clone --recursive {url} {repo_dir} && "
        f"cd {repo_dir} && "
        f"git fetch --all && "
        f"git checkout {commit} && "
        f"git reset --hard {commit} && "
        f"find . -name .git -exec rm -rf {{}} + 2>/dev/null || true"
    )
    _exec(cmd, log=cmd.replace(url, log_url))


def _exec(cmd: str, log: str | None = None) -> None:
    typer.echo(f"$ {log or cmd}")
    subprocess.run(["bash", "-c", cmd], check=True)


if __name__ == "__main__":
    app()
