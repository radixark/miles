# doc-dev: docs/ci/02-docker-build.md
# Build via docker/build.py (single source of truth for variant defaults,
# platforms, tags, and build-arg ordering); callers may append explicit
# build-arg overrides. See docs/ci/02-docker-build.md. Examples:
#       python docker/build.py --variant cu13     --image-tag dev --push   # radixark/miles:dev      (amd64+arm64)
#       python docker/build.py --variant cu13-x86 --image-tag dev --push   # radixark/miles:dev (amd64)
#       python docker/build.py --variant cu13-aarch64 --image-tag dev --push   # radixark/miles:dev (arm64)
#       python docker/build.py --variant cu12-x86 --image-tag dev --push   # radixark/miles:dev-cu12 (CUDA 12.9)
#
# A multi-arch build runs the recipe once per platform; it installs the wheels
# release for the current arch picked from TARGETARCH — WHEELS_TAG_X86 on amd64,
# WHEELS_TAG_ARM64 on arm64 (cu12-x86 overrides WHEELS_TAG_X86).

ARG SGLANG_IMAGE_TAG=v0.5.16
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG SGLANG_BRANCH=sglang-miles
ARG SGLANG_COMMIT=""

ARG MEGATRON_REPO=radixark/Megatron-LM
ARG MEGATRON_BRANCH=miles-main-20260819
# Empty means the branch HEAD at build time; release builds set it to freeze one commit.
ARG MEGATRON_COMMIT=""

ARG ENABLE_CUDA_13=1

ARG WHEELS_REPO=yueming-yuan/miles-wheels
# Two complete wheels release tags (the wheels repo's own names). The build picks
# one by TARGETARCH and installs it verbatim — no on-the-fly tag assembly here.
ARG TARGETARCH
ARG WHEELS_TAG_X86=cu130-x86_64
ARG WHEELS_TAG_ARM64=cu130-aarch64

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
# ethtool for network diagnostics
RUN apt install -y nvtop rsync dnsutils ethtool tmux

# nccl-tests for diagnostics
RUN git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests && \
    cd /tmp/nccl-tests && \
    git checkout ae98985f5599617be94042f4aa3637d10014ce89 && \
    make -j$(nproc) CUDA_HOME=/usr/local/cuda && \
    cp /tmp/nccl-tests/build/*_perf /usr/local/bin/ && \
    rm -rf /tmp/nccl-tests

# ====================================== Collect pre-built wheels ============================================
# Optional: drop pre-built wheels into <miles>/wheels/ to skip the release download below.
# `wheel[s]/` is a glob — silently no-ops when wheels/ is absent (CI's path).
COPY wheel[s]/ /tmp/wheels/

RUN case "${TARGETARCH}" in \
      amd64) WHEELS_TAG="${WHEELS_TAG_X86}" ;; \
      arm64) WHEELS_TAG="${WHEELS_TAG_ARM64}" ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    echo "Fetching wheels release ${WHEELS_TAG}" && \
    mkdir -p /tmp/wheels && \
    curl -sL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}" \
    | python3 -c "import sys,json,subprocess,os; w='/tmp/wheels'; \
[subprocess.run(['curl','-fSL','-o',os.path.join(w,a['name']),a['browser_download_url']],check=True) \
 for a in json.load(sys.stdin).get('assets',[]) \
 if a['name'].endswith(('.whl', '.tar.gz')) and not os.path.exists(os.path.join(w,a['name']))]" && \
    ls -lh /tmp/wheels/

# ====================================== Python dependencies ============================================

# flash-attn
RUN pip install /tmp/wheels/flash_attn-*.whl

# flash-attn hopper (FA3): Hopper-only (sm_90a). Coexists with FA2.
# The wheel ships flash_attn_interface top-level (what transformer_engine imports)
# plus a flash_attn_3.flash_attn_interface re-export shim, so there is nothing to
# fetch separately. Do not overwrite the shim with a copy of the full module: it
# re-runs @torch.library.custom_op("flash_attn_3::...") and double-registers.
RUN pip install /tmp/wheels/flash_attn_3-*.whl

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.2
# required for DeepSeek V4
RUN pip install tilelang==0.1.8 -f https://tile-ai.github.io/whl/nightly/cu128/
RUN pip install --no-deps tile_kernels==1.0.0
# FlashQLA backend for Qwen GDN linear-attention layers (requires SM90+, CUDA 12.8+, PyTorch 2.8+; built on tilelang above)
RUN pip install -v --no-build-isolation "git+https://github.com/QwenLM/FlashQLA.git"
# Prefer the prebuilt wheel; fall back to a source build while the current
# arch's wheels release doesn't carry it yet (QEMU-emulated arm64 source
# builds are the expensive path this avoids).
RUN if ls /tmp/wheels/fast_hadamard_transform-*.whl 2>/dev/null | grep -q .; then \
      pip install /tmp/wheels/fast_hadamard_transform-*.whl; \
    else \
      pip install "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede" --no-build-isolation; \
    fi

# Mamba kernels for nemotron_h hybrid (mamba+attention) models.
RUN if ls /tmp/wheels/causal_conv1d-*.whl 2>/dev/null | grep -q . && \
       ls /tmp/wheels/mamba_ssm-*.whl 2>/dev/null | grep -q .; then \
      pip install /tmp/wheels/causal_conv1d-*.whl /tmp/wheels/mamba_ssm-*.whl; \
    else \
      pip install causal-conv1d==1.6.1 mamba-ssm==2.3.1 --no-build-isolation; \
    fi

# transformer_engine
# --no-deps pins the three TE dists to exactly the wheels above, so pip cannot
# pull a conflicting core dist. That also drops transformer_engine_torch's own
# runtime deps, which still have to be installed: transformer_engine.pytorch
# imports onnxscript unconditionally (module/_common -> export -> onnx_extensions).
RUN --mount=type=bind,source=docker/verify_transformer_engine.py,target=/tmp/verify_transformer_engine.py \
    if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      TE_CORE_DIST=transformer_engine_cu13; \
    else \
      TE_CORE_DIST=transformer_engine_cu12; \
    fi && \
    set -- \
      /tmp/wheels/transformer_engine-2.17.0-py3-none-any.whl \
      /tmp/wheels/${TE_CORE_DIST}-2.17.0-py3-none-manylinux_2_28_*.whl \
      /tmp/wheels/transformer_engine_torch-2.17.0-cp312-cp312-linux_*.whl; \
    if [ "$#" -ne 3 ] || [ ! -f "$1" ] || [ ! -f "$2" ] || [ ! -f "$3" ]; then \
      echo "expected exactly one TransformerEngine 2.17 wheel for each component (${TE_CORE_DIST})" >&2; exit 1; \
    fi && \
    pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch && \
    pip install --force-reinstall --no-deps "$@" && \
    pip install einops onnx onnxscript pydantic nvdlfw-inspect && \
    python3 /tmp/verify_transformer_engine.py "${TE_CORE_DIST}"

# TE patches (cu13): B300/GB300 FA2 and backward override fixes
# te_dequantized_backward_override.patch is a hot fix from
# https://github.com/NVIDIA/TransformerEngine/pull/3141; drop it after TE v2.18.
COPY docker/patch/ /tmp/patches/
# A `for` loop exits with the status of its LAST iteration, so a patch that fails
# to apply here used to be swallowed whenever a later one succeeded -- the image
# shipped with TE silently unpatched. Fail the build instead.
RUN if [ "${ENABLE_CUDA_13}" = "1" ] && [ -d /tmp/patches/cu13 ]; then \
      TE_DIR=$(python -c 'import importlib.util; print(importlib.util.find_spec("transformer_engine").submodule_search_locations[0])') && \
      for p in /tmp/patches/cu13/*.patch; do \
        echo "Applying $(basename $p) to $TE_DIR"; \
        patch -d "$TE_DIR" -p1 < "$p" || { \
          echo "TE patch $(basename $p) did not apply cleanly against the installed transformer_engine" >&2; \
          exit 1; \
        }; \
      done; \
    fi && rm -rf /tmp/patches

# apex
RUN pip install /tmp/wheels/apex-*.whl

RUN git clone https://github.com/${MEGATRON_REPO}.git --recursive -b ${MEGATRON_BRANCH} Megatron-LM && \
    cd Megatron-LM && \
    if [ -n "${MEGATRON_COMMIT}" ]; then git checkout -f ${MEGATRON_COMMIT} && git submodule update --init --recursive; fi && \
    pip install -e .

# Muon optimizer support: megatron/core/optimizer/muon.py requires this for Newton-Schulz
# orthogonalization; not on PyPI (only a dependency-confusion stub), so install from the
# git source Megatron-LM itself pins in pyproject.toml.
RUN pip install "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.3.0"

# torch_memory_saver's source build needs TMS_CUDA_MAJOR; take it from torch.
RUN TMS_CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])") \
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@f05a8754daf68238d54e4cf31cb3ba866684bbaf --no-cache-dir --force-reinstall
RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
# radixark/Megatron-Bridge#24 (TE 2.17 grouped-linear contract), merged to @bridge.
# Pinned by SHA, not branch: buildkit caches this layer on the instruction text, so a
# branch that moves leaves the old revision baked into the image with nothing to show it.
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@7f0fb3456f8ffe47599b5fd167b454605d85f932 --no-deps --no-build-isolation
RUN pip install megatron-energon --no-deps
RUN pip install multi-storage-client --no-deps

COPY requirements.txt /tmp/requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/jwt /usr/lib/python3/dist-packages/PyJWT* && pip install -r /tmp/requirements.txt

# The apt copy shadows the pip one in ldconfig and the loader interleaves the two,
# so transformer_engine gets a mixed set of libcudnn sub-libraries. The packages are
# held, hence --allow-change-held-packages.
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then CU=13; else CU=12; fi; \
  apt-get remove -y --purge --allow-change-held-packages \
    libcudnn9-cuda-${CU} libcudnn9-dev-cuda-${CU} libcudnn9-headers-cuda-${CU} \
  && rm -rf /var/lib/apt/lists/*

RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
    pip install nvidia-cudnn-cu13==9.22.0.52; \
  else \
    pip install nvidia-cudnn-cu12==9.22.0.52; \
  fi


RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Install sglang-miles ============================================

# Install sglang from sglang-miles branch
# The CUDA 12 sglang images sed their dependency markers from cu13 to cu12 in
# python/pyproject.toml and never commit it, so a plain checkout aborts on the dirty
# file. Force it, then re-apply the same rewrite: the tree we just checked out carries
# the cu13 markers, and sglang's editable install makes them visible to every later pip
# resolve, which would drag cu13 packages onto a cu12 image.
RUN cd /sgl-workspace/sglang && \
    git fetch origin ${SGLANG_BRANCH} && \
    if [ -n "${SGLANG_COMMIT}" ]; then \
      git checkout -f ${SGLANG_COMMIT}; \
    else \
      git checkout -f FETCH_HEAD; \
    fi && \
    if [ "${ENABLE_CUDA_13}" != "1" ]; then \
      sed -i 's/cuda-python>=13\.0/cuda-python>=12,<13/' python/pyproject.toml && \
      sed -i 's/flashinfer_python\[cu13\]/flashinfer_python[cu12]/' python/pyproject.toml && \
      sed -i 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' python/pyproject.toml; \
    fi && \
    pip install -e "python[all]" --no-deps

# =========================== Mooncake structured object store wheel ===========================

# Miles' Mooncake backend requires the structured object store API. Install the selected
# CUDA 13 wheel as one coherent Python/native package; CUDA 12 keeps its base Mooncake.
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      set -- /tmp/wheels/mooncake_transfer_engine*-*.whl; \
      if [ "$#" -ne 1 ] || [ ! -e "$1" ]; then \
        echo "expected exactly one mooncake wheel for cu13/${TARGETARCH}" >&2; exit 1; \
      fi && \
      pip install --force-reinstall --no-deps "$1" && \
      python3 -c "from mooncake.structured_object_store import FieldSchema, MooncakeBundleTransfer, export_ref, import_ref; MooncakeBundleTransfer.put; MooncakeBundleTransfer.get; MooncakeBundleTransfer.release_result; MooncakeBundleTransfer.cleanup_dataproto"; \
    else \
      echo "CUDA 12 variant: keeping the base image's mooncake"; \
    fi

COPY docker/install-kube-tools.sh /tmp/install-kube-tools.sh
RUN bash /tmp/install-kube-tools.sh "${TARGETARCH}" && rm -f /tmp/install-kube-tools.sh

# ====================================== Install main package ============================================

ARG MILES_COMMIT=main
RUN git clone https://github.com/radixark/miles.git /root/miles && \
    cd /root/miles && \
    git checkout ${MILES_COMMIT} && \
    pip install -e . --no-deps

# int4_qat
RUN pip install /tmp/wheels/fake_int4_quant_cuda-*.whl

# ====================================== Install sgl-model-gateway ============================================
#   SGL_ROUTER_USE_WHEELS=0:
#     Build from source  https://github.com/radixark/sgl-router-for-miles
#   SGL_ROUTER_USE_WHEELS=1 (default):
#     Install the pre-built sgl-model-gateway wheel

ARG SGL_ROUTER_USE_WHEELS=1
ARG SGL_ROUTER_REPO=https://github.com/radixark/sgl-router-for-miles.git
ARG SGL_ROUTER_BRANCH=main

RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    if [ "${SGL_ROUTER_USE_WHEELS}" = "1" ]; then \
      pip install --force-reinstall /tmp/wheels/sglang_router-*.whl && \
      tar xzf /tmp/wheels/sgl-model-gateway-linux-*.tar.gz -C /usr/local/bin/ && \
      chmod +x /usr/local/bin/sgl-model-gateway; \
    elif [ "${SGL_ROUTER_USE_WHEELS}" = "0" ]; then \
      git clone --branch "${SGL_ROUTER_BRANCH}" --depth 1 "${SGL_ROUTER_REPO}" /build/sgl-model-gateway && \
      curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y && \
      export PATH="/root/.cargo/bin:${PATH}" && \
      python3 -m pip install maturin && \
      cd /build/sgl-model-gateway/bindings/python && \
      ulimit -n 65536 && \
      maturin build --release --features vendored-openssl --out /build/gateway_wheels && \
      cd /build/sgl-model-gateway && \
      cargo build --release --bin sgl-model-gateway --features vendored-openssl && \
      cp target/release/sgl-model-gateway /usr/local/bin/sgl-model-gateway && \
      chmod +x /usr/local/bin/sgl-model-gateway && \
      pip install --force-reinstall /build/gateway_wheels/sglang_router-*.whl && \
      rm -rf /root/.cargo /root/.rustup /build/sgl-model-gateway /build/gateway_wheels; \
    fi

# ==================== Reconcile apache-tvm-ffi with sglang's pin ====================
# sglang v0.5.16 bumped nvidia-cutlass-dsl 4.5.2 -> 4.6.0 while leaving its
# apache-tvm-ffi pin at 0.1.11. cutlass-dsl 4.6.0's tvm_ffi provider calls
#   tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper(..., map_dataclass_to_tuple=...)
# and that parameter only exists in apache-tvm-ffi >= 0.1.10. An earlier install in
# this file drags in an older apache-tvm-ffi, so flashinfer's CuTe rmsnorm path dies
# during CUDA-graph capture with
#   TypeError: make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'
# Restore sglang's own pin after every other install, and fail the build loudly if it
# does not take effect rather than shipping an image that only breaks under CI.
RUN pip install --force-reinstall --no-deps "apache-tvm-ffi==0.1.11" && \
    python3 -c "import inspect; from tvm_ffi.utils import kwargs_wrapper as k; \
p = inspect.signature(k.make_kwargs_wrapper).parameters; \
assert 'map_dataclass_to_tuple' in p, \
  'apache-tvm-ffi resolves to a build too old for nvidia-cutlass-dsl 4.6.0: %s' % list(p)"

# ==================== Pin nvidia-cutlass-dsl to 4.6.2 ====================
# The sglang v0.5.16 base ships nvidia-cutlass-dsl 4.6.0, whose 4.6.0/4.6.1 runtime
# hangs FA4 CuTe backward forever on sm_103 (B300/GB300): the first training backward
# spins at 100% GPU in _bwd_postprocess_convert and never returns
# (https://github.com/Dao-AILab/flash-attention/issues/2797). 4.6.2 is fixed and passes
# FA4's own SM100 forward/backward checks. The meta-package alone does NOT move what a
# process loads — the -libs-cu12/-libs-cu13 wheels carry the actual runtime — so pin
# every component and fail the build loudly if any of them slips.
RUN pip install --force-reinstall --no-deps \
      "nvidia-cutlass-dsl==4.6.2" \
      "nvidia-cutlass-dsl-libs-base==4.6.2" \
      "nvidia-cutlass-dsl-libs-core==4.6.2" \
      "nvidia-cutlass-dsl-libs-cu12==4.6.2" \
      "nvidia-cutlass-dsl-libs-cu13==4.6.2" && \
    python3 -c "from importlib.metadata import version as v; \
libs = ['nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-base', 'nvidia-cutlass-dsl-libs-core', \
        'nvidia-cutlass-dsl-libs-cu12', 'nvidia-cutlass-dsl-libs-cu13']; \
bad = {p: v(p) for p in libs if v(p) != '4.6.2'}; \
assert not bad, 'nvidia-cutlass-dsl components not at 4.6.2: %s' % bad"

# ==================== Pin flashinfer to 0.6.15.post1 ====================
# The sglang v0.5.16 base ships flashinfer 0.6.14, whose trtllm MoE cubins can hang
# 2CTA kernels on Blackwell (regression since 0.6.8, fixed cubins in 0.6.15;
# https://github.com/flashinfer-ai/flashinfer/pull/3973). 0.6.15.post1 additionally
# fixes the host-side MLA/MoE dispatch overhead that got the plain 0.6.15 bump
# reverted upstream (sgl-project/sglang#31625 -> #31927). The python + cubin + jit-cache
# trio must move together; jit-cache wheels live on flashinfer's own index, per CUDA line.
RUN FI_CUDA_INDEX=$([ "${ENABLE_CUDA_13}" = "1" ] && echo cu130 || echo cu129) && \
    pip install --force-reinstall --no-deps \
      --extra-index-url "https://flashinfer.ai/whl" \
      --extra-index-url "https://flashinfer.ai/whl/${FI_CUDA_INDEX}" \
      "flashinfer-python==0.6.15.post1" \
      "flashinfer-cubin==0.6.15.post1" \
      "flashinfer-jit-cache==0.6.15.post1" && \
    python3 -c "from importlib.metadata import version as v; \
vers = {p: v(p) for p in ['flashinfer-python', 'flashinfer-cubin', 'flashinfer-jit-cache']}; \
bad = {p: x for p, x in vers.items() if not x.startswith('0.6.15.post1')}; \
assert not bad, 'flashinfer components not at 0.6.15.post1: %s' % bad"
# The sglang base exports FLASHINFER_VERSION=0.6.14; keep the marker in step.
ENV FLASHINFER_VERSION=0.6.15.post1

RUN rm -rf /tmp/wheels

# The cu130 sglang base ships its baked Rust toolchain outside the default PATH.
ENV PATH="/root/.cargo/bin:${PATH}"
