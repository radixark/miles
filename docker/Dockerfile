# 1. radixark/miles:dev
#    use default settings
#
# 2. radixark/miles:dev-cu13-arm64
#    build-arg:ENABLE_CUDA_13=1 \
#    build-arg:SGLANG_IMAGE_TAG=v0.5.10-cu130 \
#    build-arg:WHEELS_TAG=cu130-aarch64 \

ARG SGLANG_IMAGE_TAG=v0.5.10
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG SGLANG_BRANCH=sglang-miles
ARG SGLANG_COMMIT=""

ARG MEGATRON_REPO=radixark/Megatron-LM
ARG MEGATRON_BRANCH=miles-main

ARG ENABLE_CUDA_13=0

ARG WHEELS_REPO=yueming-yuan/miles-wheels
ARG WHEELS_TAG=cu129-x86_64

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
# ethtool for network diagnostics
RUN apt install -y nvtop rsync dnsutils ethtool

# nccl-tests for diagnostics
RUN git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests && \
    cd /tmp/nccl-tests && \
    git checkout ae98985f5599617be94042f4aa3637d10014ce89 && \
    make -j$(nproc) CUDA_HOME=/usr/local/cuda && \
    cp /tmp/nccl-tests/build/*_perf /usr/local/bin/ && \
    rm -rf /tmp/nccl-tests

# ====================================== Download pre-built wheels ============================================

RUN mkdir -p /tmp/wheels && \
    curl -sL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}" \
    | python3 -c "import sys, json, subprocess; \
[subprocess.run(['curl', '-fSL', '-o', '/tmp/wheels/' + a['name'], a['browser_download_url']], check=True) \
 for a in json.load(sys.stdin)['assets'] if a['name'].endswith(('.whl', '.tar.gz'))]" && \
    ls -lh /tmp/wheels/

# ====================================== Python dependencies ============================================

# flash-attn
RUN pip install /tmp/wheels/flash_attn-*.whl

# flash-attn hopper
RUN pip install /tmp/wheels/flash_attn_3-*.whl && \
    python_path=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p $python_path/flash_attn_3 && \
    curl -fSL https://raw.githubusercontent.com/Dao-AILab/flash-attention/fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89/hopper/flash_attn_interface.py \
      -o $python_path/flash_attn_3/flash_attn_interface.py

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.2
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==25.6.0 && \
      pip -v install --no-build-isolation "transformer_engine[core_cu13,pytorch]==2.10.0"; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"; \
    fi

# apex
RUN pip install /tmp/wheels/apex-*.whl

RUN git clone https://github.com/${MEGATRON_REPO}.git --recursive -b ${MEGATRON_BRANCH} Megatron-LM && \
    cd Megatron-LM && \
    pip install -e .

RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --no-cache-dir --force-reinstall
# RUN pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
RUN pip install git+https://github.com/yushengsu-thu/Megatron-Bridge.git@merged-megatron-0.16.0rc0-miles --no-deps --no-build-isolation
RUN pip install megatron-energon --no-deps
RUN pip install multi-storage-client --no-deps

COPY requirements.txt /tmp/requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/jwt /usr/lib/python3/dist-packages/PyJWT* && pip install -r /tmp/requirements.txt

# https://github.com/pytorch/pytorch/issues/168167
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
    pip install nvidia-cudnn-cu13==9.16.0.29; \
  else \
    pip install nvidia-cudnn-cu12==9.16.0.29; \
  fi

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Install sglang-miles ============================================

# Install sglang from sglang-miles branch
RUN cd /sgl-workspace/sglang && \
    git fetch origin ${SGLANG_BRANCH} && \
    if [ -n "${SGLANG_COMMIT}" ]; then \
      git checkout ${SGLANG_COMMIT}; \
    else \
      git checkout FETCH_HEAD; \
    fi && \
    pip install -e "python[all]" --no-deps

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

RUN rm -rf /tmp/wheels
