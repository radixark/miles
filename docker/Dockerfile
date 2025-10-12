# TDOO blackwell use `v0.5.2rc2-cu128-b200`, is it still needed?
ARG SGLANG_VERSION=latest
FROM lmsysorg/sglang:${SGLANG_VERSION} AS sglang

# hopper/blackwell
ARG TARGET_VARIANT

# we need to write this again after from
ARG SGLANG_VERSION
# TODO shall we lock version
ARG MEGATRON_COMMIT=main

RUN apt update
RUN apt install -y nvtop

# TODO: change to pip install sglang-router after it has a new release
RUN pip install sglang-router --force-reinstall

RUN if [ "${TARGET_VARIANT}" = "hopper" ]; then \
          export TLIST="8.0;8.9;9.0;9.0a"; \
        else \
          export TLIST="8.0;8.9;9.0;9.0a;10.0"; \
        fi && \
    TORCH_CUDA_ARCH_LIST="$TLIST" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# TODO shall we lock version
# apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git \

# TODO shall we unify the version; also shall we lock version
# transformer engine, we install with --no-deps to avoid installing torch and torch-extensions
RUN pip install pybind11
RUN if [ "${TARGET_VARIANT}" = "hopper" ]; then \
      export TE_VERSION=3cd6870; \
    else \
      export TE_VERSION=stable; \
    fi && \
    pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@${TE_VERSION} \

# flash attn
# the newest version megatron supports is v2.7.4.post1
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1

# TODO shall we lock version
# TODO why this is only installed in hopper?
RUN pip install flash-linear-attention

# TODO why this is only installed in blackwell?
RUN git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
RUN python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
   mkdir -p $python_path/flash_attn_3 && \
   wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py

WORKDIR /root/
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && \
    pip install -e .

COPY patch/${SGLANG_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git checkout ${MEGATRON_COMMIT} && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch

# TODO remove after merging to SGLang
COPY patch/${SGLANG_VERSION}/sglang.patch /sgl-workspace/sglang/
RUN cd /sgl-workspace/sglang && \
  git apply sglang.patch && \
  if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
  fi && \
  rm sglang.patch

RUN rm /root/.tmux.conf
