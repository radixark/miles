## Sunrise Env

### 1. The Sunrise Model

```bash
docker run --name miles_tom \
  --gpus all --ipc=host --shm-size=16g \
  --privileged \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -v /data:/data \
  -v /data/tom:/host_home \
  -v /data/tom/models:/root/models \
  -v /data/tom/datasets:/root/datasets \
  -e WANDB_KEY="287b33ca363437decf04a21e95579694c6e301ea" \
  -d radixark/miles:latest \
  sleep infinity

docker exec -it miles_tom /bin/zsh

# Install miles
(cd /host_home/primary_synced/miles-sunrise && pip install -e .)

# Install megatron
(cd /host_home/primary_synced/megatron-sunrise && pip install -e .)

# Install tilelang
apt remove -y libgtest-dev
pip install z3-solver cython

(cd /tmp && rm -rf tilelang && git clone https://github.com/tile-ai/tilelang.git && cd tilelang && pip install -e . -v --no-build-isolation)

# install fast-hamadard-transform
(cd /tmp && rm -rf fast-hadamard-transform && git clone https://github.com/Dao-AILab/fast-hadamard-transform.git && cd fast-hadamard-transform && pip install -e . -v --no-build-isolation)

# do NOT apply patch there
pip install transformers==4.57.1
# apply transformers patch (because HF does not support new deepseek models)
# (cd /tmp && git clone https://github.com/huggingface/transformers.git && cd transformers && git checkout 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc &&  git apply /host_home/primary_synced/miles-sunrise/docker/deepseekv4/transformers.patch && pip install -e .)

(cd /tmp && rm -rf flash-mla && git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && cd flash-mla && git submodule update --init --recursive && pip install --no-build-isolation -v .)

pip install --no-deps --force-reinstall -e /host_home/primary_synced/NightFall/python

################################################

cd /host_home/primary_synced/miles-sunrise

# run once
python scripts/run_deepseek_v4.py prepare-single --model-name DeepSeek-V4-285B-5layer --megatron-model-type deepseek-v4-285B-5layer

# run once
python scripts/run_deepseek_v4.py prepare-spmd --model-name DeepSeek-V4-285B-5layer --megatron-model-type deepseek-v4-285B-5layer

# launch training
python scripts/run_deepseek_v4.py train --model-name DeepSeek-V4-285B-5layer --megatron-model-type deepseek-v4-285B-5layer
```

### 2. DeepSeek V3.2

DeepSeek v3.2 environment, tested on B200

```bash
docker run --name miles_yueming \
  --gpus all --ipc=host --shm-size=16g \
  --privileged \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -v /home/radixark/yueming:/workspace \
  -v /data/yueming/models:/root/models \
  -v /data/yueming/datasets:/root/datasets \
  -e WANDB_KEY="287b33ca363437decf04a21e95579694c6e301ea" \
  -it radixark/miles:latest \
  /bin/zsh

# Install miles
cd miles-sunrise && git checkout sunrise_dev_rebased
pip install -e .

# Install megatron
cd megatron-sunrise
pip install -e .

# Install tilelang
apt remove libgtest-dev
pip install z3-solver
pip install cython

git clone https://github.com/tile-ai/tilelang.git
pip install -e . -v --no-build-isolation

# install fast-hamadard-transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
pip install -e . -v --no-build-isolation

# apply transformers patch (because HF does not support new deepseek models)
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc
git apply ../miles-sunrise/docker/deepseekv4/transformers.patch
pip install -e .

```

#### Run deepseek v3.2 (5 layers) as an example
```bash
# run once
hf download Pinaster/DeepSeek-V3.2-5layer --local-dir /root/models/DeepSeek-V3.2-5layer
# run once
python scripts/run_deepseek_v32.py prepare-single --model-name DeepSeek-V3.2-5layer --megatron-model-type deepseek-v32-5layer
# run once
python scripts/run_deepseek_v32.py prepare-spmd --model-name DeepSeek-V3.2-5layer --megatron-model-type deepseek-v32-5layer

# launch training
python scripts/run_deepseek_v32.py train --model-name DeepSeek-V3.2-5layer --megatron-model-type deepseek-v32-5layer
```


<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/radixark/miles/main/imgs/miles_logo.png" alt="logo" width="400" margin="10px"></img>

[![GitHub Repo](https://img.shields.io/badge/github-radixark%2Fmiles-black?logo=github)](https://github.com/radixark/miles)


</div>


> A journey of a thousand miles is made one small step at a time.

**Miles** is an enterprise-facing reinforcement learning framework for **large-scale MoE post-training and production workloads**, forked from and co-evolving with **[slime](https://github.com/THUDM/slime)**.

Miles keeps slime’s lightweight, modular design, but focuses on:

- New hardware support (e.g., GB300 and beyond)  
- Stable, controllable RL for large MoE models  
- Production-grade features  


## News

- [2025/12] Support FSDP2 as A Training Backend for Miles ([blog](https://lmsys.org/blog/2025-12-03-miles-fsdp/)).
- [2025/11] Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL ([blog](https://lmsys.org/blog/2025-11-25-fp8-rl/)).
- [2025/11] Power Up Speculative Decoding In Reinforcement Learning ([blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md)).
- [2025/11] Introduce Miles - born after slime towards enterprise RL training ([blog](https://lmsys.org/blog/2025-11-19-miles/)).


---

## Table of Contents
- [Quick Start](#quick-start)
- [Arguments Walkthrough](#arguments-walkthrough)
- [Developer Guide](#developer-guide)
- [Recent Updates](#recent-updates)
- [Roadmap](#roadmap)
- [Architecture Overview](#architecture-overview)
- [FAQ & Acknowledgements](#faq--acknowledgements)

---

## Quick Start

> **Note:** Miles is under active development. Commands and examples may evolve; please check the repo for the latest instructions.

For a comprehensive quick start guide covering environment setup, data preparation, training startup, and key code analysis, please refer to:
- [Quick Start Guide](./docs/en/get_started/quick_start.md)

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

---

## Arguments Walkthrough

Arguments in Miles follow the same three-layer pattern as slime:

1. **Megatron arguments**: Megatron arguments are exposed unchanged, e.g. `--tensor-model-parallel-size 2`.

2. **SGLang arguments**: All SGLang arguments are exposed with a prefix `--sglang-`, e.g. `--mem-fraction-static` → `--sglang-mem-fraction-static`.

3. **Miles-specific arguments*: Please refer to [`miles/utils/arguments.py`](miles/utils/arguments.py)  for a full list

For more detailed usage, please refer to the documentation and example configs in the repo as they become available.
 


## Recent Updates

Miles starts from slime’s proven backbone and adds a series of upgrades for production environments. The recent PRs and changes have also been synced to slime side.

### ✅ True On-Policy

Miles extends slime’s deterministic training and supports **infrastructure-level true on-policy support** for SGLang + FSDP:

- Keeps the mismatch between **training** and **inference** effectively at **zero**  
- Aligns numerical behavior end-to-end between training and deployment  
- Uses:
  - FlashAttention-3  
  - DeepGEMM  
  - Batch-invariant kernels from Thinking Machines Lab  
  - `torch.compile` and careful alignment of numeric operations  

This makes Miles suitable for **high-stakes experiments** where repeatability, auditability, and production debugging matter.

### 🧮 Memory Robustness & Efficiency

To fully utilize precious GPU memory **without** constant OOM failures, Miles includes:

- Graceful handling of benign OOMs via error propagation  
- Memory margins to avoid NCCL-related OOM issues  
- Fixes for FSDP excessive memory usage  
- Support for move-based and partial offloading  
- Host peak memory savings for smoother multi-node training  

The goal is to let large MoE jobs run **closer to the hardware limit** while staying stable.

### ⚡ Speculative Training

Miles adds **speculative training** support tailored for RL:

- Performs **online SFT on the draft model during RL**, instead of freezing it  
- Avoids draft policy drift away from the target model  
- Achieves **25%+ rollout speedup** vs. frozen MTP, especially in later training stages  
- Includes:
  - MTP with sequence packing + CP  
  - Proper loss masking and edge-case handling  
  - LM head / embedding gradient isolation  
  - Weight sync flows between Megatron and SGLang  

### 🧱 Hardware & Examples

Miles actively tracks new hardware and provides usable examples:

- GB300 training support, with more recipes coming  
- A **formal mathematics (Lean)** example with SFT / RL scripts, showcasing Miles in a verifiable environment setting  

### 🛠 Miscellaneous Improvements

Additional engineering improvements include:

- Enhanced FSDP training backend  
- Option to deploy the **rollout subsystem independently** outside the main framework  
- Better debugging & profiling: more metrics, post-hoc analyzers, and profiler integration  
- Gradual refactoring for clarity and maintainability  

---

## Roadmap

We are actively evolving Miles toward a **production-ready RL engine** for large-scale MoE and multimodal workloads. Current roadmap items include:

- **Large-scale MoE RL recipes** on new hardware (e.g., GB300 and successors)  
- **Multimodal training** support  
- **Rollout accelerations**  
  - Compatibility with SGLang spec v2 for improved performance  
  - More advanced speculative training schemes (e.g., EAGLE3-style, multi-spec layers)  
- **Elasticity & fault tolerance**  
  - More robust handling of GPU / node failures in long-running jobs  
- **Resource scheduling for async training**  
  - Balancing training and serving in large-scale asynchronous RL systems  

We’ll continue to iterate based on feedback from users across research labs, startups, and enterprise teams.

---

## Architecture Overview

Miles inherits slime’s core architecture as below.


![arch](./imgs/arch.png)


**Module overview:**

- **training (Megatron)**  
  Main training loop. Reads data from the Data Buffer and synchronizes parameters to the rollout subsystem after updates.

- **rollout (SGLang + router)**  
  Generates new samples, including rewards / verifier outputs, and writes them back to the Data Buffer.

- **data buffer**  
  Manages prompt initialization, custom data sources, and rollout generation strategies. Serves as the bridge between training and rollout.

This decoupled design lets you:

- Swap in different algorithms / reward functions without touching rollout code  
- Customize rollout engines independently from training  
- Scale rollouts and training differently depending on hardware and deployment constraints  

---


## Developer Guide

* **Contributions welcome!**
  We’re especially interested in:

  * New hardware backends & tuning
  * MoE RL recipes
  * Stability / determinism improvements
  * Multimodal & speculative training use cases

* We recommend using [pre-commit](https://pre-commit.com/) to keep style consistent:

```bash
apt install pre-commit -y
pre-commit install

# run pre-commit to ensure code style consistency
pre-commit run --all-files --show-diff-on-failure --color=always
```

* For debugging tips, performance tuning, and internal architecture notes, see the `docs/` and `developer_guide/` folders (coming soon).

---

## FAQ & Acknowledgements

* For FAQs, please see `docs/en/get_started/qa.md` (to be added as the project matures).
* **Huge thanks** to the **slime** authors and community — Miles would not exist without slime’s design and ecosystem.
* We also acknowledge and rely on the broader LLM infra ecosystem, including SGLang, Megatron-LM, and related tools.

---

## Links

* **Miles GitHub**: [https://github.com/radixark/miles](https://github.com/radixark/miles)
* **slime GitHub**: [https://github.com/THUDM/slime](https://github.com/THUDM/slime)

We’re excited to see what you build — whether you choose **slime**, **Miles**, or both in different parts of your stack. 🚀

