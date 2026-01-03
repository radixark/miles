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

