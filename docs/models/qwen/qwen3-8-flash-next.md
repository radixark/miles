---
title: Qwen3.8-Flash-Next
description: RL recipe for Qwen3.8-Flash-Next, the GDN + QSA hybrid MoE preview of the Qwen4 architecture, with hyper-connections and a host-resident PLE table.
---

The complete Qwen3.8-Flash-Next RL implementation is open at the Miles pull request:
[`radixark/miles#2777`](https://github.com/radixark/miles/pull/2777). It lands as a set of
three: that branch, the SGLang
[`sglang-miles-qwen38next`](https://github.com/sgl-project/sglang/tree/sglang-miles-qwen38next)
branch, and [`radixark/Megatron-LM#89`](https://github.com/radixark/Megatron-LM/pull/89).
The image in [section 3](#3-environment-setup) pins all three.

## 1. Model Introduction

[Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
is Qwen's **176 B-parameter (6 B active) GDN + QSA hybrid Mixture-of-Experts preview of the
Qwen4 architecture**. Despite the shared prefix it is not a variant of the dense
[Qwen3.8-27B](/models/qwen/qwen3-8) — it is the next step of the *Next* line that
[Qwen3-Next](/models/qwen/qwen3-next) started, and structurally it has far more in common
with that model than with anything else carrying the Qwen3.8 name.

The block spec is still a uniform GPT decoder, but three components sit outside what a
stock Megatron layer provides, and each one is why a piece of this recipe exists:

**Hyper-connections replace the block layernorms.** The checkpoint ships none: each
hyper-connection's own `hc_norm` *is* the pre-block norm, so the spec drops every block
layernorm and a leftover TE fused norm corrupts the forward pass silently rather than
loudly. Miles fills Megatron's HC `ModuleSpec` slots, but Qwen's output contraction is not
DeepSeek-V4's `learned_output_contract` — it is the same low-rank gated mean as the
per-layer hyper-connections, with the RMS taken per stream rather than over the whole
`n*C` vector. That needed a new Megatron spec slot, `hc_head_contraction`, which is what
Megatron#89 adds; the DeepSeek-V4 default path and its parameter names are untouched.

**QSA (Qwen Sparse Attention) on the full-attention layers.** Twelve of the 48 layers are
full attention. Each projects its own indexer queries and compressed keys, scores them, and
keeps `indexer_budget` key tokens per query; attention then reads only those. Miles
reimplements the indexer and a forward *and backward* Triton sparse-attention kernel rather
than importing SGLang's inference path, with the selection rows built torch-side so the
kernel needs no causal or segment logic of its own.

**A frozen, host-resident PLE table.** The per-layer-embedding n-gram table is ~102 GB. It
lives in host memory, TP-row-sharded, and is deliberately *not* a checkpointed parameter;
token ids reach it over an explicit side channel that raises rather than defaulting when
nothing was published. It is also on the weight-update check's skip list — see
[section 5.5](#55-notable-quirks).

**Key highlights:**

- **48 layers, hybrid**: 36 GDN linear-attention layers + 12 QSA full-attention layers, from
  the released config's `layer_types` (equivalently, every 4th layer is full attention).
- **512-expert MoE at top-10**, `moe-ffn-hidden-size 640`, plus a gated shared expert.
- **Hyper-connections** at every block, with a model-supplied output contraction.
- **PLE** n-gram embeddings, frozen and host-resident.
- **Attention output gate**, `--qk-layernorm`, `--apply-layernorm-1p`.
- **Shape**: hidden 2560, 24 attention heads, 2 query groups, `kv-channels 256`,
  vocab 248320, `--rotary-base 10000000` at `--rotary-percent 0.25`.

Two things the model args deliberately leave out: MTP (`--mtp-num-layers` is omitted, the
MTP tensors are not mapped yet) and the hyper-connection / PLE / QSA fields themselves,
which have no Megatron CLI flags and are derived from the checkpoint by the spec in
`miles_plugins/models/qwen3_8_next/qwen3_8_next.py`.

On the parameter count: Qwen's published headline is 176 B total / 6 B active, while the
docstring in `scripts/models/qwen3.8-flash-next.py` counts 180 B / ~7.4 B from the released
`config.json` shapes. Nothing in the recipe depends on which figure you quote.

## 2. Supported Variants

| Variant | `--model-name` | Layers | Purpose | GPUs |
|---|---|---|---|---|
| Full | `Qwen3.8-Flash-Next` | 48 | the real model | 32 (8 × 4) |
| Smoke slice | `Qwen3.8-Flash-Next-4layer` | 4 | CI and single-node bring-up | 1 × 4 or 1 × 8 |

`--model-name` selects between them and sets the matching `megatron_model_type`
(`qwen3.8-flash-next` / `qwen3.8-flash-next-4layer`), which in turn resolves the model args
and the `torch_dist` path. The launcher asserts the node shape, so a mismatched
`--num-nodes` / `--num-gpus-per-node` fails immediately rather than mid-run.

## 3. Environment Setup

Use the `docker.io/radixark/miles:qwen38next` image. It is the rolling
[`radixark/miles:dev`](/ci/02-docker-build) image with the three moving parts checked out at
the versions this recipe was built against, and nothing else changed — every prebuilt wheel,
TransformerEngine patch and version pin comes from `dev`. It is multi-arch, so the same tag
serves GB300 (aarch64) and x86 nodes.

| Component | Pinned at |
|---|---|
| miles | [`#2777`](https://github.com/radixark/miles/pull/2777) `afd78afd` |
| SGLang | [`sglang-miles-qwen38next`](https://github.com/sgl-project/sglang/tree/sglang-miles-qwen38next) `599d7403` |
| Megatron-LM | [`#89`](https://github.com/radixark/Megatron-LM/pull/89) `e8f57451` |

### 3.1 Download model + dataset

```bash
hf download Qwen/Qwen3.8-Flash-Next --local-dir /root/models/Qwen3.8-Flash-Next
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

For the smoke slice, download
[`CharyZeng/Qwen3.8-Flash-Next-4layer`](https://huggingface.co/CharyZeng/Qwen3.8-Flash-Next-4layer)
into `/root/models/Qwen3.8-Flash-Next-4layer` instead — the launcher resolves
`--hf-checkpoint` as `<--model-dir>/<--model-name>`, which is the same path CI uses.

### 3.2 HF → Megatron `torch_dist` conversion

Unlike the launcher-prepared recipes, this one takes the converted reference checkpoint as
given: `--ref-load` resolves to `<--ckpt-dir>/<megatron_model_type>_torch_dist`, so the
conversion has to have run first. The output re-shards at load, so the conversion layout
does not have to match the training one:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3.8-flash-next)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
CONVERT_KEEP_PP1=1 PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/models/Qwen3.8-Flash-Next \
   --save          /root/ckpt/qwen3.8-flash-next_torch_dist \
   --tensor-model-parallel-size 2 \
   --pipeline-model-parallel-size 1
```

Swap `qwen3.8-flash-next` for `qwen3.8-flash-next-4layer` throughout to convert the smoke
slice.

## 4. Launch

The launcher assumes an **already-running ray cluster**: bring one up across the nodes,
`export MILES_SCRIPT_EXTERNAL_RAY=1`, then run on the head node.

Full model, 8 nodes × 4 GPUs:

```bash
cd /root/miles
python scripts/run_qwen3_8_next.py train \
  --model-name Qwen3.8-Flash-Next \
  --num-nodes 8 --num-gpus-per-node 4 \
  --num-rollout 5 --rollout-max-response-len 4096
```

Single-node smoke slice:

```bash
python scripts/run_qwen3_8_next.py train \
  --model-name Qwen3.8-Flash-Next-4layer \
  --num-nodes 1 --num-gpus-per-node 8
```

Paths come from `--model-dir` (default `/root/models`), `--data-dir` (default
`/root/datasets`), `--ckpt-dir` (default `/root/ckpt`) and `--megatron-path` (default
`/root/Megatron-LM`). Saving is off by default (`skip_saving`); turning it on writes
checkpoints under `<--save-dir>/<--run-id>/checkpoints` every 10 rollouts, without optimizer
or RNG state.

## 5. Recipe Configuration

### 5.1 Parallelism

| Shape | TP | PP | CP | EP | ETP | Rollout engine |
|---|---|---|---|---|---|---|
| 8 × 4 (full) | 2 | 8 | 1 | 4 | 1 | 8 GPUs, SGLang TP 8 / EP 8 |
| 1 × 4 (4layer) | 2 | 2 | 1 | 2 | 1 | 4 GPUs, SGLang TP 4 / EP 4 |
| 1 × 8 (4layer) | 2 | 2 | 1 | 4 | 1 | 4 GPUs, SGLang TP 4 / EP 4 |

`--sequence-parallel` is on in every shape. Activation checkpointing is full and uniform at
one layer, with `--micro-batch-size 1` and `--max-tokens-per-gpu 8192`.

### 5.2 Algorithm

GRPO, DAPO-Math-17k, thinking mode on:

```bash
--advantage-estimator grpo
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
--rollout-batch-size 4
--n-samples-per-prompt 8
--rollout-temperature 0.8
--apply-chat-template-kwargs '{"thinking_mode":"thinking"}'
```

Adam at `--lr 1e-6`, constant schedule, `--weight-decay 0.1`,
`--adam-beta1 0.9 --adam-beta2 0.98`.

### 5.3 Rollout & SGLang

Rollout is **colocated** — trainer and engines share the GPUs — and the trainer offloads to
**disk** rather than host RAM during rollout (`--offload-train-target disk`,
`--offload-train-disk-dir /tmp/train_offload`). Worth keeping in view when sizing hosts: the
PLE table is already holding ~102 GB of host memory for the full model. The 4-layer CI slice
overrides the target back to `--offload-train-target cpu`.

```bash
--sglang-linear-attn-prefill-backend flashinfer   # GDN prefill
--sglang-moe-runner-backend triton
--sglang-chunked-prefill-size 8192
--sglang-disable-radix-cache
--sglang-mem-fraction-static 0.7
--linear-attention-backend flashqla               # trainer-side GDN
--qkv-format thd
```

`QSA_BACKEND=triton` selects the sparse-attention kernel. The recipe also loosens router
health checking (`--router-health-failure-threshold 40`,
`--router-health-check-interval-secs 15`, `--router-health-success-threshold 1`) and raises
`--rollout-health-check-interval` / `--rollout-health-check-timeout` to 300 s.

### 5.4 What CI watches

`tests/e2e/megatron/model_scripts/test_qwen3_8_next_4layer_ci.py` runs the 4-layer slice on
8 × H200 in `stage-c-8-gpu-h200` with rollout-routing replay on, and gates these metrics:

- `train/grad_norm`
- `train/ppo_kl`
- `train/train_rollout_logprob_abs_diff`
- `train/train_rollout_kl`
- `rollout/raw_reward`

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up: it is
the direct measure of whether the SGLang and Megatron forward passes agree, and on this
architecture that covers the GDN, QSA and hyper-connection paths at once. See
[True On-Policy](/examples/infra-features/true-on-policy) for what the metric does and does
not tell you.

### 5.5 Notable quirks

- **Weight-update checking skips two prefixes.** `--check-weight-update-equal` runs with
  `--check-weight-update-skip-list visual. ple_embedding.` — the PLE table is frozen and
  never shipped, so comparing it would fail on a parameter that is working as intended.
- **Every block layernorm is dropped.** The checkpoint has none; each hyper-connection's
  `hc_norm` is the pre-block norm. A leftover TE fused norm corrupts silently.
- **`--moe-aux-loss-coeff 0`.** Routing is not auxiliary-loss balanced here.
- **Triton and Inductor caches are pinned to `/tmp`** (`TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`) with `TORCHINDUCTOR_COMPILE_THREADS=1`.
- **`SGLANG_DISABLE_MULTIMEM_AG=1`** and `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1` are set for
  the engines.
- The model lives in `miles_plugins/models/qwen3_8_next/`, and weight conversion in
  `miles/backends/megatron_utils/megatron_to_hf/qwen3_8_next.py`.

## 6. Pairs Well With

- [Qwen3-Next](/models/qwen/qwen3-next) — the previous generation of the same line
- [Qwen3.8](/models/qwen/qwen3-8) — the dense 27 B that shares the name and not the architecture
- [Disk Offload](/advanced/disk-offload)
- [True On-Policy](/examples/infra-features/true-on-policy)
