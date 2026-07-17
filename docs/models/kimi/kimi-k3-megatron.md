# Kimi K3 Megatron Backend

This document is the implementation contract for the Kimi K3 Megatron backend.
The SGLang implementation is the numerical golden baseline.

## Pinned references

- SGLang golden: `radixark/sglang-kimi`, PR 2 head
  `cc91738f141ff357d1c161ccb29fc736f3056466`.
- Golden model implementation:
  `sglang/python/sglang/srt/models/kimi_k3.py`.
- Golden attention-residual implementation:
  `sglang/python/sglang/srt/layers/attn_residual.py`.
- HF model metadata was read from `moonshotai/Kimi-K3` on 2026-07-15.
- HF checkpoint metadata: 1,560,860,324,864 bytes, 96 safetensor shards,
  497,220 tensors. Routed-expert weights use compressed-tensors MXFP4.
- Miles Megatron baseline: `4716f7547` on `miles-main` when this work
  started.
- NVIDIA Megatron `dev` checked at `d1384c2d9`.

## Model configuration

| Item | Value |
| --- | --- |
| Language layers | 93 |
| Hidden size | 7168 |
| Vocabulary | 163840 |
| Attention residual block size | 12 |
| KDA heads / head dim | 96 / 128 |
| KDA short convolution | 4 |
| KDA gate lower bound | -5.0 |
| MLA heads | 96 |
| MLA q LoRA / kv LoRA | 1536 / 512 |
| MLA qk NoPE / nominal RoPE / v dims | 128 / 64 / 128 |
| Dense FFN size | 33792 |
| Routed experts / top-k | 896 / 16 |
| Routed latent / expert FFN size | 3584 / 3072 |
| Shared experts | 2 |
| Activation | SiTU, beta=4.0, linear_beta=25.0 |

The HF `linear_attn_config.kda_layers` and `full_attn_layers` lists are
**1-based**. SGLang implements this as `(layer_idx + 1) in kda_layers`.
The repeating pattern is therefore three KDA layers followed by one MLA
layer. The first four zero-based decoder layers are:

| Layer | Attention | Feed-forward | Attention-residual action |
| --- | --- | --- | --- |
| 0 | KDA | dense SiTU MLP | write snapshot 0 |
| 1 | KDA | latent MoE | read snapshot 0 |
| 2 | KDA | latent MoE | read snapshot 0 |
| 3 | NoPE MLA | latent MoE | read snapshot 0 |

This makes a four-layer prefix the minimum checkpoint that covers every
decoder-layer type. It does not cover the second attention-residual block
write, which first occurs at zero-based layer 12.

## Transformer-level data flow

K3 does not use the standard Transformer residual stream. The block state is:

- `prefix_sum`: `[tokens, hidden]`, the live prefix inside the current
  attention-residual block.
- `block_residual`: `[tokens, num_snapshots, hidden]`, snapshots written every
  12 layers.

For rows `R = [snapshot_0, ..., snapshot_n, prefix_sum]`, an aggregation is:

```text
scores_i = score_proj(RMSNorm_score(R_i))
mixed    = sum_i softmax(scores)_i * R_i
output   = RMSNorm_sublayer(mixed)
```

Each decoder layer performs two independent aggregations with separate score
projection and score norm weights:

1. Aggregate and normalize before attention.
2. Update `prefix_sum` with the attention output, aggregate again, and normalize
   before the MLP/MoE.

At a block-write layer (`layer_idx % 12 == 0`), the pre-attention `prefix_sum`
is appended to the snapshot bank and the new live prefix starts from the
attention output. Otherwise the attention output is added to the existing live
prefix. The MLP/MoE output is always added to the live prefix.

After the last decoder layer, one final attention-residual aggregation uses
`model.output_attn_res_{proj,norm}`, followed by the ordinary final RMSNorm and
LM head.

### Pipeline consequence

With PP=1, the snapshot bank can be carried through the existing
`TransformerLayer.forward(...)->(hidden_states, context)` contract. Standard
Megatron discards `context` at a pipeline boundary, so the first backend
version explicitly supports PP=1 only. PP>1 requires extending pipeline
communication to transmit both `prefix_sum` and the snapshot bank.

## Decoder attention variants

### KDA

KDA has independent projections with the following HF weight layout:

```text
q_proj, k_proj, v_proj: 7168 -> 96 * 128
q_conv1d, k_conv1d, v_conv1d: depthwise causal convolution, width 4
f_a_proj: 7168 -> 128
f_b_proj: 128 -> 96 * 128
b_proj: 7168 -> 96
g_proj: 7168 -> 96 * 128
o_norm: per-head gated RMSNorm
o_proj: 96 * 128 -> 7168
A_log: 96 effective values (the checkpoint tensor has 128 entries)
dt_bias: 96 * 128
```

For prefill/training, the FLA `chunk_kda` operator receives q, k, v,
per-channel forget input, and per-head beta. It performs q/k L2 normalization,
safe-gate construction, and the delta-rule recurrence. K3 differs from
Megatron's existing GDN in two non-parameter ways:

- GDN has a scalar decay per head; KDA has a 128-wide decay per head.
- K3 uses the safe gate selected by `lower_bound=-5.0`.

The output is gated by full-rank `g_proj` through sigmoid-gated RMSNorm before
`o_proj`.

Initial backend scope is packed prefill/training. Recurrent decode state support
can reuse the Megatron GDN inference-context design after prefill parity is
established.

### NoPE MLA with output gate

The MLA projection layout is DeepSeek-V3 style:

```text
q_a_proj: 7168 -> 1536
q_a_layernorm
q_b_proj: 1536 -> 96 * (128 + 64)
kv_a_proj_with_mqa: 7168 -> 512 + 64
kv_a_layernorm
kv_b_proj: 512 -> 96 * (128 + 128)
g_proj: 7168 -> 96 * 128
o_proj: 96 * 128 -> 7168
```

The 64-dimensional fields retain the checkpoint layout normally called the
RoPE dimension, but K3 does **not** apply RoPE. They are concatenated directly
into q and k. Existing Megatron MLA cannot be selected unchanged because it
always rotates these fields. K3 also multiplies the attention output by
`sigmoid(g_proj(input))` before `o_proj`.

## Feed-forward variants

### Dense layer

Only zero-based layer 0 is dense. It is a gated 7168 -> 2*33792 -> 7168 MLP.
SiTU is applied in fp32:

```text
gate = 4 * tanh(gate / 4) * sigmoid(gate)
up   = 25 * tanh(up / 25)
out  = gate * up
```

### Latent MoE

Layers 1..92 are MoE layers:

1. Router logits are computed from the original 7168-wide hidden state.
2. Routing uses sigmoid scores, frozen `e_score_correction_bias`, top-16 of
   896 experts, and renormalizes selected scores.
3. Routed input is projected 7168 -> 3584 before dispatch.
4. Routed experts apply 3584 -> 2*3072 -> 3584 SiTU MLPs.
5. Routed outputs are combined in latent space.
6. Combined routed output is RMS-normalized in 3584 space.
7. Routed output is projected 3584 -> 7168.
8. Two shared experts run on the original 7168-wide input and are added after
   the routed up projection.

The ordering of steps 5-7 is semantically important. Any TP/EP partial results
must be complete before the nonlinear latent RMSNorm.

## Existing Megatron foundations

The miles fork already contains the relevant upstream foundations:

- NVIDIA Megatron PR 1989, commit `20d66d5c7`: GDN module and heterogeneous
  attention-layer specs.
- NVIDIA Megatron PR 2645, commit `2d1fa8d37`: packed-sequence GDN support.
- NVIDIA Megatron PR 2296, commit `b51db3e07`: latent MoE projections and
  latent-space dispatch/combine.

The miles fork additionally has Qwen3.5/Qwen3-Next FLA wrappers and Kimi-K2.5
MLA/MoE conversion patterns. These foundations avoid changes to token
dispatch, expert parallelism, packed-sequence plumbing, and the generic GPT
model.

## Implemented parallelism

The backend uses PP=1/CP=1 and keeps K3-specific modules under
`miles_plugins/models/kimi_k3`. Both attention variants use head-wise TP:

- KDA q/k/v, forget, beta, and output-gate projections are column-sharded;
  the short-convolution weights, `A_log`, and `dt_bias` are sharded by head;
  and the output projection is row-parallel.
- MLA q and kv expansion plus output-gate projections are column-sharded and
  the output projection is row-parallel. The shared 64-wide key field is
  replicated because every local head consumes it.
- TE column/row/duplicated linear modules are used consistently with Megatron.
- Sequence parallel input is gathered before the K3 attention recurrence and
  its output is scattered afterward. Dense MLPs and latent MoE use Megatron's
  existing TP/EP/SP implementations.
- Per-head output norms and attention-residual score parameters remain
  replicated. Their gradients are summed across TP ranks because each rank
  contributes a disjoint head/output shard to the shared parameter gradient.

Replicating K3 attention is not a viable production option: it multiplies the
96-head attention state and compute by TP size and does not fit the intended
8-GPU training configuration. The head-wise implementation is therefore the
default, not an optional follow-up optimization.

The implemented modules are:

1. `KimiK3Attention`: selects packed-prefill KDA or NoPE MLA by global layer
   index.
2. `KimiK3TransformerLayer`: implements the two attention-residual
   aggregations and carries the snapshot bank in `context`.
3. `KimiK3MoELayer`: extends Megatron latent MoE with latent RMSNorm at the
   post-combine/pre-up-projection point.
4. K3 model spec: starts from the TE GPT decoder spec, replaces attention,
   TransformerLayer, and MoELayer modules, and installs the 1-based attention
   pattern.
5. SiTU support: use an unfused GLU path. Both the gate and linear
   branch must use the exact soft-tanh formulas; hard clipping is not a valid
   substitute.
6. MBridge mapping for all attention-residual, KDA, MLA, latent projection,
   latent norm, shared-expert, router-bias, and expert tensors.

## Validation contract

The first numerical target is a four-layer checkpoint and identical token IDs.

1. Run SGLang prefill and save per-token final log probabilities.
2. Run Megatron with `miles.utils.debug_utils.run_megatron` and the same token
   IDs.
3. Require full-model average absolute log-probability difference below 0.1.
4. Also compare layer-truncated variants; each must remain on the same order.
5. When final logits fail, use Dumper on both paths with these initial points:
   embedding output, each attention-residual aggregate, attention output,
   router logits/top-k, latent down output, latent combined output, latent norm,
   latent up output, layer output, final aggregate, final norm, and logits.

The default activation dims are `t h` for SGLang and `t 1 h` for packed
Megatron. Attention-head tensors use `t num_heads[tp] head_dim`; TP partial
outputs use `[tp:partial]`. Dims errors must be corrected with comparator
`--override-dims` rather than rerunning inference.

The four-layer real-checkpoint comparison achieved mean absolute final
log-probability difference `0.04797336033412388` between SGLang and the TP
Megatron backend (maximum `0.261236`, p95 `0.112028`). A replicated-attention
reference and the TP implementation differed by mean absolute `0.0307079`.
Random-input KDA backward and real-checkpoint TP8+EP8+SP full-model backward
also completed successfully.

## Checkpoint preparation

The gated HF repository is accessible with the provided read token. Large files
must be downloaded directly to cluster shared storage under a neutral path such
as `checkpoints/yueming-model-support`; do not stage them on the laptop or use
the model codename in shared non-personal paths.

The published checkpoint is MXFP4 for routed experts. Validation uses:

- A four-layer native MXFP4 HF checkpoint for one-H200 SGLang inference.
- A four-layer BF16 HF checkpoint for bridge conversion and numerical debugging.
- A TP8 Megatron distributed checkpoint produced through MBridge after BF16
  dequantization. KDA short-convolution weights, `A_log`, and `dt_bias` remain
  fp32 during conversion and checkpoint load; rounding them through bf16 causes
  a measurable recurrence error.

The four-layer prefix covers KDA+dense, KDA+MoE, and MLA+MoE. A separate small
synthetic or selectively remapped test is required for the layer-12 snapshot
transition if a 13-layer checkpoint is too large for the debug allocation.

## Miles RL integration

`scripts/run_kimi_k3.py` provides data preparation and an 8-H200 training
launcher. The smoke configuration uses TP8, EP8, sequence parallelism,
colocated SGLang rollout, and strict
`--check-weight-update-equal`. K3's fused SGLang q/kv-A loader requires q-A and
kv-A tensors to be sent as one atomic update group. The SGLang weight-update
transaction restores packed model tensors at `begin`, accepts the complete
version as grouped updates, and runs loader post-processing only at `end`.
Transfer failures propagate and abort the job instead of leaving a silently
partially updated model.

`debug_minimal` uses distributed SGD with zero momentum and zero weight decay so
the one-node smoke exercises forward, backward, optimizer step, sleep/wake,
weight synchronization, and rollout without optimizer state for the
95B-parameter four-layer model. It uses the deterministic-random reward model
so the smoke always covers mixed rewards and nonzero policy gradients. This
reward is a deterministic hash of tokens and response text; it is independent
of answer correctness and must not be interpreted as a model-quality metric.
Zero-momentum SGD uses `torch.optim.SGD` because the installed TE `FusedSGD`
allocates a full momentum buffer even when momentum is zero. `normal` uses Adam
with fp32 moments and CPU optimizer offload. This distinction is necessary
because `torch_memory_saver.pause()` CPU-backs both CPU-resident and
GPU-resident optimizer allocations: moving fp32 state between the hybrid
optimizer's CPU and GPU sub-optimizers does not reduce the paused process's
total host-memory footprint. The Adam path completed a full optimizer step
during validation, but its subsequent paused state exceeds the 8-H200 node's
Ray memory limit.

The equality checker omits only multimodal tensors absent from the language
training model (`vision_tower.` and `mm_projector.`). Its initial snapshot is
cleared immediately after the first successful comparison so eight full
inference-model copies are not retained in host memory during Adam state
creation.

The final four-layer infrastructure smoke completed two rollouts and two valid
Megatron training steps on one 8-H200 node. Both steps completed log-prob
evaluation, backward, optimizer update, sleep/wake, and Megatron-to-SGLang
weight transfer; the final Ray job exited successfully. The deterministic-hash
reward averages were `0.4375` and `0.6875`, producing gradient norms `5.3781`
and `5.5038`. These synthetic values validate the backward and update paths,
not generation quality. As expected for the pruned model's nonsensical output,
a separate run with the real DeepScaler verifier produced all-zero rewards.
The second synthetic-reward rollout observed `weight_version=2`, and the final
post-training transfer updated all 90 tensor groups. The initial strict weight
comparison reported every included language-model tensor equal after the
rollout model was randomized, so the check validates the mapping and transfer
rather than only same-value assignment.
