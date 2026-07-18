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

### LoRA training

`scripts/run_kimi_k3_lora.py` is the verified LoRA launcher. K3 uses
`--megatron-to-hf-mode raw`: the native K3 model provider inserts adapters
before Megatron wraps the model in DDP, freezes every base parameter, and then
loads the base DCP checkpoint without requiring adapter entries in that
checkpoint. Megatron Bridge is not in the online model, forward, backward, or
weight-update path. MBridge remains the offline HF-to-DCP conversion tool.

The initial target set covers the modules whose Megatron and SGLang LoRA
contracts agree without changing a fused K3 operator:

- KDA and MLA output projection (`self_attention.o_proj`)
- MLA q/kv low-rank input projections (`q_a_proj` and
  `kv_a_proj_with_mqa`)
- dense MLP gate/up and down projections
- latent routed-expert gate/up and down projections

KDA's q/k/v, convolution, decay, and gate projections are intentionally not
targeted because SGLang packs them into K3-specific fused serving modules.
Shared-expert projections are also excluded from this first target set because
the serving path may fuse the shared expert into the routed MoE. The launcher
disables shared-expert fusion and `SGLANG_K3_FUSE_MOE_FRONT` so the supported
module boundaries stay explicit.

Routed experts use Megatron's shared-outer adapter layout: gate/up A and down B
are shared across experts, while gate/up B and down A remain per expert. K3's
routed experts operate on the 3584-dimensional latent state, whereas shared
experts operate on the 7168-dimensional transformer hidden state. SGLang must
therefore size routed `gate_up_proj_moe` and `down_proj_moe` buffers from
`routed_expert_hidden_size`, not `hidden_size`, and must initialize the A and B
per-expert dictionaries independently when only one factor is shared.

Several adapter factors consume a TP- or EP-partial activation. Those
parameters are marked with their required reduction domain, and Miles sums
their gradients over that domain before normal Megatron gradient finalization.
This keeps the implementation in the native K3 modules while preserving the
same replicated-factor semantics that a parallel LoRA implementation requires.
MLA's duplicated `q_a_proj` and `kv_a_proj_with_mqa` deltas are added directly
to the duplicated projection output. Their latent slices feed TE
column-parallel q/kv expansion, whose backward already sums input gradients
across TP; the key-extra slice passes through the attention implementation's
copy-to-TP mapping. Adding another copy-to-TP around the adapter delta would
double-reduce its gradient.

The raw exporter gathers TP- and EP-sharded factors into one HF-named chunk per
adapter module. A MoE layer has separate attention, routed-expert, and
shared-expert chunks. In colocate mode it materializes each parameter from the
torch memory-saver CPU backup, transfers only that adapter, and releases it
before the next one. SGLang accumulates the chunks and normalizes the adapter
only after the final chunk. A full 896-expert adapter is therefore not
duplicated as one long-lived GPU or CPU object per trainer rank.

The immutable rollout base is either retained or reloaded from node-local NVMe,
so adapter updates do not resend it. `--check-lora-weight-equal` compares every
rank's full pre-slice adapter payload with SHA-256 before SGLang performs TP/EP
slicing. A newly initialized K3 adapter also requires every exported `lora_B`
tensor to be exactly zero. The separate base reload checker verifies that the
native checkpoint reload restores all language-model tensors before rollout.

The verified topology is TP8, EP8, sequence parallel, and PP1 on one 8-H200
node. The base projections remain Megatron/TE modules; the adapter deltas use
PyTorch linear/grouped-matrix operations plus the explicit TP/EP reductions
described above. This is already a parallel implementation rather than a
single-rank reference path, so a second TP-specific backend is not needed. PP
greater than one remains unsupported by the underlying K3 backend.

The final four-layer LoRA smoke completed two rollouts and two valid training
steps. The deterministic-random raw rewards were `0.4375` and `0.5625`, and the
gradient norms were `0.4762578444` and `0.4851193323`. The generated text was
nonsensical, as expected from a four-layer prefix; the rewards only force
nonzero gradients and are not quality measurements. Each TP rank saved 28
native adapter tensors. Between the two TP0 checkpoints, 15 of those tensors
changed, with an L2 delta of `1.7596e-7` at the smoke learning rate. Across the
three rollout updates, the log contains three adapter loads but only one base
transaction (89 base chunks), proving that the second and final updates were
adapter-only. The second update took 2.27 seconds and the final update took 2.3
seconds.

### Full-model GB300 validation status

The full-model development topology is fixed at 16 four-GPU GB300 nodes. The
Megatron actor uses TP32, EP64, ETP1, PP1, and CP1 across all 64 GPUs. The
colocated SGLang engine uses TP48 and EP16 on the first 48 GPUs; the remaining
16 slots are placeholders during rollout. This is time sharing, not 112
simultaneously resident model ranks. An eight-node run cannot preserve either
EP64 or TP48 and is not a full-training configuration.

Megatron loads a 5.56 TB TP32/EP32 distributed checkpoint from shared storage
and lets DCP reshard it to TP32/EP64. SGLang reloads the immutable native HF
checkpoint from node-local NVMe before rollout. The smoke uses rank-32,
alpha-64 LoRA, zero dropout, SGD without momentum, eight prompts with two
samples each, a 64-token response limit, and deterministic-random reward. The
reward exists only to exercise policy gradients and is not a quality metric.

Full-model run 1139 completed the initial adapter transfer and all 16 rollout
responses, then entered Megatron log-probability evaluation. No rank completed
that phase because the first EP64 all-to-all did not make progress. Standalone
jobs 1143, 1144, and 1146 proved that EP64 all-to-all works after every rank
warms the recreated process group with an all-reduce followed by a tiny
all-to-all. The actor now performs that warmup after wake-up.

Run 1147 stopped at the base reload equality check because the language-only
rollout engine was compared against `vision_tower` and `mm_projector`
parameters. After restricting the check to parameters owned by the language
engine, run 1152 passed the base reload check, initialized all 64 trainer ranks,
and reported 447,167,488 trainable LoRA parameters per rank (0.888427%). It
transferred 186 adapter chunks as weight version 1 (93 attention, one dense
MLP, and 92 routed-expert chunks) and completed 16 rollout responses in 75.66
seconds. The responses became corrupted after an initially coherent prefix,
so completion of the rollout RPCs is not an accuracy result.

All ranks in run 1152 eventually completed the EP64 wake-up warmup and entered
`compute_log_prob`. Slurm terminated the job seven seconds after the last rank
entered that phase. Ray reported `ActorUnavailableError: Socket closed` only
after the Slurm termination, so that exception is a consequence of external
job cancellation rather than evidence of a model or Ray failure. There was no
completed log-probability phase, backward pass, optimizer step, gradient norm,
or weight version 2.

The full launcher currently enables `--check-lora-weight-equal` and
`--check-rollout-weight-reload-equal`; it does not enable the generic
`--check-weight-update-equal`. Adapter payload checks compare every TP rank's
full pre-slice tensor dictionary against trainer rank 0 with SHA-256. For a new
K3 adapter, version 1 additionally requires every exported `lora_B` tensor to
be exactly zero before the payload is sent. This distinguishes a nonzero
exported delta from a serving-side TP48/EP16 problem.

Standalone SGLang jobs 1155, 1156, and 1157 used eight nodes with TP32/EP32.
They showed coherent output for the base model, an enabled LoRA backend without
an active adapter, and an active adapter with both A and B zero. They did not
exercise the full TP48/EP16 topology or the real initialization in which A is
random and B is zero, so they do not replace the 16-node validation.

Job 1158 ran only unit tests in the production container and did not load a
checkpoint or start training. Miles passed 36 targeted LoRA synchronization
tests. SGLang passed seven tests, including a regression that models TP48/EP16
as MoE TP3 plus EP-local expert remapping and verifies that random A with zero B
leaves every runtime B buffer exactly zero. Full CUDA forward correctness,
backward, gradient norm, optimizer update, and weight version 2 remain pending
the next 16-node run.

Job 1159 used the same one-GPU unit-test configuration during the MLA backward
audit. Its helper-level q/kv tests were later superseded by the distributed TE
gradient comparison below; it did not load a checkpoint or start training.

The subsequent LoRA module audit found that the native K3 path wrapped routed
experts but not each MoE layer's shared expert. The shared expert now uses the
same sequence-parallel/TP-aware dense-MLP adapter as the first dense layer and
exports under `block_sparse_moe.shared_experts`. The versioned adapter sender
also records the complete version-1 tensor-name/checksum set; version 2 must
contain exactly the same names and at least one exported BF16 tensor must have
changed. This is a LoRA-specific update check and does not trigger a redundant
full-base-model synchronization.

The old `447,167,488` trainable count decomposes exactly into attention, the
layer-zero dense MLP, and 92 routed-expert adapters. Shared experts add
`43,900,928` parameters per TP32/EP64 rank, so the corrected full-model run
must report `491,068,416` trainable parameters per rank before rollout begins.
The corrected exporter must send 278 chunks per version: the original 186 plus
one shared-expert chunk for each of the 92 MoE layers. SGLang strict LoRA
loading is enabled so an unmatched name aborts the run instead of being
silently skipped.

Job 1160 ran the production-container unit suite for those changes. Miles
passed 47 tests, including shared-expert injection/export shape checks and the
version-1-to-version-2 change validator; SGLang passed seven tests. Job 1163
then extended the SGLang regression to consume the raw K3 routed and shared
expert names together. It models TP48/EP16 rank slicing and verifies that the
routed weights populate the MoE buffers, shared-expert weights populate the
standard TP buffers, both A paths are nonzero, and every B buffer remains
exactly zero. Job 1163 completed with the same 47 Miles and seven SGLang tests
passing. Neither unit job loaded a checkpoint or started training.

Job 1165 compared a two-rank TE column-parallel graph against a full-weight
reference. The direct duplicated delta matched the reference A/B gradients to
`1.53e-5`; wrapping that delta in another copy-to-TP produced gradients exactly
two times too large. The native attention adapter therefore relies on the
existing downstream TE/copy mappings and does not add its own TP reduction.
Job 1166 reran the production-container suite after that correction; all 45
remaining Miles tests and all seven SGLang tests passed.

Job 1170 extended the TP48/EP16 SGLang layout regression to the MLA attention
path. It normalizes separate raw `q_a_proj` and `kv_a_proj_with_mqa` adapters
into the replicated fused projection, slices `o_proj` A on outer TP rank 47,
and checks those buffers together with routed and shared experts. Every loaded
A path is nonzero and every B buffer remains exactly zero. The native exporter
now derives the expected adapter `(kind, HF prefix)` multiset from the actual
transformer layers and rejects a missing attention, dense, routed, or shared
adapter before sending any chunk; the 93-layer model therefore requires all
278 adapter chunks by construction. Job 1170 used no GPU and loaded no
checkpoint; 46 Miles tests and seven SGLang tests passed in the production
container.
