# Qwen3 dense true-on-policy: forward review

The Megatron TOP spec runs the dense Qwen forward with the same numerical
operations as SGLang. `--true-on-policy` on the Qwen launcher selects the versioned
`true_on_policy_v1` contract, the local Megatron spec, FA3, batch-invariant matmul,
and native unfused-bias SwiGLU. It uses decode-produced log probabilities; it does
not enable prefill rescoring.

This is a **draft for forward review**, not a merge-ready training qualification.
Autograd implementations accompany the delegated modules, but backward, optimizer
behavior, and learning are outside this review. MoE and GLM programs are not
registered or bound here. The high-level launcher supports Megatron only; raw
FSDP launch callers are updated for the versioned contract name.

## Manual review order

| Order | Code | What to check |
|---|---|---|
| 1 | `miles/true_on_policy/{schema,contracts,model_profiles,config}.py`; `program.py` | One dense program at every admitted TP degree; explicit Ulysses CP; sequence-parallel and unsupported sampling refusals; no implicit prefill replay. |
| 2 | `spec.py`; `residual_norm.py` | Preserve stock TransformerLayer forward. Carry `[branch, residual]` without an early BF16 addition. Clone both buffers before SGLang's mutating add+norm kernel. Require `2H` pipeline transport and a pair at every nonfirst residual site. |
| 3 | `ops.py`; `install.py`; `spec.py:TopRowParallelLinear` | Delegate row matmul and fixed-tree TP reduction; verify the SGLang contract rejects unsupported K dimensions before a fallback is reachable. Review forward methods in this pass. |
| 4 | `rope.py`; `install.py` | Enforce fused TE RoPE at construction and the actual attention call site; pass the real CP group size/rank for packed sequences. |
| 5 | `attention.py`; `cp_layout.py` | FA3 varlen with the fixed split policy; explicit process group; global packed lengths; restore zigzag token order around Ulysses collectives. |
| 6 | `miles/backends/training_utils/loss_hub/logit_processors.py`; `miles/rollout/generate_utils/prefill_logprobs.py` | Identity-only scoring contract; reject temperature/filter/mask changes. Optional prefill scoring sends temperature 1.0. LM-head producer changes belong to the SGLang prerequisite. |

Tests beside these changes cover launch admission, role binding, residual ownership,
fused RoPE, CP layout bookkeeping, and GPU kernel premises. Synthetic CP/KV tests
are not a claim that every topology has passed the full-model gate.

## Cross-repository prerequisites

The default dependency pins are not sufficient for this draft. Companion PRs and
final dependency pins remain to be prepared before merge.

- **Megatron:** validated source was `c6449f0b23be397449f21c0967c5fc90785e55ea`
  plus forward patches in `model_parallel_config.py`,
  `pipeline_parallel/{schedules,p2p_communication}.py`, and
  `transformer/{transformer_layer,transformer_config,dot_product_attention}.py`.
  These provide the `pipeline_hidden_size` transport width, PP communication
  ordering, residual lifetime through BDA, and CP admission at the selected
  attention implementation. File paths are relative to `megatron/core/`.
- **SGLang:** validated source was `774d7d2d878c58162404847a04dc88e5d85dfcf4`
  plus LM-head/logit-buffer and sampling patches. The LM-head change spans
  `layers/logits_processor.py`, `model_executor/graph_shared_output.py`, three
  `model_executor/runner/` files, two speculative draft-extend graph runners, and
  `true_on_policy/{config,__init__}.py`; sampling admission also changes
  `sampling/sampling_params.py`, `managers/tokenizer_manager.py`, and
  `entrypoints/http_server.py`. Paths are relative to `python/sglang/srt/`.
  The source pin also supplies the unconditional versioned numerical contract,
  stock RMSNorm dispatch, TP-invariant row operations, and deterministic reduction.
  Speculative producer coverage is a buffer-contract requirement, not a claim of
  speculative end-to-end qualification.

## Existing forward evidence and limits

A frozen development snapshot was tested on four H200s on 2026-09-07 with the
full 36-layer Qwen3-4B, BF16, frozen weights, identity sampling, CUDA graphs through
batch 8, and prefill chunk size 1,024. Training and rollout offloading were disabled.

| Trainer TP × CP × PP | Rollout TP | Unique response scores | Maximum absolute difference |
|---|---|---|---|
| 4 × 1 × 1 | 4 | 1,024 | 0 |
| 2 × 2 × 1 | 2, two engines | 1,024 | 0 |
| 1 × 2 × 2 | 4 | 1,024 | 0 |

Each run produced eight responses of 128 tokens. An audit compared original
HTTP-returned scores with trainer scores, checked complete per-sample CP ownership,
and avoided double-counting TP replicas. These are selected-token score results,
not full-vocabulary or all-hidden-state equality. Twenty live HTTP sampling
contract checks also passed on that snapshot.

**This extracted branch has not been GPU-revalidated.** The earlier snapshot
included changes outside this draft, so the table is supporting development
evidence rather than a test result for this PR. Before merge, land/pin the companion
changes, rerun the full-model forward matrix on the extracted revisions, and review
backward separately. CP4, other hardware, offloading, and speculative decoding are
not covered by this full-model matrix.

Local extraction checks: 25 configuration/CP tests passed on Python 3.12. Three
binding/RoPE test modules were skipped without Torch/Megatron; launcher tests could
not run without Ray. GPU and sampling integration suites have not run on this
branch. Python syntax and whitespace checks passed.
