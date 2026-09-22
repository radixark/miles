# miles.kernels

Every hand-written training kernel in miles, filed by the op it computes.

## Layout

```
miles/kernels/
├── attention/
│   ├── dsa/          DeepSeek Sparse Attention: lightning indexer, top-k, sparse attention
│   │   └── tilelang/     one kernel pair; RoPE tail (GLM-5, DSv3.2) and attention sink (DSv4) are
│   │                     compile-time parameters
│   ├── delta_rule/   GDN / KDA kernel selection (FLA, FlashQLA)
│   └── dense_bwd/    Triton backward paired with sglang's Triton forward
├── moe/              fused expert GEMMs with Triton backward
├── activation/       fp32 SwiGLU and residual short conv (Triton, sglang-bit-exact forward)
└── quant/            fp8 blockwise / fp8 activation / NVFP4 QDQ / MXFP8 / fake INT4 (CUDA)
```

## Rules

- **Filed by op, not by model or backend.** The backend (TileLang, Triton, CuTe, CUDA) is a
  filename prefix or a subdirectory inside the op, never a top-level axis. Adding a second
  backend for an op is a sibling file, not a new tree.
- **One entry function per op.** Callers import the plain Python entry point, never a
  `tilelang_*` or `*_kernels` module. Device kernels and `autograd.Function` classes are
  implementation detail.
- **Parallelism-agnostic.** A kernel sees `[local heads x local-or-gathered sequence]`. No
  Megatron import, no process group. TP/SP/CP live in the module that calls the kernel: shared
  parallel-aware modules under `miles_plugins/models/layers/` (delta-rule attention for GDN/KDA),
  model-specific ones under `miles_plugins/models/<model>/`. They shard heads across TP, gather and
  scatter for SP at the module boundary, and pass CP through an all-gather of KV or a `cp_context`.
- **Variants are parameters, not copies.** An optional feature (RoPE tail, attention sink) is a
  compile-time kernel parameter, so the specialised code paths cost nothing at runtime. Layouts
  are adapters around a packed kernel, never a second kernel: `indexer_logits_sbhd` runs the
  packed indexer once per batch element, and `sparse_attention` takes a batch dimension that
  packed callers set to 1.
- **Every kernel has a single-GPU test against a torch reference.**

## Adding support for a new model

Ask first whether the model's ops already exist here.

- Another GDN / KDA model: nothing changes in this package. Subclass
  `miles_plugins.models.layers.delta_rule_attention.DeltaRuleAttention`, declare the model's input
  projections under their HF names with `sharded_linear`, pick `GatedDeltaRule` or `KimiDeltaRule`,
  and wrap it in `LinearAttentionLayer` in the layer spec. Heads are sharded across TP for free.
- A DSA variant with a new layout or feature: add a parameter branch in `attention/dsa`, do
  not copy the kernel pair.
- A genuinely new op: add `miles/kernels/<op>/` with one entry function and a torch-reference
  test under `tests/fast-gpu/kernels/<op>/`. The model plugin imports the entry function only.
