# miles.kernels

Every hand-written training kernel in miles, filed by the op it computes.

## Layout

```
miles/kernels/
├── attention/
│   ├── dsa/          sparse attention: indexer + sparse MLA (TileLang), top-k selection
│   │   ├── glm5/         thd layout, MLA, no sink (GLM-5.x, DeepSeek-V3.2)
│   │   └── deepseek_v4/  bshd layout, MQA, fp32 attention sink, batched
│   ├── delta_rule/   GDN / KDA kernel selection (fla, FlashQLA)
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
  Megatron import, no process group. TP/SP/CP live in the wrapper module that calls the kernel
  (`miles_plugins/models/`), which does TP through Column/RowParallel projections, SP through
  gather/scatter at the kernel boundary, and CP through an all-gather of KV or a `cp_context`.
- **Variants are parameters, not copies.** A layout (`thd` / `bshd`) or an optional feature
  (attention sink) is a kernel argument. `attention/dsa/glm5` and `attention/dsa/deepseek_v4`
  are the one remaining pair of copies and are scheduled to merge; until then the DeepSeek-V4
  plugin imports its indexer's TileLang module directly, and the DSA tests are manual scripts
  under `tests/manual/`.
- **A new or changed kernel comes with a single-GPU test against a torch reference**, under
  `tests/fast-gpu/kernels/<op>/`.

## Adding support for a new model

Ask first whether the model's ops already exist here.

- Another GDN / KDA model: nothing changes in this package; pick the fla or FlashQLA kernel
  through `attention/delta_rule/backend.py`.
- A DSA variant with a new layout or feature: add a parameter branch in `attention/dsa`, do
  not copy the kernel pair.
- A genuinely new op: add `miles/kernels/<op>/` with one entry function and a torch-reference
  test under `tests/fast-gpu/kernels/<op>/`. The model plugin imports the entry function only.
