# miles.kernels

```
activation/        swiglu_fp32 · short_conv_fp32
attention/
  dense/           dense_attention_backward
  dsa/             sparse_attention · lightning_indexer · indexer_logits[_sbhd] · indexer_topk_scores · kpool · topk
    tilelang/      one indexer + sparse attention pair; RoPE tail and attention sink are compile-time parameters
  qsa/             qsa_sparse_attention · qsa_block_sparse_attention
embedding/         gather_ple_rows · ple_gate_conv
hyper_connection/  hc: hc_mix_inject · hc_combine    mhc: mhc_mix · mhc_aggregate
moe/               fused_experts
norm/              grouped_rmsnorm (device helpers shared by hc and ple)
position/          rope: apply_rotary_emb
quant/             fp8_blockwise_cast · act_quant · fake_quant_{fp8,fp4,compressed_kv} · fused_nvfp4_qdq · int4_fake/ (CUDA)
```

- Only kernels we write (Triton, TileLang, CuTe, CUDA). A third-party kernel we merely call stays with its caller.
- One directory per op. Names say what is computed, never the backend; a backend gets a subdirectory only when two coexist.
- Callers import the entry from the op's module or package, never an implementation file.
- No Megatron, no process groups, no env policy. Parallelism lives in `miles_plugins/models`.
- Variants are parameters, not copies.
- Every op has a torch-reference test in `tests/fast-gpu/kernels/<area>/`.
