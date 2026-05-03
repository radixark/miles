# Temporary SGLang patches

These patches are required to run the `random_async` example with
disaggregated SGLang. They are temporary workarounds.

| Patch | Targets | Why |
|---|---|---|
| `00_tbo_forward_delegate.patch` | `layers/attention/tbo_backend.py` | TboAttnBackend is missing a generic `forward()` method; delegate to `self.primary.forward(...)`. |
| `01_pause_generation_send_pyobj_sync.patch` | `managers/tokenizer_manager.py` | Make pause-generation sends synchronous so the engine reaches a quiesced state before weight transfer. |
| `04_mamba_hicache_kernel_indices_cuda.patch` | `mem_cache/memory_pool_host.py` | Move Mamba state indices to CUDA before kernel execution. |
| `13_hicache_load_record_stream_early.patch` | `managers/cache_controller.py` | Record index tensor stream associations early so the allocator doesn't recycle them mid-load. |
| `14_minimal_pd_pause_gate.patch` | `disaggregation/decode.py` | Gate prefill/decode scheduling during pause so weight updates don't race in-flight requests. |

## Apply

```bash
./apply_patches.sh
```

Auto-detects the installed SGLang location via
`python -c 'import sglang; ...'` and applies each `.patch` with
`patch -p2`. Fails loudly on any error — re-running on an already-patched
tree will fail. Override the target with
`SGLANG_PARENT=/path/to/sitepackages`.

## Revert

```bash
for p in *.patch; do patch -p2 -R -d "${SGLANG_PARENT}" < "$p"; done
```
