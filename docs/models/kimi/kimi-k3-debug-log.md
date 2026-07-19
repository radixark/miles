# Kimi K3 Debug Log

Keep entries short. Record the symptom, proven root cause, fix, and verification.

## SGLang JIT cache race

- Symptom: full native-MXFP4 startup hit `SIGBUS` during Marlin decode graph capture.
- Root cause: TP ranks concurrently relinked and truncated the same TVM-FFI `.so`.
- Fix: serialize JIT cache builds with a per-cache lock.
- Verification: job `1173` completed model load, graph capture, and deterministic generation; the cache contained one compile and one link.

## Derived tensors stale after reload

- Symptom: same-checkpoint reload changed prefill logprobs by `1.02246` on average.
- Root cause: K3 attention-residual combined weights and MLA `w_kc`/`w_vc` were derived tensors, not parameters, and were not refreshed after loading.
- Fix: recompute them after load and copy in place to preserve CUDA graph addresses.
- Verification: base reload average logprob diff fell to about `0.012`; LoRA reload mean diff was `0.0138`.

## Missing checksum actions

- Symptom: 16-node job `1231` stopped before Megatron load with `Unsupported action='snapshot_checksum'`.
- Root cause: the K3 SGLang branch lacked the checksum snapshot/compare actions used by Miles reload validation.
- Fix: port the existing SGLang weight-checker actions and expose per-rank and per-engine checksums.
- Verification: job `1232` completed checksum snapshot, full SGLang load, shared DCP load, offload, and native weight reload.

## K3 multimodal module mismatch

- Symptom: job `1232` reported reload checksum changes for `vision_tower.patch_embed.proj.bias` and `mm_projector.pre_norm.weight`.
- Root cause: SGLang reused K2.5 modules, but native K3 has no patch projection bias and uses two bias-free projector linears followed by RMSNorm.
- Fix: honor `patch_embed_proj_bias` and add the K3 projector matching native checkpoint names and structure in SGLang commit `770e96f37b`.
- Verification: job `1238` preserved all 382 registered tensor checksums/rank; base and LoRA fixed-prefill mean logprob diffs were `0.01765` and `0.00616`.

## Incomplete node-local 4-layer cache

- Symptom: gate job `1233` failed before model load because `c002` lacked `native-4layer-debug`.
- Root cause: the diagnostic launcher requires the pruned checkpoint on both TP8 nodes, but it had only been prepared on `c001`.
- Fix: build the same native-MXFP4 4-layer checkpoint from `c002`'s full local NVMe cache.
- Verification: job `1235` completed; both nodes have 55 GB, 24-file checkpoints with identical config and index hashes.

## Greedy reload comparison is unstable

- Symptom: job `1237` preserved all weight checksums but greedy generation diverged after token 61.
- Root cause: the pruned model emits low-quality, near-tied logits; one argmax change makes later generated tokens incomparable.
- Fix: compare logprobs on a fixed token sequence instead of requiring identical generated text.
- Verification: job `1238` passed base and LoRA fixed-prefill comparison with mean diffs below `0.1`.

## Full-model KV resume OOM

- Symptom: job `1239` passed full reload checksum and initial LoRA sync, then `onload_kv` failed in `torch_memory_saver.resume` with `CUDA_ERROR_OUT_OF_MEMORY`.
- Root cause: retained update-only process groups raised sleeping trainer usage from `3.27 GB` to `7.32 GB`; after that leak was fixed, job `1240` exposed a race where SGLang resumed before all 64 ranks completed cleanup.
- Fix: destroy update-only process groups, wait for every actor update to return, then resume generation from the group coordinator; `wake_up()` recreates and warms the groups before training.
- Verification: job `1241` completed all 64 updates before both engines resumed and generated 16 full-model samples without KV OOM.

## LoRA CUDA IPC cleanup

- Symptom: job `1241` generated successfully, then ranks 4-7 and 12-15 aborted during the first training CUDA allocation with `could not unlink the shared memory file /torch_*`.
- Root cause: the chunked K3 LoRA path omitted the upstream producer `torch.cuda.ipc_collect()`; 278 chunks of CUDA IPC state remained until trainer wake-up.
- Fix: release the final chunk references, collect producer IPC state, and empty the allocator cache before process-group teardown.
- Verification: pending full-model rerun.
