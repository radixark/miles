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

- Symptom: jobs `1241` and `1243` generated successfully, then trainer ranks aborted on a later CUDA allocation with `could not unlink the shared memory file /torch_*`.
- Root cause: the chunked LoRA path lacked the paired SGLang receiver cleanup; producer-only collection in `1243` ran before all receiver IPC references were reaped.
- Fix: release and collect receiver tensors after every chunk, collect completed producer chunks, and collect once more after the updater frame returns.
- Verification: job `1244` passed the focused producer/receiver tests. Full-model job `1245` completed rollout, trainer wake-up, log-probability evaluation, backward, save, sleep, and a second 278-chunk adapter transfer without an IPC unlink error.

## Full-model LoRA update is zero

- Symptom: job `1245` reported `loss=0`, `grad_norm=0`, and no change in any of 1,392 exported LoRA tensors after the first optimizer step.
- Isolation: job `1253` loaded the shared DCP in `297.83s`, started both TP8 rollout engines, produced 512 active tokens and three nonzero advantages per effective DP rank, and kept the rollout/Megatron mean log-probability difference at `0.00774`. Backward then found all 77,184 aggregated LoRA `main_grad` tensors exactly zero.
- Root cause: job `1261` proved the training logits had no autograd graph. Native LoRA freezes the embedding, so full reentrant activation recompute received no grad-enabled tensor input and detached every adapter computation inside the checkpoint.
- Fix: require gradients on the frozen embedding output during K3 LoRA training when full recompute is enabled.
- Verification: job `1265` completed two full-model steps. Step 0 updated all LoRA B tensors with `grad_norm=0.340486`; step 1 had nonzero A and B gradients with `grad_norm=0.664772`. Both steps updated all 64 ranks, and rollout/training log-probability mean absolute differences were `0.034334` and `0.028430`.

## LoRA checkpoint barrier after offload

- Symptom: job `1263` wrote all 64 adapter and optimizer shards, then all `save_model` RPCs remained in the final checkpoint barrier.
- Root cause: the CPU/shared-disk LoRA checkpoint path used the reloadable default NCCL group for coordination after colocated training offload.
- Fix: use the persistent Gloo group for both LoRA checkpoint barriers.
- Verification: job `1265` completed both 64-rank saves, then sent adapter versions 2 and 3 to both TP8 rollout engines. The job completed with exit code 0.

## LoRA A update visibility

- Observation: job `1265` changed `696/1392` exported tensors after step 0 and `714/1392` relative to initialization after step 1.
- Explanation: standard zero-B initialization makes the first A gradient zero. At step 1 all A gradients were nonzero, but most `1e-6` SGD updates to randomly initialized A values remained below one BF16 rounding interval; 18 A tensors and all 696 B tensors changed exported bytes.
- Verification: the validator hashes each exported BF16 tensor against version 1; both SGLang engines accepted and checksum-verified adapter versions 2 and 3.

## High-cardinality LoRA CUDA IPC

- Symptom: full GSM8K job `1283` transferred version 1 and completed a 64-sample rollout, then trainer IPC producer ranks aborted during the first log-probability allocation with `could not unlink the shared memory file /torch_*`.
- Root cause: the sender exported 2,828 tensors as separate CUDA IPC allocations. PyTorch reported more than 1,000 outstanding producer blocks; delayed allocator collection after the 10-minute EP warm-up hit the stale refcounter file.
- Fix: flatten each existing LoRA chunk into one `FlattenedTensorBucket` payload. Tensor values, BF16 precision, checksums, chunk boundaries, and SGLang TP/EP slicing are unchanged; IPC allocation count falls from 2,828 to 278.
- Verification: job `1285` passed all 42 Miles sync tests and the SGLang CUDA-source load test. Job `1286` passed 278 flattened chunks plus 300 subsequent CUDA allocations without a limbo warning or unlink failure. Full-model verification is pending.

## GSM8K response-256 baseline

- Job `1283` used the approved 16-node TP32/EP64 BF16 trainer, two official TP8/EP1 native-MXFP4 rollout engines, GSM8K math reward, eight prompts with eight samples each, and a 256-token response limit.
- The first rollout completed 64 samples with raw reward `0.828125` (53/64), mean response length `179.45`, and six truncated responses. Two of eight prompt groups had nonzero reward variance, so GRPO advantages were nonzero.
- The job failed before Megatron log probabilities, backward, gradient norm, or optimizer update; it is generation evidence, not a training-quality result.
