# True On-Policy between Training and Inference

True on-policy (TOP) makes the trainer's log-probs bitwise equal to the rollout engine's, so the wandb metric
`train/train_rollout_logprob_abs_diff` is exactly `0`.

TOP runs on the Megatron backend. One flag turns it on:

```bash
--true-on-policy-mode
```

It selects the only contract, `true_on_policy_v1` (trainer and engines), and the TOP block spec
(`--spec miles_plugins.top.spec get_top_spec`). Spelling either out is allowed; any other value is refused.

A launch is admitted only when the op registry admits the model's family and its trainer and serving layouts. SGLang
owns the op and family list (`sglang.srt.true_on_policy.registry`); Miles declares each op's trainer cells
(`miles_plugins/top/registry.py`). Every refusal names the op and the axis.

## Admitted families

| Family | `model_type` | Trainer / serving layout | e2e gate |
|---|---|---|---|
| Qwen3 dense | `qwen3` | TP1, CP1, any PP / engine TP1, no speculation | `tests/e2e/precision/test_qwen3_0.6B_true_on_policy.py`: Qwen3-0.6B, trainer PP2, 2 GRPO steps, trainer and rollout log-probs equal, per-layer gradients against HF, bitwise stage handoff |
| Qwen GDN dense (Qwen3.5/3.6/3.8 dense) | `qwen3_5` | TP1, all-gather CP, any PP / engine TP1, chain EAGLE/MTP speculation (top-k 1) | the dense image gate (`test_qwen3_8_27B_4layer_vision_true_on_policy.py`, below) on image, video and text samples; its GDN and attention ops also run in the Qwen GDN MoE gate |
| Qwen GDN MoE (Qwen3.5/3.6 MoE) | `qwen3_5_moe` | TP1, EP>=2 dividing the experts, all-gather CP, any PP / engines with attention DP and EP>=2 over DeepEP, chain EAGLE/MTP speculation (top-k 1) | `tests/e2e/precision/test_qwen3_5_35B_A3B_4layer_true_on_policy.py` (nightly): Qwen3.5-35B-A3B 4-layer slice, an attention-DP4 x EP4 engine with radix reuse, 2 GRPO steps at trainer CP4 x EP4 and CP1 x EP4, trainer and rollout log-probs equal; CP4 backward against CP1 per captured tensor, and CP1 per-module gradients against HF's contract and per-layer gradients against HF FP32; then CP1 with forced retraction (exact log-probs and its own verdict) and CP4 agentic sessions (exact log-probs across turns, later turns hit their previous reply) |
| Qwen3.8-Flash-Next (BF16) | `qwen4_exp` | TP1, EP>=2 dividing the experts, CP1, PP1 / engines with attention DP and EP>=2 over DeepEP (DeepEP itself needs EP>=4 for the 512 experts); EAGLE speculation with the checkpoint's MTP layer as the draft, recipe (3,1,4) | `tests/e2e/precision/test_qwen3_8_flash_next_4layer_true_on_policy.py` (nightly): Qwen3.8-Flash-Next cut to 4 layers plus its MTP layer (three GDN layers, one QSA attention layer, hyper-connections, the PLE layer), an attention-DP4 x EP4 engine, 3,072-token responses (past QSA's 2,048-key budget, so its block top-k is selective), 2 GRPO steps at trainer CP1 x EP4, trainer and rollout log-probs equal; per-module gradients of the GDN, QSA attention, MoE and hyper-connection modules of layers 1-3 against transformers' `qwen4_exp` at TOP's routing and index selection (the contract) and per-layer gradients against it in FP32 (the PLE layer's parameters are reported there only); then the same rollouts with forced retraction (requests resume from the radix cache), and with retraction plus MTP speculation, each reproducing the step-1 gradient norm. The slice's draft is accepted almost never; accepted drafts are not covered by CI |
| GLM (GLM-5.x, BF16 or block-FP8) | `glm_moe_dsa` | TP1, EP>=2 dividing the experts, all-gather CP, PP1 (block-FP8: any PP) / engines with attention DP and EP>=2 over DeepEP, BF16 KV cache (block-FP8: FP8 E4M3); block-FP8 admits EAGLE speculation with the checkpoint's MTP layer as the draft, recipes (1,1,2) and (5,1,6), with no e2e gate | `tests/e2e/precision/test_glm5_2_5layer_true_on_policy.py` (nightly; block-FP8; BF16 has no CI job): GLM-5.2-FP8 5-layer slice, 3,072-token responses (sparse DSA selection), trainer CP2 x EP4 against an attention-DP4 x EP4 engine, 2 GRPO steps, trainer and rollout log-probs equal; then CP1 x EP4 with per-layer gradients against HF on the dequantized checkpoint at TOP's routing and index selection, and PP2 x EP2 with layers 0-1 and 2-4 on the two stages; the step-1 gradient norm equal across the three |
| DeepSeek-V4 (DeepSeek-V4-Flash, block-FP8 only) | `deepseek_v4` | TP1, EP>=2 dividing the experts, all-gather CP, any PP / engines with attention DP and EP>=2 over DeepEP, FP8 E4M3 KV cache, FP8 experts, engine PP1; EAGLE speculation with the checkpoint's MTP layer, recipes (1,1,2), (3,1,4) and (5,1,6) | `tests/e2e/precision/test_deepseek_v4_flash_4layer_true_on_policy.py` (nightly): DeepSeek-V4-Flash-FP8 4-layer slice, 3,072-token responses (sparse C4 selection), 2 GRPO steps against an attention-DP4 x EP4 engine at trainer CP1 x EP4, all-gather CP2 x EP4 and PP2 x EP2 (layers 0-2 and 3), trainer and rollout log-probs equal; CP2 and PP2 backward against CP1 per captured tensor, and CP1 per-module gradients against DeepSeek's inference Block (contract) and per-layer gradients against it in FP64 |
| DeepSeek-V4.1 (DeepSeek-V4.1-Flash release checkpoint, Hopper) | `deepseek_v41` | TP1, EP>=2 dividing the experts, CP1, PP1, 2 or 8, no activation recompute / engines with attention DP and EP>=2 over DeepEP, attention TP1, FP8 E4M3 KV cache, the release's 32x32 FP8 dense weights and MXFP4 experts (cast losslessly to 128x128 FP8 at load), radix reuse from the engine's own tree at whole 256-token pages (no HiCache, external caches, sessions or PD, and not with the vision tower loaded), eager prefill with chunks covering the context, images through the frozen tower (below); DSpark speculation with the checkpoint's three draft stages at block size 5, recipe (1,1,6), static verify | `tests/e2e/precision/test_deepseek_v4_1_flash_8layer_true_on_policy.py` (nightly): DeepSeek-V4.1-Flash 8-layer slice (layers 0-3, 20, 21, 24, 25; experts 0-63), 3,072-token responses, trainer CP1 x EP4 and PP2 x EP2 (the stage boundary after layer 2, then after layer 4) against an attention-DP4 x EP4 engine speculating with DSpark, 2 GRPO steps, trainer and rollout log-probs equal, per-layer gradients of layers 1, 2, 4 and 6 replayed in DeepSeek's inference code at TOP's routing, index selection and engram rows (reported: the V4.1 reference declares no contract modules yet; the capture checks are judged); then CP1 x EP4 on agentic sessions with forced retraction (later turns and retracted requests resume from radix hits: exact log-probs, its own verdict, later turns hit their previous reply). The slice's draft accepts almost nothing; DSpark's accepted drafts are not covered by CI |
| GLM-5.3-Flash (BF16, Hopper) | `glm5_next` | TP1, EP>=2 dividing the experts, CP1, PP1 / engines with attention DP and EP over DeepEP, BF16 KV cache, radix reuse on the mamba track (`extra_buffer`, 64-aligned pages and track interval) | `tests/e2e/precision/test_glm5_3_flash_4layer_true_on_policy.py` (nightly): GLM-5.3-Flash 4-layer slice (KDA, DSA, KDA, DSA; mHC hyper-connections), an attention-DP4 x EP4 engine with radix reuse, 3,072-token responses (crossing KDA chunk boundaries in decode), 2 GRPO steps at trainer EP4, trainer and rollout log-probs equal; then the same with the engine forcing retraction (requests resume from the radix cache), and agentic sessions with retraction (exact log-probs across turns, later turns hit their previous reply) |

No text gate of the Qwen GDN families speculates: on a sliced checkpoint the MTP draft is almost always rejected, so
accepted drafts are not covered by CI.

### Images and video (Qwen GDN dense and MoE, Qwen3.8-Flash-Next; DeepSeek-V4.1 images)

The Qwen GDN families and Qwen3.8-Flash-Next train samples with images and video, DeepSeek-V4.1 with images. The checkpoint's vision tower stays frozen: engines and trainer
encode an image with one forward program (`sglang.srt.true_on_policy.vision`: batch-invariant GEMMs with the bias
added in BF16, FP32 2D rotary tables through eager rotary, noncausal FA4 with one key split), so its rows do not
depend on which side encoded it or on the other images in the batch. The trainer writes the rows over the placeholder
embeddings (under all-gather CP, each rank the rows of the placeholders it holds), and full attention rotates by the
model's mRoPE positions on both sides. Qwen3.8-Flash-Next's QSA indexer rotates each lane at its mRoPE axis while
it counts key groups and blocks by token index, and its PLE layer hashes the id serving's language model reads at a
placeholder.

DeepSeek-V4.1 runs its ViT and 3x3 aligner on the same program (FP32-weight RMSNorm, the release's FP32 half-split
2D rotary) and writes each image's whole span (start, aligner rows, newlines, end) over the span's tokens. Every row of
an engine built with the vision tower routes through the VL-biased gate, image rows by the VL bias, and Engram leaves
image rows' streams as they came and stops its n-grams at them; the trainer does the same on the first pipeline stage.

Launch with `--multimodal-keys`, `--multimodal-token-space-processing` and `--sglang-enable-token-space-processor`:
each trainer rank rebuilds a sample's pixels, placeholder placement and positions with SGLang's token-space processor,
on the CPU as the engines preprocess under the contract. A TOP launch with `--multimodal-keys` and without them is
refused.

| Gate | What it runs |
|---|---|
| `tests/e2e/precision/test_qwen3_8_27B_4layer_vision_true_on_policy.py` (nightly) | Qwen3.8-27B 4-layer slice with its 27-layer tower, a 1-GPU engine, prompts with one or two drawn images (36 to 1,496 placeholder tokens) and one text-only prompt, 2 GRPO steps at trainer CP1, all-gather CP2, PP2 and PP4 (one layer per stage): trainer and rollout log-probs equal, the phases' step-1 gradient norms agree |
| `tests/e2e/precision/test_qwen3_5_35B_A3B_4layer_vision_true_on_policy.py` (nightly) | Qwen3.5-35B-A3B 4-layer slice, an attention-DP4 x EP4 engine, the same prompts, 2 GRPO steps at trainer CP1 x EP4, CP4 x EP4 and PP2 x EP2, and at CP1 with the engine speculating through the checkpoint's MTP layer (recipe (3, 1, 4)): trainer and rollout log-probs equal |
| `tests/e2e/precision/test_qwen3_8_flash_next_4layer_vision_true_on_policy.py` (nightly) | Qwen3.8-Flash-Next 4-layer slice with its MTP layer, an attention-DP4 x EP4 engine, prompts that open with 2,264 image tokens (past QSA's 2,048-key budget) and one text-only prompt, 2 GRPO steps at trainer CP1 x EP4, plain and with the engine retracting and speculating (EAGLE (3, 1, 4)): trainer and rollout log-probs equal |
| `tests/e2e/precision/test_deepseek_v4_1_flash_8layer_vision_true_on_policy.py` (nightly) | The DeepSeek-V4.1 text gate's 8-layer slice with the release's tower, aligner and VL routing biases added back, an attention-DP4 x EP4 engine over DeepEP, the drawn-image prompts (184 to 912 span tokens) and one text-only prompt, 2 GRPO steps at trainer EP4 and PP2 x EP2, and at EP4 with the engine speculating through DSpark (recipe (1, 1, 6)): trainer and rollout log-probs equal, EP4 and PP2 step-1 gradient norms agree |
| `tests/fast-gpu/test_top_dsv41_vision.py` | The DeepSeek-V4.1 trainer tower against serving's ViT, aligner and span over the same tensors, bit for bit, and against FP64; the trainer's VL-biased gate against serving's; Engram's gate and hash on image rows against the text program |
| `tests/fast-gpu/test_top_vision.py` | The trainer's tower against serving's tower class over the same parameters, bit for bit, alone and in batches; the mRoPE rotary reference against the served rotation and its gradient against FP64; QSA's rotary kernels lane by lane against the text program |

The image gates judge no backward of their own: the tower has none, and the text model's backward verdicts are the
text gates'. Each image gate also has a `video` phase on drawn clips (one or two clips per prompt, a drawing and a
clip together, and a text-only prompt), judged by its log-probs. The slices' MTP drafts are accepted almost never, so
the speculating phases check verify on image requests, not accepted drafts. Audio is refused at launch: TOP has no
program for an audio tower.

Qwen3 MoE (`qwen3_moe`) is refused at every layout: no e2e gate runs it. Its op (`qwen3_moe_experts`) stays in the
registry, unadmitted.

Under speculative decoding, verify draws each draft row's target with the TOP sampler's race, keyed by the request's
seed and the row's position, and accepts drafts while they equal it: every emitted token is the one non-speculative
decode would draw, whatever the draft proposes, and its log-prob is scored from the target's logits as TOP decode scores
it. The weight sync leaves the draft out (unless `--enable-mtp-training`), so a stale draft lowers acceptance, not
correctness.

`miles.true_on_policy.launch.top_launch` returns the flags and environment a TOP run needs (Megatron's local spec, eager
SwiGLU, deterministic serving); `scripts/run_qwen3_4b.py --true-on-policy` launches Qwen3 dense with them.
[`recipes/`](recipes) has full-scale launchers for DeepSeek-V4-Flash and GLM-5.2 math RL, each with its TOP arm and a
standard + R3 + TIS arm, and for Qwen3.6-35B-A3B async text-to-SQL with five TOP and standard arms.

## Debug and tuning knobs

Environment variables read by the TOP trainer and its backward gates. None of them changes the forward's bytes.

| Variable | Default | Effect |
| --- | --- | --- |
| `TOP_MOE_CHUNKS` | `1` | Runs each micro-batch's TOP MoE layer in this many token chunks, bounding the rows one hot EP rank receives; every kernel on the path is row-wise batch-invariant, so the bytes do not change (the DeepSeek-V4 and GLM-5.2 gates set 2) |
| `TOP_TRANSPORT_CHECKS` | `1` | `0` turns off the slot-order DeepEP transport identity checks (annotated slot rows, device-side receive and return receipts); the per-step exactness audit stands in for them |
| `TOP_SURROGATE_CHECK` | unset | `1` keeps each surrogate's forward value and, in backward, fails if the reference differs from the kernel output by more than `TOP_SURROGATE_CHECK_TOL`; row surrogates then recompute the reference instead of using their closed-form backward |
| `TOP_SURROGATE_CHECK_TOL` | `0.05` | Relative tolerance of `TOP_SURROGATE_CHECK` |
| `MILES_TOP_VJP_SERIAL` | unset | `1` runs the backward gate's reference replays in turn in the gate process instead of one process per visible GPU |
| `MILES_TOP_VJP_REPLAYS` | unset | A path where the backward gate saves its reference replays, or loads them when the file exists, so a capture can be judged again without the GPU |
| `MILES_TOP_GATE_REPORT_ONLY` | unset | `1` logs the backward gate's layout and VJP failures instead of failing the gate, while thresholds are measured |

## Not yet verified

TOP admits some paths that no gate runs. Nothing is known to break on them, and nothing has verified them either.
`miles/true_on_policy/unverified.py` owns the list a launch can detect and logs a `[top]` warning naming each one it
uses. These can only be detected at run time:

- a microbatch in which some local expert receives no tokens (short agentic samples, small microbatches, many experts);
  the expert backward then relies on the grouped GEMM writing zeros for an empty group;
- a prompt split across chunked-prefill chunks (the gates' prompts fit one chunk);
- agentic chat requests that carry the model's generation defaults (for example Qwen3's top-k 20 and top-p 0.95):
  SGLang refuses non-neutral sampling per request, so agentic TOP runs must send neutral sampling;
- Blackwell (SM100): the Qwen3 dense and Qwen3.5-35B-A3B 4-layer gates pass on B200 (SM100), but every FP8 family
  (GLM, DeepSeek-V4, DeepSeek-V4.1) and GLM-5.3-Flash are qualified on H200 only, so the FP8 paths (packed-UE8M0
  DeepGEMM, FlashMLA, DeepSeek-V4 `wo_a`) are untested on SM100; SGLang warns when an FP8 TOP engine starts on SM100.
