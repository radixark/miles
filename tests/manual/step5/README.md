# Step5 parity checks

Golden per-token log-probs from StepFun's reference implementation, and a comparison of SGLang against them.
The reference code ships in the checkpoint (`fp8_vit_mtp3/inference/`). It needs Transformers 5.15, so run it
with an overlay (`pip install --target /scratch/ref-pylib transformers==5.15.0`, then
`PYTHONPATH=/scratch/ref-pylib`) instead of changing the image's environment.

1. Slice the checkpoint (it must carry the step5 `config.json`):

   ```bash
   python tools/slice_step5_checkpoint.py --src /scratch/step5_ckpt/fp8_vit_mtp3 \
       --dst /scratch/step5_slices/step5-5layer --layers 0,3,4,88,91
   ```

2. Shard the slice for the reference, then write prompts and golden log-probs. `--valid-vocab` matches SGLang,
   which normalizes over the tokenizer's 128,815 ids rather than the physical 128,896. `$MILES` is this checkout:

   ```bash
   cd /scratch/step5_ckpt/fp8_vit_mtp3/inference
   python convert.py --checkpoint /scratch/step5_slices/step5-5layer \
       --out-dir /scratch/step5_ref/step5-5layer --tp-size 4 --ep-size 4
   python $MILES/tests/manual/step5/make_prompts.py /scratch/step5_golden/prompts.json
   PYTHONPATH=/scratch/ref-pylib torchrun --standalone --nproc-per-node=4 $MILES/tests/manual/step5/golden_logprobs.py \
       --ref-code . --tp-dir /scratch/step5_ref/step5-5layer/tp4 \
       --prompts /scratch/step5_golden/prompts.json --valid-vocab 128815 \
       --out /scratch/step5_golden/step5-5layer.pt
   ```

3. Serve the same slice with SGLang and compare. Truncated slices have only a few full-attention layers, so cap
   the KV pool (`--max-total-tokens 1048576`); otherwise one layer's K cache exceeds the int32 strides of the
   CuTe DSA kernels:

   ```bash
   python $MILES/tests/manual/step5/compare_sglang.py --golden /scratch/step5_golden/step5-5layer.pt \
       --url http://127.0.0.1:30000 --out /scratch/step5_golden/step5-5layer-sglang.json
   ```

The prompts are 32, 600 and 5,000 random tokens: short, past the 512-token sliding window, and past the
4,103-token point where the DSA layers start selecting regions.

## Megatron trainer

4. Cast the slice to BF16 (`tools/fp8_cast_bf16.py`) and write BF16 goldens. The reference reads every MoE
   layer as FP8 unless its config excludes it, so give it a view of the BF16 slice first:

   ```bash
   python $MILES/tests/manual/step5/reference_bf16_view.py --bf16 /scratch/step5_slices/step5-5layer-bf16 \
       --fp8 /scratch/step5_slices/step5-5layer --out /scratch/step5_ref_input/step5-5layer-bf16
   python convert.py --checkpoint /scratch/step5_ref_input/step5-5layer-bf16 \
       --out-dir /scratch/step5_ref/step5-5layer-bf16 --tp-size 4 --ep-size 4
   ```

   then run `golden_logprobs.py` on `step5-5layer-bf16/tp4` as in step 2, writing
   `--out /scratch/step5_golden/step5-5layer-bf16.pt`.

5. Build the Megatron model from the Step5 model flags, load the HF slice through the Step5 bridge (FP8
   experts are dequantized on read, so the FP8 slice works too) and compare:

   ```bash
   PYTHONPATH=/root/Megatron-LM:$PWD torchrun --standalone --nproc-per-node=1 \
       tests/manual/step5/megatron_logprobs.py \
       $(python miles/utils/external_utils/model_args_utils.py step5-5layer) \
       --hf-checkpoint /scratch/step5_slices/step5-5layer-bf16 \
       --golden /scratch/step5_golden/step5-5layer-bf16.pt \
       --report /scratch/step5_golden/step5-5layer-bf16-megatron.json
   ```

To judge the numbers, compare the reference against itself at another layout (`--tp-size 1 --ep-size 1`):
on GB300 the Megatron model's mean absolute log-prob difference equals the reference's own TP1-vs-TP4
difference (measured on an earlier 8-layer slice: 0.0089 vs 0.0092 on 32 tokens, 0.0069 vs 0.0066 on 600). On 5,000 tokens, where
the full-attention layers select regions, it is 0.0070 vs 0.0062, and 0.0072 vs 0.0067 on the 895 positions
past 4,103; the max is 0.22 vs 0.24. The Megatron model's own log-probs move by up to 0.17 between two runs
of the same prompt (0.04 on 600 tokens before the sparse layers too), so compare means.


## Vision tower and VL model

6. Compare the trainer's frozen vision tower (`miles_plugins/models/step5/vision.py`) with StepFun's reference
   encoder and SGLang's `PerceptionEncoder` on the checkpoint's real vision weights:

   ```bash
   PYTHONPATH=$PWD:/root/Megatron-LM python tests/manual/step5/vision_parity.py \
       --hf-checkpoint /scratch/step5_slices/step5-5layer --ref-code /scratch/step5_ref_code \
       --report /scratch/step5_golden/vision-parity.json
   ```

   On GB300 the 4096-wide features of a 728-pixel overview (mean |x| 4.1) differ from SGLang's by 0.026 mean
   absolute (max 1.0, one BF16 ulp at the largest values), as much as the tower differs from itself with flash
   instead of cuDNN attention. The reference is 0.030 away from both: it rounds the RoPE angles to BF16, where
   SGLang and the tower keep them in fp32. With BF16 angles the tower equals the reference bit for bit up to the
   projector. 504-pixel crops give the same numbers.

7. Build the VL model through its provider, load the converted text checkpoint strictly, and run one forward and
   backward on mixed image/text micro-batches from the real processor and collator:

   ```bash
   PYTHONPATH=$PWD:/root/Megatron-LM torchrun --standalone --nproc-per-node=1 tests/manual/step5/vl_model_smoke.py \
       $(python miles/utils/external_utils/model_args_utils.py step5-5layer) \
       --custom-model-provider-path miles_plugins.models.step5.vl_model.step5_vl_model_provider \
       --hf-checkpoint /scratch/step5_slices/step5-5layer --no-gradient-accumulation-fusion \
       --load /scratch/step5_slices/step5-5layer_torch_dist --no-load-optim --no-load-rng
   ```

   `--load` is the slice's torch_dist conversion (`tools/convert_hf_to_torch_dist.py`, as
   `scripts/run_step5.py prepare` runs it). The smoke test checks image-feature placement, gradients only through text parameters, text-only batches,
   and exclusion of the frozen tower from checkpoints and weight sync. For TP 2 with sequence parallelism,
   drop `--load ...` to read HF weights through the bridge, use `--nproc-per-node=2`, and add
   `--tensor-model-parallel-size 2 --sequence-parallel --expert-model-parallel-size 2
   --expert-tensor-parallel-size 1`.

## Image preprocessing

8. `test_processor_parity.py` runs miles' `Step5TrainProcessor` and SGLang's Step3-VL processor on the request
   miles sends (a small square, a 1000x300 and a 1600x1200 image, and a tall one). The expanded prompt must match
   token for token; the pixel tensors match exactly on the CPU and within 5e-4 when SGLang resizes on the GPU:

   ```bash
   STEP5_HF_CHECKPOINT=/scratch/step5_slices/step5-5layer pytest -s tests/manual/step5/test_processor_parity.py
   ```


## DSA region selection

The trainer recomputes the full-attention layers' region selection with the frozen indexer
(`miles_plugins/models/step5/indexer.py`); these check it against SGLang's Step5 indexer kernels on GB300.

9. Capture the first full-attention layer's attention input (the input norm's BF16 output) from the Megatron
   model for random prompts of 5,000 and 32,768 tokens and 32,768 tokens of each text glob:

   ```bash
   PYTHONPATH=$PWD:/root/Megatron-LM torchrun --standalone --nproc-per-node=1 \
       tests/manual/step5/indexer_inputs.py \
       $(python miles/utils/external_utils/model_args_utils.py step5-5layer) \
       --hf-checkpoint /scratch/step5_slices/step5-5layer \
       --load /scratch/step5_slices/step5-5layer_torch_dist --no-load-optim --no-load-rng \
       --text-files 'docs/**/*.md' '/root/Megatron-LM/megatron/core/**/*.py' \
       --out /scratch/step5_golden/indexer_inputs.pt
   ```

   Put this tree before Megatron-LM on `PYTHONPATH`: Megatron's own `tools` package shadows ours otherwise.

10. Compare `Step5Indexer` with SGLang's kernels (norm/RoPE, compression, the prefill and decode scoring
    paths, DeepSelect) and StepFun's reference on those inputs, per full-attention layer of the slice:

    ```bash
    PYTHONPATH=$PWD:/root/Megatron-LM:/sgl-workspace/sglang/python \
        python tests/manual/step5/indexer_parity.py --hf-checkpoint /scratch/step5_slices/step5-5layer \
        --inputs /scratch/step5_golden/indexer_inputs.pt --layers 1 4 \
        --ref-code /scratch/step5_ref_code --report /scratch/step5_golden/indexer_parity.json
    ```

    The `stages` report compares normalized keys, FP8 queries, and compressed keys bit for bit. The
    "shared" path holds projections fixed to isolate kernel arithmetic; "served" includes SGLang's
    projection shapes. Both require an SGLang build with Step5 support. On CUDA the trainer computes the
    head weights with `miles/kernels/linear/batch_invariant_linear.py`, rounded as SGLang's batch-invariant
    kernel rounds them; on CPU it falls back to the FP64 GEMM.

    **Exact selector parity requires identical scores.** DeepSelect's tie visitation is reproduced,
    including the tail-first permutation beyond 8,192 candidates. Score generation still has two
    numerical limits: PyTorch FP32 dot products can differ from tensor-core accumulation in the last bit,
    and SGLang's prefill head reduction uses FMAs. Either can change a selection at the top-k boundary.
    The earlier real-weight run matched prefill on 99.992–99.998% of rows; that measurement is not a
    guarantee for other inputs or revisions.


## Sparse full attention

The full-attention layers run the indexer's selection through the sparse region attention kernel
(`miles_plugins/models/step5/sparse_attention.py`). `tests/fast-gpu/test_step5_sparse_attention.py` checks it
against the dense layer it replaced, at TP 1, 2 and 4 with and without sequence parallelism, and checks that
full-layer recompute selects the forward's regions bit for bit. It also compares block IDs, attention outputs
and gradients with reuse enabled versus disabled, and checks that reuse reduces Indexer calls from two to one.
See the [Step5 model guide](../../../docs/models/stepfun/step5.md#54-notable-quirks).

11. Time one layer's attention on its real input (step 9's capture), against the dense layer, with and without
    Megatron's activation checkpoint:

    ```bash
    PYTHONPATH=$PWD:/root/Megatron-LM torchrun --standalone --nproc-per-node=4 \
        tests/manual/step5/bench_sparse_attention_layer.py --hf-checkpoint /scratch/step5_slices/step5-5layer-bf16 \
        --inputs /scratch/step5_golden/indexer_inputs.pt --prompt text0-32768 --sequence-parallel
    ```

    Compare `sparse_recompute_ms` (reuse disabled) with `sparse_recompute_reuse_ms` (reuse enabled).
    Both runs keep the Indexer frozen and report forward, backward and total time, with corresponding
    `*_peak_mib` memory increments.

    On GB300 at TP4 + SP, one 32,768-token sequence of the first full-attention layer (original layer 3; medians, ms, and peak MiB over the weights):

    | | indexer | forward | backward | fwd + bwd | with recompute | peak | peak, recompute |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | sparse | 34-37 | 44-46 | 25.5 | 69.5-71.4 | 112-116 | 2393 | 2457 |
    | dense | | 20.4 | 55.4 | 75.9 | 96.1 | 2473 | 2537 |

    The kernel alone takes 5.5 ms forward and 19.4 ms backward, so the indexer dominates the layer, and full
    recompute runs it twice. Two 32K sequences in one pack cost twice one. At TP1 the indexer takes 90 ms and
    the layer 340 ms with recompute, against 368 ms dense.

    These timings predate the trainer's CUDA head-weight kernel and were not remeasured after that change.

To benchmark only the attention kernel:

```bash
python tests/manual/step5/bench_sparse_region_attention.py --tokens 4096 16384 32768 --groups 1 2 4
```

`--selection <file>` replays a captured selection instead: a torch file whose `region_ids` is an int32
`[T, G, 513]` tensor (no script here writes one). Without it, the benchmark samples earlier regions with a bias toward recent tokens. Dense causal
attention is equivalent only when every earlier region is selected; it is a timing baseline for longer sparse
sequences. Layer benchmarks report the indexer, full attention, kernel-only attention, and recomputation costs
separately, as CUDA-event medians and peak-memory increments above the resident weights and inputs.

## Cross-validation on October 5, 2026

The trainer, SGLang, and Megatron `a84b10547` were checked on four GB300 GPUs with synthetic
inputs. The environment used PyTorch `2.13.0+cu130`, Triton `3.7.1`, and Transformer Engine `2.17.1`.

- 28 weight/attention kernel tests and 24 distributed attention tests passed, including forward/backward,
  TP 1/2/4, sequence parallelism, and exact selection agreement during recomputation.
- 142 model, bridge, vision, processor, launcher, and checkpoint-slicing tests passed.
- 135 shared-backend tests passed; one optional `megatron.bridge` checkpoint test was skipped.
- All 888 DeepSelect comparison rows matched across 513–131,073 candidates. Seven head-weight cases matched
  both SGLang and FP64, including strided inputs. Selector CUDA graph replay matched.
- Synthetic decode checks at 5,000 and 16,393 tokens matched the normalization/compression stages, valid
  scores, and all 13,185 scored selection rows, for one provider group per length.
- The PyTorch attention reference matched Triton on packed sparse inputs: relative L2 errors were 0.00162
  for outputs and 0.00313/0.00664/0.00656 for query/key/value gradients; all values were finite.

Not executed in this pass: a full-checkpoint training job, real-weight vision parity, SGLang prefill parity,
or end-to-end rollout log-probability parity. The measurements in earlier sections are historical results.
