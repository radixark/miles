# BF16 stochastic rounding validation helpers

Helpers used to validate `OPEN_TRAINING_BF16_STOCHASTIC_ROUNDING_FLAG=1` on a `radixark/miles` dev image.
They are published in one commit and reverted in the next, so the PR's final diff does not include them.

- `bench_sr_copy_back_glm5_experts.py`: HDO's FP32 -> BF16 copy-back on GLM-5 routed-expert shapes
  (linear_fc1 [4096, 6144] + linear_fc2 [6144, 2048] FP32 masters in pageable host memory, initialized
  once and copied back layers x experts times). Modes: `rn` (current copy-back), `sr` (stochastic
  copy-back), `rn-kernel` / `sr-kernel` (device cast or SR kernel only).

  ```bash
  export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
  for mode in rn-kernel sr-kernel rn sr; do
    python tests/manual/bf16_stochastic_rounding_validation/bench_sr_copy_back_glm5_experts.py --mode $mode
  done
  for trial in 1 2 3; do for mode in rn sr; do
    numactl --cpunodebind=0 --membind=0 \
      python tests/manual/bf16_stochastic_rounding_validation/bench_sr_copy_back_glm5_experts.py \
      --mode $mode --layers 75 --experts 32
  done; done
  ```

- `make_e2e_variants.sh`: derives two validation-only copies of
  `tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py` outside the repo: `sr_gpu_adam`
  (no `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`, so the
  distributed optimizer's main-to-model copy rounds) and `rn` (flag `0`, round-to-nearest baseline).
- `queue_e2e.sh LABEL=TEST_FILE ...`: runs e2e files back to back from `/root/miles` with the CI
  `PYTHONPATH` (`/root/miles:/root/Megatron-LM`), one log and exit code per case.

  ```bash
  bash make_e2e_variants.sh
  bash queue_e2e.sh \
    sr-final=tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py \
    sr-gpu-adam=/hai-workspace/bf16-sr/e2e/test_glm5_2_5layer_nvfp4_w4a16_sr_gpu_adam.py \
    rn-final=/hai-workspace/bf16-sr/e2e/test_glm5_2_5layer_nvfp4_w4a16_rn.py
  ```
