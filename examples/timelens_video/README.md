# TimeLens Video Temporal Grounding RL (Qwen3.5)

GRPO on [TimeLens](https://github.com/TencentARC/TimeLens) temporal grounding: given a video and a sentence, the model
answers the start and end time of the event, and the reward is the IoU of its first predicted span with the ground
truth. The recipe follows TimeLens GRPO (tIoU reward, no KL, no std normalization, constant lr 1e-6) on Qwen3.5-9B,
plus one training-reward step of ours (see Notes).

The model is the plugin Qwen3.5 text model with the frozen vision tower
(`miles_plugins.models.qwen3_5.qwen3_5_mm_model_provider`). Videos travel as shared file paths: SGLang samples the
frames and expands the placeholders in token space, and every trainer rank rebuilds the same tensors from the files
in the background (`--multimodal-token-space-processing`).

## Data

```bash
hf download --repo-type dataset TencentARC/TimeLens-Bench --local-dir /root/datasets/timelens/TimeLens-Bench
hf download --repo-type dataset JungleGym/TimeLens-Qwen3-VL-8B-filter-data --local-dir /root/datasets/timelens/filter-data
hf download --repo-type dataset TencentARC/TimeLens-100K --local-dir /root/datasets/timelens/TimeLens-100K
# unpack the video shards of TimeLens-100K into TimeLens-100K/videos, then:
python examples/timelens_video/prepare_data.py /root/datasets/timelens
```

`prepare_data.py` writes `timelens_grpo_train.parquet` (2.5K prompts, sampled by difficulty as TimeLens does) and
`timelens_bench_charades_600.parquet` (600 Charades items for eval). The parquets store absolute video paths.

## Run

```bash
python examples/timelens_video/run_timelens_qwen3_5.py
```

The launcher downloads Qwen3.5-9B, converts it to `torch_dist` (the multimodal model keeps the text model's
parameter names), and trains 300 steps of 16 prompts x 8 samples on 8 GPUs, evaluating Charades every 10 steps.

The trainer and the engines share the GPUs and offload in turn. The vision tower is frozen, and every weight sync
re-sends it from the checkpoint, so the engines get it back after each offload.

## Notes

- Responses cut at the 512-token limit score 0 in training (`zero_truncated_rewards`). This step is ours, not part
  of the TimeLens recipe: the tIoU reward reads only the first predicted span, so text after it is free, and without
  this step a run grew its answers to the 512-token limit for the second half of training. Eval keeps the TimeLens
  scoring.
- Videos decode on CPU (`processor_class_attributes` in `--sglang-mm-process-config`), on the engines and the
  trainer alike: SGLang's GPU video decode would put every colocated engine on cuda:0 and run it out of memory.
- 300 steps x 16 prompts is about two passes over the 2.5K prompts.
