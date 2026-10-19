# Primus backend

GRPO on GLM-5.3-Flash with the Primus trainer: DAPO-Math reward, FP8 rollout from the released checkpoint, and context-parallel dynamic batching.

Primus builds GLM-5.3-Flash (`model_type: glm5_next`) and loads the HuggingFace
checkpoint in process, so this recipe does not convert weights to `torch_dist`
first. The trainer stays BF16. SGLang serves the released FP8 checkpoint.

## Why this task

DAPO-Math-17k grades the boxed final answer, so the reward is 0 or 1 and a run
that is learning shows it immediately. On the Megatron recipe the same data
goes from about 0.5 to 0.94 mean reward inside 10 rollouts, with
`train/train_rollout_logprob_abs_diff` between 0.007 and 0.011. AIME-2024 is
the greedy eval.

The rollout checkpoint is the released FP8 model
(`quant_method: fp8`, `weight_block_size: [128, 128]`). Weight updates
re-quantize the BF16 trainer with that config and leave every module in
`modules_to_not_convert` in BF16, which is the layout SGLang loaded. Read
`train/train_rollout_logprob_abs_diff` on the first step before trusting a
longer run. A gap near 0.01 is the healthy band; a gap near 0.1 means the
rollout kernels and the trainer disagree and TIS will spend the update
correcting that instead of the reward.

## What the launcher turns on

`examples/primus/run-glm5.3-flash-reasoning-rl.sh`:

| Piece | Setting |
|---|---|
| Data | DAPO-Math-17k train, AIME-2024 eval, `--rm-type math` |
| Rollout | `HF_CKPT` = released FP8 directory. `MILES_SGLANG_DUMMY_LOAD=0` |
| Trainer init | `REF_CKPT` = BF16 cast of the same checkpoint, `--ref-load` |
| Algorithm | GRPO, 8 prompts × 8 samples, TIS, rollout routing replay, clip 0.2 / 0.28, no KL |
| Optimizer | Adam, constant `lr 1e-6`, CPU offload |
| Parallelism | PP = one stage per trainer node, EP = 8, CP = 8 by default |
| Batching | `--qkv-format thd --allgather-cp --use-dynamic-batch-size`, 4096 tokens per GPU |
| Schedule | Synchronous. Fully async is off so staleness does not flatten the reward |

`MAX_TOKENS_PER_GPU * CP_SIZE` has to be at least the prompt budget plus the
longest response. The defaults are 4096 × 8 = 32768, against an 8192-token
response and a 2048-token prompt. CP shares the trainer GPUs with EP; it does
not add nodes. `CP_SIZE=1` falls back to one sample per micro-batch and no
dynamic batching.

`--distributed-timeout-minutes` is 30. The Gloo group uses that same timeout,
so a long context-parallel step is not killed at PyTorch's 10-minute default.

## Launch

Bring up Ray, then on the head:

```bash
export HF_CKPT=/path/to/GLM-5.3-Flash
export REF_CKPT=/path/to/GLM-5.3-Flash-bf16
export DATA_DIR=/path/to/datasets
export PRIMUS_DIR=/path/to/Primus
bash examples/primus/run-glm5.3-flash-reasoning-rl.sh
```

`HF_CKPT` and `REF_CKPT` must be visible to every node. The datasets are
`zhuzilin/dapo-math-17k` and `zhuzilin/aime-2024`.

Other scripts in this directory are bring-up variants: a BF16 rollout of the
same math task, an SFT launcher, and a multi-turn HIP-kernel task. The
reasoning script above is the one to start from.
