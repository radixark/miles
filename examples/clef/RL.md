# Decision-head RL

`python -m examples.clef.rl_train` under `torchrun` warm-starts both the
backbone and the trained `joint_head.safetensors` from an HF export. This is
categorical policy optimization, not supervised JSON completion or token RL.
The head jointly reads the schema; the action policy factorizes across fields.

For each record, sample 32 complete decision sets at temperature 1. Reward is
`0.5 * fraction_of_exactly_correct_fields + 0.5 * all_fields_correct` by default.
There is no credit for nearby ordinal values. Each record's group gets centered,
population-standard-deviation-normalized advantages; constant groups contribute
zero policy gradient. The probability ratio uses the sum of field log probabilities.

Loss is clipped GRPO plus field-averaged differentiable Brier loss (weight 1)
and exact categorical KL to the initial checkpoint (weight 0.1). The Brier term
is not a common detached reward. Accuracy reward alone can encourage confident
predictions; the auxiliary terms do not guarantee calibration after training.

This first version uses one optimizer update per fresh rollout batch. Consequently
ratios start at one and clipping is normally inactive: it has no multiple-epoch
PPO replay or asynchronous rollout workers. Sampling a group costs one head
forward, rather than 32 prompt evaluations. The gradient pass is a second forward.
Dropout is disabled, and schemas/options remain fixed while record order shuffles.

Only exact one-hot training labels are accepted. Soft probability tasks may still
be used for validation. Microbatch size is currently one; default global batch
is 64 records, with sharded gradient accumulation across GPUs. Backbone/head
learning rates default to `1e-7`/`1e-6`, constant throughout the run.

The entrypoint reuses supervised FSDP2 FP32 master weights, checkpoint export,
validation, dashboard, W&B, and per-rank trace infrastructure. Traces include
probabilities, sampled option indices, option IDs, rewards, advantages, old log
probabilities, and objective metrics. Metrics include exact field accuracy,
collapse, reward variance, constant groups, entropy, KL, and clipping fraction.

The initial reference probabilities are cached in `output_dir/reference.json`.
Save this file with checkpoints: resuming requires copying it to the resumed
output directory on every node when output storage is local. Resume validates
its content digest, initial model file digests, dataset digests, and
optimizer/objective settings. A supervised native
checkpoint is not an RL resume: use its HF export to initialize a new RL run
with a fresh optimizer. Initial head loading is strict and handles the scalar
shape conversion used by FSDP exports.

Before production launch, validate memory requirements and native checkpoint
round-trip on the actual model. `python -m examples.clef.rl_check --device cuda:0`
checks objective behavior and a tiny real Qwen/head gradient update. Under
two-GPU `torchrun`, add `--distributed` to check a sharded update and reference
cache round-trip. These fixtures do not establish full-model training readiness.

No new dependencies are required beyond the existing Clef training environment.
