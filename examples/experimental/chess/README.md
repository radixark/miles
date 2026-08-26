# Chess RL with TITO v2

This experimental recipe trains `Qwen3.6-35B-A3B` with GRPO on games against
Stockfish. Each game is one stateful Miles TITO v2 session. The chess harness
owns the board, move validation, Stockfish opponent, compaction, replay journal,
and reward; Miles owns policy serving, exact training samples, optimization,
telemetry, and session cleanup.

The recipe temporarily uses Miles' `qwen35` TITO family for Qwen 3.6. This is
useful for testing because Qwen 3.5 and Qwen 3.6 share the relevant message
boundaries and thinking format, but production Qwen 3.6 training should move to
a dedicated, verified TITO family when one is available.

## Default smoke configuration

- One node with eight H200 GPUs.
- Ten GRPO steps.
- Eight prompts per step and eight trajectories per prompt: 64 games per step.
- Four prompts assign the policy White and four assign it Black.
- Stockfish Elo 1320.
- Eight policy moves per game.
- Training parallelism follows Miles' tested long-context Qwen layout:
  TP2, CP2, EP8, PP1, and ETP1.
- Qwen thinking is enabled and retained in subsequent turns.
- TITO v2 keeps post-compaction trajectory segments trainable.
- Full training and rollout entropy, Miles dashboard, Prometheus, W&B when
  `WANDB_API_KEY` is present, full Miles traces, and chess replay journals.
- W&B defaults to team `ch271828n-team`, project `miles-chess_run`, and a run
  name equal to the reproducible run ID. Both team and project are configurable.
- No checkpoint is saved by default. A full Qwen 3.6 checkpoint is roughly
  464 GB; pass `--save-checkpoint` only when that artifact is wanted.

The turn-cap reward is `1.0` when the final Stockfish score from the policy's
perspective is positive and `0.0` otherwise. Games ending normally keep the
harness rewards: win `1.0`, draw `0.0`, loss `-1.0`, infrastructure error `0.0`.
Groups containing aborted or infrastructure-error games are rejected and
resampled rather than trained as chess failures.

## Launch

Use a current Miles checkout and provide a reproducible run ID:

```bash
python examples/experimental/chess/run.py \
    --run-id 260825-deadbeef \
    --output-dir /scratch \
    --num-rollout 10 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 8 \
    --max-model-turns 8
```

Preparation downloads and converts `Qwen/Qwen3.6-35B-A3B`, installs Stockfish,
checks out the pinned radix_raft chess harness, and installs its Python package.
Use `--skip-prepare` only after those artifacts are present.

When a verified Hugging Face checkpoint already exists on shared storage, pass
it with `--hf-checkpoint-path` and put `--model-dir` on a filesystem large
enough for the converted Megatron checkpoint. The source checkpoint is reused
without downloading or modifying it.

Run artifacts are grouped under `/scratch/<run-id>/`:

- `chess_prompts.jsonl`: eight balanced prompt records.
- `chess_games/`: complete chess replay journals and summaries.
- `traces/`: Miles rollout and model traces.
- `checkpoints/`: present only with `--save-checkpoint`.

The default 65,536-token Miles limit, 8,192-token response allowance, and
10,000-token reserve make the chess harness compact its active conversation at
47,344 input tokens. Original generations remain in the replay journal, while
TITO v2 returns the trainable trajectory segments created around compaction.
