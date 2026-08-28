# Chess RL with TITO v2

This experimental recipe trains `Qwen3.6-35B-A3B` with GRPO on games against
Stockfish. Each game is one stateful Miles TITO v2 session. The chess harness
owns the board, move validation, Stockfish opponent, compaction, replay journal,
and reward; Miles owns policy serving, exact training samples, optimization,
telemetry, and session cleanup.

The recipe uses Miles' native `qwen36` TITO family so the fixed Qwen 3.6
template retains reasoning and applies the correct message-boundary semantics.

## Default smoke configuration

- One node with eight H200 GPUs.
- Ten GRPO steps.
- Eight prompts per step and eight trajectories per prompt: 64 games per step.
- At most 16 games run simultaneously. The remaining trajectories queue, which
  limits the node to 32 resident Stockfish processes because each active game
  owns one opponent engine and one independent review engine.
- Stockfish gets 20 seconds to start and complete its UCI handshake.
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
    --max-model-turns 8 \
    --learning-rate 3e-7 \
    --kl-loss-coef 0.01 \
    --stockfish-max-concurrent-games 16 \
    --stockfish-startup-timeout-seconds 20
```

Preparation downloads and converts `Qwen/Qwen3.6-35B-A3B`, installs Stockfish,
checks out the pinned radix_raft chess harness, and installs its Python package.
Use `--skip-prepare` only after those artifacts are present.

The launcher applies the game limit both in Miles' rollout scheduler and in the
chess agent itself. This bounds the complete engine lifetime, not just the
startup burst. Increase `--stockfish-max-concurrent-games` only after a real
load probe succeeds on the target host.

Run the same 64-game, two-engine-per-game load envelope without model inference:

```bash
PYTHONPATH=examples/experimental/chess python \
    examples/experimental/chess/stockfish_load_probe.py \
    --num_games 64 \
    --max_concurrent_games 16 \
    --stockfish_timeout_seconds 20
```

The probe starts, configures, and exercises both engines in every game, reports
the peak number of live engines, and fails if any Stockfish process remains.

If the node cannot authenticate to the radix_raft remote, transfer a complete
Git checkout to `--radix-raft-dir` before launching. Preparation reuses the
pinned revision when it already exists locally and only fetches it when absent.

When a verified Hugging Face checkpoint already exists on shared storage, pass
it with `--hf-checkpoint-path` and put `--model-dir` on a filesystem large
enough for the converted Megatron checkpoint. The source checkpoint is reused
without downloading or modifying it.

Run artifacts are grouped under `/scratch/<run-id>/`:

- `chess_prompts.jsonl`: eight balanced prompt records.
- `run_manifest.json`: exact launcher arguments, configuration, source
  revisions, container digest, and immutable snapshot references. Secret values
  are never written.
- `chess_games/`: complete chess replay journals and summaries.
- `traces/`: Miles rollout and model traces.
- `checkpoints/`: present only with `--save-checkpoint`.

The default 65,536-token Miles limit, 8,192-token response allowance, and
10,000-token reserve make the chess harness compact its active conversation at
47,344 input tokens. Original generations remain in the replay journal, while
TITO v2 returns the trainable trajectory segments created around compaction.

Set `--kl-loss-coef` to a positive value to regularize the policy toward the
reference model. The default is `0.0`, preserving the original unregularized
recipe.

Set `--learning-rate` to control the constant Adam learning rate. Its default is
`1e-6`, preserving the original recipe.

To extend a run beyond the rollout horizon stored in its checkpoint, resume
with both options below:

```bash
python examples/experimental/chess/run.py \
    --load-checkpoint-path /path/to/checkpoints \
    --override-opt-param-scheduler \
    --num-rollout 320
```

The scheduler override deliberately uses the new launch settings instead of
requiring the new horizon to equal the checkpoint's original horizon. Specify
the same learning-rate and decay settings as the source run when they must stay
unchanged.
