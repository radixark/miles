# Chess RL with TITO v2

This experimental recipe trains `Qwen3.8-27B` with GRPO on games against
Stockfish. Each game is one stateful Miles TITO v2 session. The chess harness
owns the board, move validation, Stockfish opponent, compaction, replay journal,
and reward; Miles owns policy serving, exact training samples, optimization,
telemetry, and session cleanup.

The recipe defaults to Miles' native `qwen38small` TITO family so the fixed
Qwen 3.8 template retains reasoning and applies the correct message-boundary
semantics. For Qwen3.6-35B-A3B, select the model and its matching tokenizer with
`--model-name Qwen3.6-35B-A3B --megatron-model-type qwen3.6-35B-A3B --tito-model qwen36`.
The harness, reward policy, and observability hooks are shared by both models.

For asynchronous runs with one trainer node, pass
`--extra-args "--actor-preferred-node-ip 192.0.2.10"` to put the trainer first on
a particular allocated node, for example one with sufficient checkpoint space.
Use the actual Ray node IP. The default numeric-IP ordering is unchanged when
this option is absent, and an unallocated requested node is rejected.

Set `--system-prompt-variant random` to select one of the chess harness's five
UCI-only system prompts independently for every rollout. The selected prompt
stays fixed for the complete game, including retries and context compaction.

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
- Training parallelism follows Miles' supported dense Qwen 3.8 layout:
  TP4, CP1, EP1, PP1, and ETP1. SGLang uses one GPU per rollout engine.
- Qwen thinking is enabled and retained in subsequent turns.
- TITO v2 keeps post-compaction trajectory segments trainable.
- Full training and rollout entropy, Miles dashboard, Prometheus, W&B when
  `WANDB_API_KEY` is present, full Miles traces, and chess replay journals.
- W&B defaults to team `ch271828n-team`, project `miles-chess_run`, and a run
  name equal to the reproducible run ID. Both team and project are configurable.
- No checkpoint is saved by default; pass `--save-checkpoint` only when that
  artifact is wanted.

The base reward is `1.0` for a win or a positive final Stockfish score at the
turn cap, and `0.0` otherwise. Training defaults to `--max-llm-retries-per-move 0`:
there is one initial answer attempt, and an empty, malformed, illegal, or
output-limit answer ends the game immediately. These trajectories receive a base
reward of `0.0`, remain in the training batch, and are not treated as infrastructure errors.
For evaluation, the harness can allow three or five retries after the initial
attempt. Provider/API errors use a separate retry budget.

After assigning the base reward, the chess sample postprocessor subtracts the configured
`--repetition-reward-penalty` (default `0.1`) once from any
rollout whose training samples contain a repetitive 10,000-character window.
Windows use a 5,000-character stride, widened for very long responses to bound the
scan to 32 windows; the first window and exact final suffix are always checked.
TITO compaction siblings share this penalty so the rollout keeps one reward.
Repetitive invalid-move trajectories also receive the penalty: with a configured
penalty of `0.5`, their final training reward is `0.0 - 0.5 = -0.5`, while a
non-repetitive invalid trajectory stays at `0.0`. This creates a reward difference
between repetitive and ordinary failures even when no game in a group succeeds.
The detector is a compression-based heuristic, not a guarantee that every semantic
loop will be detected. The launcher
passes the penalty through chess metadata and disables Miles' second, global
application (`--repetition-reward-penalty 0` in the generated trainer command),
so penalties are not applied twice. The configured chess penalty is recorded in
the recipe configuration, prompt metadata, and saved sample metadata.
Groups containing aborted or infrastructure-error games are rejected and
resampled rather than trained as chess failures.

The harness reports `chess_result.invalid_move_termination` on every trajectory.
The custom rollout logger adds the following metrics to the standard Miles
tracking stream (including W&B and the dashboard):

- `rollout/chess/invalid_move_termination_rate`: fraction of retained training
  trajectories that ended because the model exhausted its answer budget.
- `rollout/chess/invalid_move_termination_count`: number of those trajectories.
- `rollout/chess/trajectory_count`: the denominator, counting each original game
  once even if compaction produced several training samples.

Zero is logged when no games failed. These hooks use the ordinary Miles loss and
advantage calculation; no token-level credit constraint is enabled.

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

Preparation downloads and converts `Qwen/Qwen3.8-27B`, installs Stockfish,
checks out the pinned radix_raft chess harness, and installs its Python package.
Use `--skip-prepare` only after those artifacts are present.

For fully asynchronous training, reserve separate training and rollout nodes:

```bash
python examples/experimental/chess/run.py \
    --run-id 260901-deadbeef \
    --num-nodes 2 \
    --train-num-nodes 1 \
    --fully-async \
    --num-rollout 1000 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 8
```

This uses `train_async.py`, keeps rollout production running during optimizer
updates, trains on one node, and hosts one eight-GPU SGLang engine on the other.
Truncated importance sampling is enabled to account for policy staleness. The
synchronous default remains colocated and continues to use `train.py`.

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
