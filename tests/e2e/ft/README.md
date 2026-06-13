# Fault Tolerance E2E Tests

## Layout

Each scenario's logic lives in a library module `conftest_ft/scenario_<scenario>.py`. CI
runs it through thin **per-mode entry files** `test_trainer_ft_<scenario>_<mode>.py` — one
mode each, registered with `register_cuda_ci(est_time=..., suite="stage-c-8-gpu-h200",
labels=["ft"])`. The CUDA CI runner executes each entry as bare `python3 <file>` (exit code
= pass/fail), so the entry just calls the scenario's `run_ci(mode)`.

| Scenario (`conftest_ft/scenario_*.py`) | Type | What it verifies |
|------|------|-----------------|
| `scenario_no_failure` | Comparison | indep_dp matches normal DP when no faults |
| `scenario_with_failure` | Comparison, multi-phase | indep_dp matches normal DP under fault + healing, on cold start and on resume from a post-FT ckpt |
| `scenario_deterministic` | Comparison, multi-phase | healing state transfer is bitwise-correct (stop+start), on cold start and on resume from a post-healing ckpt |
| `scenario_ft_random` | Non-comparison | system survives random crashes without hanging |
| `scenario_realistic_gsm8k` | Non-comparison | model still reaches gsm8k accuracy under random crashes |

## Mode Variants

Each scenario runs with a `--mode`:

All modes are **disaggregated** (training and rollout on separate nodes). Modes without rollout use debug rollout data.

| Mode | Nodes | DP cells | Parallelism | Rollout | Model | Coverage |
|------|-------|----------|-------------|---------|-------|----------|
| `dp2_cp2_tp2_ep2` | 1 | 2 | CP2 TP2 EP2 | debug data | 5-layer MoE | TP + EP |
| `dp2_cp2_pp2` | 1 | 2 | CP2 PP2 | debug data | 5-layer MoE | PP |
| `dp4_cp2` | 1 | 4 | CP2 | debug data | 5-layer MoE | Multi-replica (>=4 cells) |
| `dp2_cp2_real_rollout` | 1 | 2 | CP2 | 4 engines × 1 GPU | 5-layer MoE | Real rollout engines + weight update path (no_failure, deterministic) |
| `dp2_cp2_real_rollout_dense` | 1 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | Real rollout under a fault + injection match guard (with_failure) |
| `6node_dp4_cp2_tp2_pp2_ep2_etp2` | 4+2 | 4 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | full MoE | Large-scale, all parallelism |

All scenarios use `--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256`
(256 samples per rollout), which divides evenly across both 2 and 4 cells. Uneven sample
distribution across replicas is **not** currently exercised by these tests.

The 1-node modes use the truncated 5-layer MoE model (`Qwen3-30B-A3B-5layer`), except
`dp2_cp2_real_rollout_dense`, which uses a small real dense model (`Qwen3-0.6B`) — see the
`scenario_with_failure` definition for why the injection match guard requires it.

Authorized CI skips (no entry file): `6node_dp4_cp2_tp2_pp2_ep2_etp2` (multi-node) and `with_failure × dp4_cp2`.

## Running

### In CI

The FT entries are gated on the `run-ci-ft` PR label (FT is expensive — it does not run on
every PR). With that label set, every `test_trainer_ft_<scenario>_<mode>.py` runs on the
`stage-c-8-gpu-h200` suite. To add a new `(scenario, mode)` to CI, add an entry file (copy
an existing one and change `run_ci(...)`'s mode). To add a brand-new label, edit
`tests/ci/labels.py` and create the matching `run-ci-<label>` GitHub label.

### Manually

Set `PYTHONPATH` to the repo root (CI sets it automatically). Two ways:

1. One mode, exactly as CI runs it — invoke the entry file:

   ```bash
   PYTHONPATH=. python tests/e2e/ft/test_trainer_ft_no_failure_dp2_cp2_tp2_ep2.py
   ```

2. Any mode (incl. the authorized-skip ones) — invoke the scenario's typer app with a
   subcommand:

   | subcommand | does |
   |---|---|
   | `run` | full pipeline: prepare + baseline + target + compare |
   | `baseline` / `target` | run one side only (debugging) |
   | `compare` | re-run comparison on existing dumps (no GPU) |

   ```bash
   PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_<name>.py run --mode <mode>
   ```

   When debugging a failure, prefer the individual subcommands (with a shared `--dump-dir`,
   and `--phase` for multi-phase scenarios) over `run`, so you only re-run the part you
   changed and reuse what already passed — e.g. re-run just `compare` (no GPU) on existing
   dumps, or re-run a single side / phase.

   `scenario_ft_random` is non-comparison: only `run`, with `--seed` / `--num-steps` /
   `--crash-probability`. `scenario_realistic_gsm8k` is likewise non-comparison (and has
   no `--mode`): only `run`, with `--seed` / `--num-rollout` / `--crash-probability` /
   `--metric-threshold`. Each scenario's modes, phases, and knobs are in Mode Variants and
   Test Definitions.

Dumps are written under `/node_public/dumps/<test_name>/` (see `conftest_ft/app.py`
`resolve_dump_dir`).

## Comparison criterion

Dumps use per-tensor boolean predicates over `rel`/`max_abs`/`mean_abs`
(`compare_dumps(diff_thresholds=[(name_regex, predicate), ...])`): the deterministic
scenario requires bitwise equality (`rel <= 0`), which relies on both
`--deterministic-mode` (kernel determinism) and `--debug-deterministic-collective`
(fixed-tree SUM collectives). Metrics are also compared at `rtol=atol=0`, except
`train/grad_norm` (`rtol<=1e-6`): its bracketing depends on the distributed-optimizer
shard count (8 flat vs 2 per cell), so a few fp32 ulps are inherent, while the grads
themselves stay bitwise-checked via the dumps. Every other scenario allows a small
relative diff (`rel <= 0.0085`, with_failure also flooring near-zero MoE-expert and
QK-norm (`q_layernorm`/`k_layernorm`) grads at `max_abs <= 1e-3`). Unmatched tensors are a fail-closed error, so end each list with a `.*`
catch-all. Exact per-scenario thresholds are in Test Definitions below.

## Debug Rollout Data

Modes without rollout engines (`has_real_rollout == False`) use pre-recorded rollout data via `--load-debug-rollout-data --debug-train-only`, skipping real rollout generation.

`conftest_ft/execution.py` `prepare()` downloads the data via `U.hf_download_dataset()`.

### How to regenerate

The debug rollout data **must** be generated using the 5-layer model (not the full model).
Using the full model produces `rollout_log_probs` incompatible with the 5-layer training model,
causing NaN gradients in GRPO training.

```bash
# Step 1: Generate rollout data (5-layer model + real sglang rollout, no dumper)
PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_no_failure.py generate-data \
    --mode dp2_cp2_real_rollout --num-steps 12 --output-dir /tmp/gen_rollout

# Step 2: Locate the generated rollout data
ls /tmp/gen_rollout/rollout_data/

# Step 3: Upload to HF
huggingface-cli upload --repo-type dataset fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer \
    /tmp/gen_rollout/rollout_data/
```

---

## Test Definitions

### `scenario_no_failure`

Comparison test. Verifies indep_dp produces the same results as normal DP when no faults occur.

```
Type: comparison (baseline=normal DP, target=indep_dp)
Steps: 2

1. Baseline: run normal DP training with debug rollout data
2. Target: run indep_dp training with the same data
3. Compare:
   - Tensor-level: compare_dumps (weights, grads via dumper & sglang comparator), rel <= 0.0085
   - Metric-level: compare_metrics (MetricEvent, requires train/grad_norm and train/loss)

Roughly equal, not bitwise — allreduce kernel ordering differs across topologies.
```

### `scenario_with_failure`

Multi-phase comparison test. Verifies indep_dp matches normal DP under fault + healing, in
both start regimes: cold start (phase_a) and resume from a post-FT checkpoint (phase_b).

```
Type: comparison, multi-phase (phase_a + phase_b)
One shared builder parameterized by the phase's start rollout id P emits both phases: 3
rollouts, the same relative fault timeline, ckpt save after every rollout (--save-interval
1). Only the start regime differs:
  phase_a: cold start (no --load, start_rollout_id=0) — rollouts 0..2 (P=0)
  phase_b: resumes from phase_a's last (rollout-2) ckpt
           (start_rollout_id = loaded + 1 = 3) — rollouts 3..5 (P=3)
--num-rollout is 6 for both phases (exclusive end rollout id, not a per-run count); each
phase stops after its 3 rollouts via --debug-exit-after-rollout 3, which counts rollouts
executed within the run and fires after that rollout's ckpt save.

Per-phase timeline — baseline:
  1. Rollouts P..P+2: normal steps, save ckpt after every rollout

Per-phase timeline — target:
  1. Rollout P: N cells normal
  2. Rollout P+1, attempt 0: crash_before_allreduce on last cell rank 0
     → os._exit(1) → allreduce timeout → should_commit=false → retry
  3. Rollout P+1, attempt 1: _refresh_cells() reconfigure → N-1 cells → commit
  4. After rollout P+1: stop_cell_at_end(last) + start_cell_at_end(last)
  5. Rollout P+2: _refresh_cells() healing → N cells, trains with the healed cell
  Saving after every rollout (the P+1 degraded-quorum commit and the P+2 post-healing
  step included) makes phase_b's resume exercise a post-FT checkpoint.

Compare: BOTH phases' dumps per rollout (rel <= 0.0085; MoE expert grads and QK-norm grads
also tolerate max_abs <= 1e-3; in the real_rollout mode every rollout from the first
post-fault one (rollout 2) onward — including all of phase_b, whose target resumes from a
drift-carrying ckpt — tolerates max_abs <= 3e-3 on the measured grad families, see the
dense-mode section below) and metrics (rtol=5e-2).

Healing witness: each target phase runs the same fault timeline, so each target event dir
must contain exactly two CellReconfigureEvents, in order — a shrink at rollout P+1 (alive
N -> N-1, positive proof the fault injection fired) and a healing at rollout P+2 (healed =
last cell, ckpt src = cell 0, alive back to N). Concretely the global rollout ids are
phase_a: shrink at 1, heal at 2; phase_b: shrink at 4, heal at 5. Both baseline event dirs
must contain zero reconfigure events. This positively proves the crash -> shrink -> heal
path executed in every phase; without it the comparison could silently degenerate to
fault-free runs.

Fault injection via --ci-ft-test-actions JSON (data-driven, executed by RayTrainGroup).
The JSON `at_rollout` field specifies which rollout_id triggers the action.
The `attempt` field (for actor-level actions like `crash_before_allreduce`) specifies which retry attempt to match.
```

The `dp2_cp2_real_rollout_dense` mode runs this scenario with live generation (real sglang
engines, deterministic inference, temperature 0.8), and the target's **post-fault rollouts
inject the baseline's recorded rollout data** (`--ci-inject-rollout-data-path` pointing at
the same phase's baseline `--save-debug-rollout-data` output; both sides record every
phase). The injection start id is `max(phase start, first post-fault rollout id)` per
phase: phase_a injects from rollout 2 (right after its fault); phase_b injects from rollout
3 — all of its rollouts — because the target resumes from phase_a's drift-carrying ckpt, so
every phase_b rollout generates on drifted weights (including its own crash rollout 4,
whose injected data is redriven by the retry).
Rationale: the post-crash degraded-quorum commit accumulates microbatches in a different
floating-point bracketing than the fault-free side — a fault-inherent ulp-level weight
difference no collective ordering removes. Under live sampling that drift flips individual
sampled tokens (the recovered weights are numerically correct; the sampler amplifies ulps),
after which the two runs' rollout data diverges wholesale — so a strict vs-baseline
numerical comparison of *real-sampled* post-fault rollouts is ill-posed, not merely strict.
Injection makes post-fault training inputs identical by construction and restores the full
strict grad/activation/metric comparison with zero threshold relaxation.

What stays real on the target during injected rollouts: engines and generation itself (the
generated samples are discarded after the fact), update_weights after the degraded commit
and after healing, and health-monitor pause/resume — i.e. the whole
crash→retry→heal→weight-sync path. Known gap: the update_weights after each phase's
post-healing train step (rollouts 2 and 5, the last rollout of each phase) runs but its
output is consumed by no later rollout in that run (phase_b restarts from the checkpoint),
so a regression there is only caught by the generation match guard at the injected rollouts
(post-degraded-commit / post-resume weights) and by `realistic_gsm8k` accuracy, not by this
scenario's strict comparison. Injected rollouts' dump comparison gives a
`max_abs <= 3e-3` floor to the **measured noisy grad families only** (decoder-layer
QK-norms, folded `layer_norm_weight`s, and the attention/MLP weight matrices): the
training data is bitwise-identical, but the target's weights carry the fault-inherent ulp
drift of the degraded commit, which lands in those cancellation-dominated near-zero grads
as absolute noise measured up to 2.8e-3 (40 tensors, 2026-06-12; e.g. q_layernorm grads at
rel 20% but max_abs 2.6e-3) while real grads sit at ~1e-2 — the same argument as the
pre-existing 1e-3 QK-norm floor, recalibrated for the converged dense model whose
near-zero grads are smaller. Embedding/output-layer/final-norm grads, all activations, and
all pre-fault rollouts keep the strict predicate set.

The discarded generation is not wasted: each injected
rollout asserts the generated responses match the recording at a mean per-token positional
ratio above a calibrated threshold with bitwise-identical prompts
(`RolloutDataInjectionUtil.assert_matches_generated`). ulp-level drift only flips an
occasional sampled token (which then cascades within that one response), keeping the mean
ratio high, while grossly wrong post-fault engine weights (e.g. a broken update_weights)
make responses unrelated to the recording and drop it by two orders of magnitude — so
wrong-weights bugs still fail the test even though the injected data replaces the
generated content for training. What the scenario does not assert is the exact post-fault
sampled content beyond that ratio. Pre-fault rollouts (phase_a's rollouts 0 and 1; the
crash rollout's data is generated before the crash and redriven by the retry) are not
injected — they remain a real sampled-data comparison.

The guard's calibration (measured 2026-06-12, first post-fault rollout, 256 samples,
correct weights — note the metric counts everything after a response's first flipped
token as mismatched, so even rare flips cost a large fraction):

| Model | mean response-token match | min |
|-------|---------------------------|-----|
| dense Qwen3-0.6B | **0.63** | 0.035 |
| 5-layer MoE | **0.19** | 0.005 |

This is why the scenario needs the **dense** model: on the truncated MoE the uncalibrated
logits and router near-ties amplify ulp drift into near-wholesale divergence (0.19), which
is not separable from the unrelated-content regime, while the dense model's 0.63 sits two
orders of magnitude above it. The scenario passes
`--ci-inject-rollout-data-min-match-ratio 0.5` — below the legitimate 0.63
and far above what any gross weight corruption can produce.

### `scenario_deterministic`

Multi-phase comparison test. Verifies healing state transfer is **bitwise** correct, in
both start regimes: cold start (phase_a) and resume from a post-healing checkpoint
(phase_b).

```
Type: comparison, multi-phase (phase_a + phase_b)
One shared builder parameterized by the phase's start rollout id P emits both phases: 3
rollouts, stop/start healing at the same relative offset, ckpt save after every rollout
(--save-interval 1). Only the start regime differs:
  phase_a: cold start (no --load, start_rollout_id=0) — rollouts 0..2 (P=0)
  phase_b: resumes from phase_a's last (rollout-2, post-healing) ckpt
           (start_rollout_id = loaded + 1 = 3) — rollouts 3..5 (P=3)
--num-rollout is 6 for both phases (exclusive end rollout id, not a per-run count); each
phase stops after its 3 rollouts via --debug-exit-after-rollout 3, which counts rollouts
executed within the run and fires after that rollout's ckpt save.
Comparison: BOTH phases' dumps rel <= 0 (bitwise), metrics rtol=0 / atol=0 (exact)

Per-phase target timeline:
  1. Rollout P, P+1: all N cells normal
  2. After rollout P+1: stop_cell_at_end(last) + start_cell_at_end(last) — trigger healing
  3. Rollout P+2: healing at start (recv_ckpt from cell_0), then normal execution
     (P+2 must exist, otherwise healing never executes)

phase_a exercises healing on a cold-started run (no --load sets
no_load_optim/no_load_rng/finetune); phase_b exercises it after checkpoint resume and —
since it must reproduce the baseline bit-for-bit — also proves phase_a's post-healing
checkpoint round-trips bitwise.

Both baseline and target use --deterministic-mode + env vars (NCCL_ALGO=Ring,
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, CUBLAS_WORKSPACE_CONFIG=:4096:8) for kernel
determinism, plus --debug-deterministic-collective so every order-sensitive SUM
collective uses a fixed-tree fold and the different reduction topologies of normal DP
(baseline) and indep_dp (target) become bitwise-comparable. Together they make the run
fully deterministic, so healing must reproduce the no-fault baseline bit-for-bit. A
state-copy bug is easy to make and an approximate check would miss it, hence zero tolerance.

Bitwise verification: --use-fault-tolerance --ft-components train auto-enables
--save-local-weight-checksum and --enable-event-analyzer. The event_analyzer
cross_replica_weight_checksum rule checks cell-to-cell bitwise equality after healing.

Healing witness: each target phase heals once, so each target event dir must contain
exactly one CellReconfigureEvent — a healing at rollout P+2 (healed = last cell, ckpt src =
cell 0, alive back to N; the stop+start pair is absorbed by a single _refresh_cells, so
there is no standalone shrink). Concretely the global rollout ids are phase_a: heal at 2;
phase_b: heal at 5. Both baseline event dirs must contain zero reconfigure events. This is
the regression gate for the off-by-one class of bugs where healing silently never runs and
the comparison passes on fault-free runs.
```

### `scenario_ft_random`

Non-comparison soak test. Verifies the system survives random crashes without hanging.

```
Type: non-comparison (no baseline, no compare)
Steps: 30 (default), configurable via --num-steps

Architecture (external fault injection, not inside training loop):
  1. Start training with indep_dp + control server (port 18080) + mini FT controller
  2. Start a background daemon thread that:
     a. Sleeps a random interval (exponential distribution, mean ~60s / crash_probability)
     b. GET /api/v1/cells — discover alive cells
     c. Pick a random alive cell (skip if <=1 alive)
     d. POST /api/v1/cells/{name}/inject-fault with random failure mode
     e. Repeat until training finishes
  3. The actor's inject_fault() runs in a dedicated ray concurrency group thread
     and kills the process immediately (SIGKILL, os._exit, or segfault)
  4. Health checker detects dead actor via heartbeat timeout
  5. Mini FT controller auto-recovers (suspend → resume)
  6. Verify: training completes, no hangs, prod assertions pass
  7. Healing witness: the injector must report >=1 accepted injection (else the soak
     proved nothing), and the event dir must contain >=1 healing CellReconfigureEvent.
     Faults are random, so neither an exact sequence nor the end-state membership is
     asserted — the witness only proves a fault was injected and healing actually ran.

CLI options: --seed (default 42), --num-steps (default 30), --crash-probability (default 0.1)
```

### `scenario_realistic_gsm8k`

Non-comparison accuracy test (entry `test_trainer_ft_realistic_gsm8k.py`, no mode
variants). Runs the same external fault injection as `scenario_ft_random` (shared
machinery in `conftest_ft/fault_injection.py`) over the real gsm8k RL recipe of
`tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py` (whose regular CI runs serve as the no-fault
reference wandb curves) and additionally asserts accuracy — i.e. fault recovery preserves
end-to-end learning, which the comparison scenarios cannot observe (e.g. whether engines
receive correct weights after recovery, and whether the post-fault on-policy loop still
improves accuracy).

```
Type: non-comparison (no baseline run; reference = the baseline test's wandb curves)
Recipe: Qwen2.5-0.5B, GRPO, 250 rollouts; parallelism mirrors dp2_cp2_real_rollout
        (2 cells x CP2 on 4 train GPUs + 4 rollout engines x 1 GPU, disaggregated)
Faults: same external random injection loop as scenario_ft_random
        (train cells via control server)

Assertion: --ci-metric-checker-key eval/gsm8k with a threshold that must stay
  identical to the no-fault baseline's (0.55): fault recovery must not cost
  accuracy. The checker passes if ANY eval reaches the threshold.

CLI options: --seed (default 42), --num-rollout (default 250),
  --crash-probability (default 0.1), --metric-threshold (default 0.55)
```
