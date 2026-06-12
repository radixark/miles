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
| `scenario_with_failure` | Comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `scenario_deterministic` | Comparison, multi-phase | healing state transfer is bitwise-correct (stop+start) |
| `scenario_ft_random` | Non-comparison | system survives random crashes without hanging |

## Mode Variants

Each scenario runs with a `--mode`:

All modes are **disaggregated** (training and rollout on separate nodes). Modes without rollout use debug rollout data.

| Mode | Nodes | DP cells | Batch | Parallelism | Rollout | Model | Coverage |
|------|-------|----------|-------|-------------|---------|-------|----------|
| `dp2_cp2_tp2_ep2` | 1 | 2 | 3 | CP2 TP2 EP2 | debug data | 5-layer MoE | TP + EP |
| `dp2_cp2_pp2` | 1 | 2 | 3 | CP2 PP2 | debug data | 5-layer MoE | PP |
| `dp4_cp2` | 1 | 4 | 5 | CP2 | debug data | 5-layer MoE | Multi-replica (>=4 cells) |
| `dp2_cp2_real_rollout` | 1 | 2 | 3 | CP2 | 4 engines × 1 GPU | 5-layer MoE | Real rollout engines + weight update path |
| `6node_dp4_cp2_tp2_pp2_ep2_etp2` | 4+2 | 4 | 5 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | full MoE | Large-scale, all parallelism |

Batch sizes are deliberately **not** divisible by num_cells to test uneven sample distribution across replicas (e.g. DP4 + batch 5 → 2,1,1,1).

The 1-node modes use the truncated 5-layer MoE model (`Qwen3-30B-A3B-5layer`).

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
   `--crash-probability`. Each scenario's modes, phases, and knobs are in Mode Variants and
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

Multi-phase comparison test. Verifies indep_dp matches normal DP after fault + checkpoint resume.

```
Type: comparison, multi-phase (phase_a + phase_b)
Phase A steps: 1, Phase B steps: 4, metrics rtol: 5e-2

Phase A (both baseline and target):
  1. Run 1 step of training
  2. Save checkpoint (--save-interval 1)

Phase B — baseline:
  1. Resume from phase_a checkpoint
  2. Run 4 normal steps

Phase B — target:
  1. Resume from phase_a checkpoint
  2. Rollout 1: N cells normal
  3. Rollout 2, attempt 0: crash_before_allreduce on last cell rank 0
     → os._exit(1) → allreduce timeout → should_commit=false → retry
  4. Rollout 2, attempt 1: _refresh_cells() reconfigure → N-1 cells → commit
  5. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last)
  6. Rollout 3: _refresh_cells() healing → N cells
  7. Rollout 4: N cells stable

Compare: phase_b dumps (rel <= 0.0085; MoE expert grads and QK-norm grads also
tolerate max_abs <= 1e-3) and metrics (rtol=5e-2).

Fault injection via --ci-ft-test-actions JSON (data-driven, executed by RayTrainGroup).
The JSON `at_rollout` field specifies which rollout_id triggers the action.
The `attempt` field (for actor-level actions like `crash_before_allreduce`) specifies which retry attempt to match.
```

The `dp2_cp2_real_rollout` mode runs this scenario with live generation (real sglang
engines, deterministic inference, temperature 0.8), and the target's **post-fault rollouts
inject the baseline's recorded rollout data** (`--ci-inject-rollout-data-path` pointing at
the baseline phase_b's `--save-debug-rollout-data` output, start id = crash rollout + 1).
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
crash→retry→heal→weight-sync path. What this scenario deliberately does not assert: the
content the engines generate post-fault (the injected data replaces it). Pre-fault rollouts
(up to and including the crash rollout, whose data is generated before the crash and
redriven by the retry) are not injected — they remain a real sampled-data comparison.

### `scenario_deterministic`

Multi-phase comparison test. Verifies healing state transfer is **bitwise** correct.

```
Type: comparison, multi-phase (phase_a + phase_b)
Phase A steps: 1, Phase B steps: 3
Comparison: dump rel <= 0 (bitwise), metrics rtol=0 / atol=0 (exact)

Phase A: same as with_failure (1 step + save ckpt).

Phase B — target timeline:
  1. Rollout 1, 2: all N cells normal (2 good steps, accumulate meaningful state)
  2. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last) — trigger healing
  3. Rollout 3: healing at start (recv_ckpt from cell_0), then normal execution

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

CLI options: --seed (default 42), --num-steps (default 30), --crash-probability (default 0.1)
```
