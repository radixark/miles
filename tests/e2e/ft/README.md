# Fault Tolerance E2E Tests

## Layout

- Scenario logic lives in `conftest_ft/scenario_<name>.py`.
- CI runs it via thin per-mode entry files `test_trainer_ft_<scenario>_<mode>.py`, each registered with `register_cuda_ci(est_time=..., suite="stage-c-8-gpu-h200", labels=["ft-short"])` (comparison scenarios) or `labels=["ft-long"]` (soak scenarios).
- The CUDA CI runner executes each entry as bare `python3 <file>` (exit code = pass/fail); the entry just calls the scenario's `run_ci(mode)`.

| Scenario (`conftest_ft/scenario_*.py`) | Type | What it verifies |
|------|------|-----------------|
| `scenario_no_failure` | Comparison | indep_dp matches normal DP when no faults |
| `scenario_with_failure` | Comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `scenario_deterministic` | Comparison, multi-phase | healing state transfer is bitwise-correct (stop+start), on cold start and on resume from a post-healing ckpt |
| `scenario_ft_random` | Non-comparison | system survives random crashes without hanging |
| `scenario_realistic_gsm8k` | Non-comparison | model still reaches gsm8k accuracy under random crashes |

## Mode Variants

- Each scenario runs with a `--mode`.
- All modes are **disaggregated** (training and rollout on separate nodes). Modes without rollout use debug rollout data.

| Mode | Nodes | DP cells | Parallelism | Rollout | Model | Coverage |
|------|-------|----------|-------------|---------|-------|----------|
| `dp2_cp2_tp2_ep2` | 1 | 2 | CP2 TP2 EP2 | debug data | 5-layer MoE | TP + EP |
| `dp2_cp2_pp2` | 1 | 2 | CP2 PP2 | debug data | 5-layer MoE | PP |
| `dp4_cp2` | 1 | 4 | CP2 | debug data | 5-layer MoE | Multi-replica (>=4 cells) |
| `dp2_cp2_real_rollout` | 1 | 2 | CP2 | 4 engines × 1 GPU | 5-layer MoE | Real rollout engines + weight update path (no_failure, deterministic) |
| `dp2_cp2_real_rollout_dense` | 1 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | Real rollout under a fault + injection match guard (with_failure) |
| `6node_dp4_cp2_tp2_pp2_ep2_etp2` | 4+2 | 4 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | full MoE | Large-scale, all parallelism |

- All scenarios use `--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256` (256 samples/rollout), which divides evenly across both 2 and 4 cells. Uneven sample distribution across replicas is **not** exercised.
- 1-node modes use the 5-layer MoE (`Qwen3-30B-A3B-5layer`), except `dp2_cp2_real_rollout_dense` (dense `Qwen3-0.6B` — see `scenario_with_failure` for why).
- Authorized CI skips (no entry file): `6node_dp4_cp2_tp2_pp2_ep2_etp2` (multi-node), `with_failure × dp4_cp2`.

## Running

### In CI

- Gated on the `run-ci-ft-short` / `run-ci-ft-long` PR labels (FT is expensive — not run on every PR). `ft-short` covers the comparison scenarios (no_failure / deterministic / with_failure, minutes each); `ft-long` covers the soak scenarios (random-crash survival, realistic-gsm8k convergence — tens of minutes to hours). With a label set, the matching entries run on `stage-c-8-gpu-h200`.
- Add a `(scenario, mode)` to CI: copy an entry file, change `run_ci(...)`'s mode.
- Add a new label: edit `tests/ci/labels.py` and create the matching `run-ci-<label>` GitHub label.

### Manually

Set `PYTHONPATH` to the repo root (CI sets it automatically).

- One mode, exactly as CI runs it — invoke the entry file:

  ```bash
  PYTHONPATH=. python tests/e2e/ft/test_trainer_ft_no_failure_dp2_cp2_tp2_ep2.py
  ```

- Any mode (incl. authorized-skips) — invoke the scenario's typer app:

  ```bash
  PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_<name>.py run --mode <mode>
  ```

  | subcommand | does |
  |---|---|
  | `run` | full pipeline: prepare + baseline + target + compare |
  | `baseline` / `target` | run one side only (debugging) |
  | `compare` | re-run comparison on existing dumps (no GPU) |

- When debugging, prefer the individual subcommands (shared `--dump-dir`, `--phase` for multi-phase) over `run`, so you re-run only what changed (e.g. just `compare` on existing dumps, or one side / phase).
- `scenario_ft_random`: non-comparison; only `run` with `--seed` / `--num-steps` / `--crash-probability`.
- `scenario_realistic_gsm8k`: non-comparison, no `--mode`; only `run` with `--seed` / `--num-rollout` / `--crash-probability` / `--metric-threshold`.
- Dumps land under `/node_public/dumps/<test_name>/` (`conftest_ft/app.py` `resolve_dump_dir`).

## Comparison criterion

- Dumps: per-tensor boolean predicates over `rel`/`max_abs`/`mean_abs` (`compare_dumps(diff_thresholds=[(name_regex, predicate), ...])`).
- `scenario_deterministic`: bitwise (`rel <= 0`), relying on `--deterministic-mode` (kernel determinism) + `--debug-deterministic-collective` (fixed-tree SUM collectives).
- Metrics: `rtol=atol=0`, except `train/grad_norm` (`rtol<=1e-6`): its bracketing depends on dist-optimizer shard count (8 flat vs 2 per cell), so a few fp32 ulps are inherent; the grads stay bitwise-checked via the dumps.
- Other scenarios: `rel <= 0.0085`; `with_failure` also floors near-zero MoE-expert and QK-norm (`q_layernorm`/`k_layernorm`) grads at `max_abs <= 1e-3`.
- Unmatched tensors are a fail-closed error — end each list with a `.*` catch-all.
- Exact per-scenario thresholds: Test Definitions below.

## Debug Rollout Data

- Modes without rollout engines (`has_real_rollout == False`) use pre-recorded data via `--load-debug-rollout-data --debug-train-only`.
- `conftest_ft/execution.py` `prepare()` downloads it via `U.hf_download_dataset()`.

### How to regenerate

- **Must** use the 5-layer model (the full model produces `rollout_log_probs` incompatible with the 5-layer training model → NaN gradients in GRPO).

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

```
Type: comparison, multi-phase (phase_a + phase_b)
Phase A steps: 1, Phase B steps: 3 (rollouts 1..3; --num-rollout 4 resumed from the
rollout-0 checkpoint), metrics rtol: 5e-2

Phase A (both baseline and target):
  1. Run 1 step of training
  2. Save checkpoint (--save-interval 1)

Phase B — baseline:
  1. Resume from phase_a checkpoint
  2. Run 3 normal steps (rollouts 1..3)

Phase B — target:
  1. Resume from phase_a checkpoint
  2. Rollout 1: N cells normal
  3. Rollout 2, attempt 0: crash_before_allreduce on last cell rank 0
     → os._exit(1) → allreduce timeout → should_commit=false → retry
  4. Rollout 2, attempt 1: _refresh_cells() reconfigure → N-1 cells → commit
  5. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last)
  6. Rollout 3: _refresh_cells() healing → N cells, trains with the healed cell

Compare: phase_b dumps per rollout (rel <= 0.0085; MoE expert grads and QK-norm grads
also tolerate max_abs <= 1e-3; in the real_rollout mode the post-fault/injected rollouts'
grads tolerate max_abs <= 3e-3 — see the dense-mode section below) and metrics (rtol=5e-2).

Healing witness: the target phase_b event dir must contain exactly two
CellReconfigureEvents, in order — a shrink at rollout 2 (alive N -> N-1, positive proof
the fault injection fired) and a healing at rollout 3 (healed = last cell, ckpt src =
cell 0, alive back to N). Baseline and phase_a event dirs must contain zero reconfigure
events. This positively proves the crash -> shrink -> heal path executed; without it the
comparison could silently degenerate to two fault-free runs.

Fault injection via --ci-ft-test-actions JSON (data-driven, executed by RayTrainGroup).
The JSON `at_rollout` field specifies which rollout_id triggers the action.
The `attempt` field (for actor-level actions like `crash_before_allreduce`) specifies which retry attempt to match.
```

#### `dp2_cp2_real_rollout_dense` mode

Runs `scenario_with_failure` with live generation (real sglang engines, deterministic inference, temperature 0.8).

- Post-fault rollouts **inject the baseline's recorded rollout data** (`--ci-inject-rollout-data-path` → baseline phase_b's `--save-debug-rollout-data`, start id = crash rollout + 1).
- Why inject: the degraded-quorum commit accumulates microbatches in a different fp bracketing than the fault-free side — a fault-inherent ulp diff no collective ordering removes. Under live sampling it flips sampled tokens, after which the two runs' rollout data diverges wholesale, so a strict vs-baseline comparison of real-sampled post-fault rollouts is ill-posed. Injection makes training inputs identical by construction → full strict comparison, zero relaxation.
- Stays real on the target: engines + generation (samples discarded), `update_weights` after the degraded commit and after healing, health-monitor pause/resume — the whole crash→retry→heal→weight-sync path.
- Post-healing `update_weights` is consumed: real_rollout asserts the target pushed bitwise-identical engine weights to the baseline (see inference engine weight checksum).
- Injected rollouts' dump comparison floors `max_abs <= 3e-3` on the **noisy grad families only** (decoder-layer QK-norms, folded `layer_norm_weight`s, attn/MLP matrices): training data is bitwise-identical, but target weights carry the degraded commit's ulp drift, landing as ≤2.8e-3 absolute noise in those near-zero grads while real grads sit ~1e-2 (40 tensors, 2026-06-12; same argument as the 1e-3 QK-norm floor, recalibrated for the dense model). Embedding/output/final-norm grads, all activations, and all pre-fault rollouts keep the strict set.
- Generation is still asserted: each injected rollout checks generated responses match the recording at a mean per-token ratio above threshold with bitwise-identical prompts (`RolloutDataInjectionUtil.assert_matches_generated`). Gross weight bugs (e.g. broken `update_weights`) drop the ratio ~2 orders → still fail. Exact post-fault sampled content beyond the ratio is not asserted. Pre-fault rollouts are not injected (real comparison).

Guard calibration (2026-06-12, first post-fault rollout, 256 samples, correct weights; metric counts everything after a response's first flipped token as mismatched):

| Model | mean response-token match | min |
|-------|---------------------------|-----|
| dense Qwen3-0.6B | **0.63** | 0.035 |
| 5-layer MoE | **0.19** | 0.005 |

- Needs the **dense** model: on the truncated MoE, uncalibrated logits + router near-ties amplify ulp drift to near-wholesale divergence (0.19, not separable from the unrelated-content regime); dense's 0.63 sits 2 orders above. Scenario uses `--ci-inject-rollout-data-min-match-ratio 0.5` (below the legitimate 0.63, far above any gross corruption).

### `scenario_deterministic`

```
Type: comparison, multi-phase (phase_a + phase_b)
One shared builder parameterized by the phase's start rollout id P emits both phases: 3
rollouts, stop/start healing at the same relative offset, ckpt saved only at the phase's
last rollout (--save-interval 3 = NUM_ROLLOUTS_PER_PHASE). Only the start regime differs:
  phase_a: cold start (no --load, start_rollout_id=0) — rollouts 0..2 (P=0)
  phase_b: resumes from phase_a's last (rollout-2, post-healing) ckpt
           (start_rollout_id = loaded + 1 = 3) — rollouts 3..5 (P=3)
--num-rollout is 6 (exclusive end rollout id, not a per-run count); each phase stops after
3 rollouts via --debug-exit-after-rollout 3, which counts rollouts within the run and fires
after that rollout's ckpt save.
Comparison: BOTH phases' dumps rel <= 0 (bitwise), metrics rtol=0 / atol=0 (exact)

Per-phase baseline timeline: rollouts P..P+2 all normal (normal DP, no stop/start, no
healing) — the no-fault reference the target must reproduce bit-for-bit.

Per-phase target timeline:
  1. Rollout P, P+1: all N cells normal
  2. After rollout P+1: stop_cell_at_end(last) + start_cell_at_end(last) — trigger healing
  3. Rollout P+2: healing at start (recv_ckpt from cell_0), then normal execution
     (P+2 must exist, otherwise healing never executes)

phase_a exercises healing on a cold-started run (no --load sets
no_load_optim/no_load_rng/finetune); phase_b exercises it after resume and — reproducing
the baseline bit-for-bit — also proves phase_a's post-healing ckpt round-trips bitwise.

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

Inference engine weight checksum (real_rollout mode only): each update_weights logs one
InferenceEngineWeightChecksumEvent per rollout (all engines). _compare asserts per phase that baseline
and target pushed bitwise-identical weights for every (rollout, engine) pair; the
event_analyzer inference_engine_weight_checksum_consistency rule independently checks that all engines
of a rollout agree (the production-facing function A).

Healing witness: each target phase heals once, so each target event dir must contain
exactly one CellReconfigureEvent — a healing at rollout P+2 (healed = last cell, ckpt src =
cell 0, alive back to N; the stop+start pair is absorbed by a single _refresh_cells, so
there is no standalone shrink). Global ids: phase_a heal 2; phase_b heal 5. Both baseline
event dirs must contain zero reconfigure events. This is the regression gate for the
off-by-one class of bugs where healing silently never runs and the comparison passes on
fault-free runs.
```

### `scenario_ft_random`

```
Type: non-comparison (no baseline, no compare)
Steps: 30 (default), configurable via --num-steps

Architecture (external fault injection, not inside training loop):
  1. Start training with indep_dp + control server (port 18080) + mini FT controller
  2. Start a background daemon thread that:
     a. Sleeps a random interval (exponential, mean = 60s / crash_probability ≈ 120s at the default)
     b. GET /api/v1/cells — read each cell's Healthy condition
     c. Count the genuinely-alive cells — reported Healthy, minus cells we injected that have
        not finished a down->up recovery (RecoveryGate) — and skip if injecting would leave
        <=1 of them. The control server reports a just-killed cell Healthy for ~95s >> the
        inject interval, so excluding still-recovering cells is what keeps >=1 live replica
        (indep_dp cannot heal from zero survivors).
     d. Otherwise POST /api/v1/cells/{name}/inject-fault with a random failure mode
     e. Repeat until training finishes
  3. The actor's inject_fault() runs in a dedicated ray concurrency group thread
     and kills the process immediately (SIGKILL, os._exit, or segfault)
  4. Health checker detects dead actor via heartbeat timeout
  5. Mini FT controller auto-recovers (suspend → resume)
  6. Verify: training completes, no hangs, prod assertions pass
  7. Healing witness: the injector must report >=2 accepted injections (a single heal could
     be a fluke) and the event dir must contain >=2 healing CellReconfigureEvents. The default
     --crash-probability is set high enough that the soak reliably clears this floor. Faults are
     random, so neither an exact sequence nor the end-state membership is asserted — the
     witness only proves repeated faults were injected and healing actually ran.

CLI options: --seed (default 42), --num-steps (default 30), --crash-probability (default 0.5)
```

### `scenario_realistic_gsm8k`

Entry `test_trainer_ft_realistic_gsm8k.py`, no mode variants. Runs the same external fault injection as `scenario_ft_random` (shared `conftest_ft/fault_injection.py`) over the real gsm8k RL recipe of `tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py` (its regular CI runs are the no-fault reference wandb curves), and additionally asserts accuracy — i.e. fault recovery preserves end-to-end learning, which the comparison scenarios cannot observe.

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
