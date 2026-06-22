# Fault Tolerance E2E Tests

## Layout

- Scenario logic lives in `conftest_ft/scenario_<name>.py`.
- CI runs it via thin per-mode entry files `test_trainer_ft_<scenario>_<mode>.py`, each registered with `register_cuda_ci(est_time=..., suite="stage-c-8-gpu-h100", labels=["ft-short"])` (comparison scenarios) or `labels=["ft-long"]` (soak scenarios).
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

- Gated on the `run-ci-ft-short` / `run-ci-ft-long` PR labels (FT is expensive — not run on every PR). `ft-short` covers the comparison scenarios (no_failure / deterministic / with_failure, minutes each); `ft-long` covers the soak scenarios (random-crash survival, realistic-gsm8k convergence — tens of minutes to hours). With a label set, the matching entries run on `stage-c-8-gpu-h100`.
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

