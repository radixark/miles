# Fault Tolerance E2E Tests

## Layout

- Scenario logic lives in `conftest_ft/scenario_<name>.py`.
- CI runs it via thin per-mode entry files `test_trainer_ft_<scenario>_<mode>.py`, each registered with `register_cuda_ci(est_time=..., suite="stage-c-8-gpu-h200", labels=["ft"])`.
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

- Gated on the `run-ci-ft` PR label (FT is expensive — not run on every PR). With the label set, every entry runs on `stage-c-8-gpu-h200`.
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

