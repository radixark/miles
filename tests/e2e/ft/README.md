# Fault Tolerance E2E Tests

## Overview Table

### CI Entries

- **CI entry files**: `test_<TEST_NAME>__<mode>.py`, or `test_<TEST_NAME>__<kill>.py` when the scenario pins its own topology and takes no mode; split on the first `__` to read the scenario and the rest back out, which is why no scenario name contains one.
- **The segment after the scenario is always the kill segment**: a mode name starts with it, and an entry with no mode carries it alone, so every entry says what its run crashes without anyone opening the file.
- **Entry file content**: `register_cuda_ci(est_time=..., suite=..., labels=[...])` plus `run_ci(_MODE)` under `__main__`, no test logic.
- **Execution model**: bare `python3 <file>` from the repo root, exit code = pass/fail (`tests/ci/ci_utils.py` `run_unittest_files`).

| Scenario | Modes with an entry file |
| --- | --- |
| `scenario_trainer_no_failure` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer` |
| `scenario_trainer_deterministic` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer` |
| `scenario_trainer_with_failure` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2` |
| `scenario_random_crash` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer`, `kill_rollout__dp2_cp2__colocate` |
| `scenario_realistic_gsm8k` | `test_realistic_gsm8k__kill_train.py`, no modes |

- **Forced absences**: `kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full` is multi-node with no CI lane; `kill_rollout__dp2_cp2__colocate` only fits `scenario_random_crash`, the one scenario that crashes engines; `kill_train__dp2_cp2` supersedes `kill_train__dp2_cp2__moe_5layer` in `scenario_trainer_with_failure`. `scenario_trainer_with_failure` x `kill_train__dp4_cp2__fake_rollout__moe_5layer` is an authorized skip. Every other absence is an unclaimed cell, not a decision — adding an entry file is all it takes.

### Scenarios

- **Scenario logic**: `conftest_ft/scenario_<name>.py` — a typer app plus a `run_ci(mode)` runner.

| Scenario (`conftest_ft/scenario_*.py`) | Type | What it verifies |
| --- | --- | --- |
| `scenario_trainer_no_failure` | comparison | indep_dp matches normal DP when no faults |
| `scenario_trainer_with_failure` | comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `scenario_trainer_deterministic` | comparison, multi-phase | healing state transfer is bitwise-correct, on cold start and on resume from a post-healing ckpt |
| `scenario_random_crash` | soak | system survives random crashes without hanging |
| `scenario_realistic_gsm8k` | soak | model still reaches gsm8k accuracy under random crashes |

### Modes

- **Selection**: `--mode`, defined in `conftest_ft/modes.py`; `scenario_realistic_gsm8k` takes none.
- **Mode names**: `<kill>__<parallelism>[__fake_rollout][__moe_5layer|__moe_full][__colocate]`, segments separated by `__` and joined by `_` inside a segment.
- **What a name carries**: the `kill` segment always, then only the axes that differ from the naming defaults — real sglang engines, the dense `Qwen3-0.6B`, disaggregated placement. Node counts, engine counts and cell counts are never in the name; read them from the table below.
- **Why `kill` leads**: what a run crashes is the subject of this suite, so it is the first thing the name answers, and it is a property of the mode alone — no scenario widens it at runtime.
- **Declared per mode**: cell count, parallelism, model, train/rollout GPU split, `colocate` (default disaggregated, i.e. training and rollout on separate nodes), `ft_components` (default `("train",)`).
- **No rollout engines**: modes with `rollout_num_engines == 0` train on pre-recorded debug rollout data.

| Mode | Nodes | GPUs (train + rollout) | DP cells | Parallelism | Rollout | Model | `ft_components` | Why it exists |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 2 | CP2 TP2 EP2 | debug data | 5-layer MoE | `("train",)` | TP + EP coverage |
| `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 2 | CP2 PP2 | debug data | 5-layer MoE | `("train",)` | PP coverage, via `--decoder-first-pipeline-num-layers 3 --decoder-last-pipeline-num-layers 2` |
| `kill_train__dp4_cp2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 4 | CP2 | debug data | 5-layer MoE | `("train",)` | multi-replica coverage (>= 4 cells) |
| `kill_train__dp2_cp2__moe_5layer` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | 5-layer MoE | `("train",)` | real engines + the weight-update path |
| `kill_train__dp2_cp2` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | `("train",)` | `scenario_trainer_with_failure` under real generation; needs the dense model (see below) |
| `kill_rollout__dp2_cp2__colocate` | 1 | 4 shared | 2 | CP2 | 4 engines × 1 GPU, colocated | dense Qwen3-0.6B | `("rollout",)` | the only rollout-only mode: crashes engines, not trainer cells |
| `kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full` | 4 train + 2 rollout | 32 + 16 | 4 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | full MoE | `("train",)` | full model, all parallelism; multi-node, so no CI entry |

- **Batch shape**: `--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256` everywhere — 256 samples per rollout, divisible by both 2 and 4 cells. Uneven distribution across replicas is **not** exercised.
- **Model**: 1-node modes use the 5-layer MoE `Qwen3-30B-A3B-5layer`, except the two dense modes.

## Running the code

### In CI

- **Gating labels**: `run-ci-ft-short` for the comparison scenarios (minutes each), `run-ci-ft-long` for the soaks (tens of minutes to hours). Nothing here runs on an unlabelled PR.
- **Broad scopes**: `run-ci-all` includes both; the nightly cadence includes `ft-short` but not `ft-long`; `run-ci-image` excludes both.
- **Suite**: `suite="stage-c-8-gpu-h200"`, run by the job of the same name in `.github/workflows/pr-test.yml`.
- **ft-long is disabled**: all four entries pass `disabled="FT soak tests pending CI infra support"`, and `tests/ci/run_suite.py` drops every test with a non-`None` `disabled`, so `run-ci-ft-long` executes nothing. Unblocked by an ft-long capable lane; nothing in the tests is known broken.
- **Fast-layer stand-in**: `tests/fast/e2e/ft/test_rollout_gated_recovery.py` covers suspend → gated relaunch → recovery on CPU meanwhile.
- **Add a `(scenario, mode)`**: copy an entry file, change `_MODE`.
- **Add a label**: an entry in `tests/ci/labels.py` plus the matching `run-ci-<key>` GitHub label; the workflow needs no edit.

### Manually

`PYTHONPATH` must point at the repo root (CI sets it automatically).

```bash
# One mode, exactly as CI runs it
PYTHONPATH=. python tests/e2e/ft/test_trainer_no_failure__kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer.py

# Any mode, including the ones with no entry file
PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py run --mode kill_train__dp4_cp2__fake_rollout__moe_5layer
```

| Subcommand | Does | Available in |
| --- | --- | --- |
| `run` | full pipeline: prepare + every phase's baseline/target + compare | all scenarios |
| `baseline` / `target` | one side only, for debugging | comparison scenarios |
| `compare` | re-run the comparison on existing dumps (no GPU) | comparison scenarios |
| `generate-data` | record debug rollout data with real engines, no dumper | comparison scenarios |

- **Debugging**: prefer the individual subcommands over `run` — with a shared `--dump-dir` (plus `--phase` when multi-phase) you re-run only what changed.
- **`scenario_random_crash`**: only `run`, with `--mode` / `--seed` / `--num-steps` / `--crash-interval-seconds`.
- **`scenario_realistic_gsm8k`**: only `run`, with `--seed` / `--num-rollout` / `--crash-interval-seconds` / `--metric-threshold`; no `--mode`.
- **Dumps**: `/node_public/dumps/<test_name>/` via `resolve_dump_dir` in `conftest_ft/app.py`, deleted at the end of `run`.

### Generate Debug Rollout Data

- **Who uses it**: modes with `has_real_rollout == False`, through `--load-debug-rollout-data --debug-train-only`.
- **Where it comes from**: `prepare()` in `conftest_ft/execution.py`, via `U.hf_download_dataset()` on `fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer`.
- **Soak reuse**: `materialize_cyclic_debug_rollout_data()` symlinks the recorded files cyclically, so a soak can run more steps than were recorded.
- **Regenerating it needs the 5-layer model**: the full model's `rollout_log_probs` are incompatible with the 5-layer training model and produce NaN GRPO gradients.

```bash
# 1. Generate with the 5-layer model and real sglang engines (no dumper)
PYTHONPATH=. python tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py generate-data \
    --mode kill_train__dp2_cp2__moe_5layer --num-steps 12 --output-dir /tmp/gen_rollout

# 2. Inspect
ls /tmp/gen_rollout/rollout_data/

# 3. Upload
hf upload --repo-type dataset fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer \
    /tmp/gen_rollout/rollout_data/
```

## Test Specifications

### Comparison Criterion

- **Dumps**: per-tensor predicates over `rel` / `max_abs` / `mean_abs`, as `compare_dumps(diff_thresholds=[(name_regex, predicate), ...])` onto the sglang comparator's `--diff-threshold`.
- **Fail-closed**: a tensor matching no regex fails, so every list ends with a `.*` catch-all and the specific families come first.
- **Model inputs**: `INPUT_TENSORS_ALLOW_FAILED_PATTERN` exempts `input_ids`, `positions`, `cu_seqlens_*`, `qkv_format`; `INPUT_TENSORS_SKIP_PATTERN` skips those plus `.*witness.*`. Nothing else is exempt.
- **Metrics**: `compare_metrics` reads `MetricEvent`s, requires `train/grad_norm` and `train/loss` in the baseline and equal event counts on both sides, and compares only the highest-attempt event per rollout id.


- **Why only some are bitwise**: baseline and target reduce over different topologies, so allreduce kernel ordering differs — unless `--deterministic-mode` and `--debug-deterministic-collective` are on.
- **Why `train/grad_norm` is exempt in `scenario_trainer_deterministic`**: it sums squared shard fragments, so its bracketing follows the dist-optimizer shard count (8 flat vs 2 per cell); a few fp32 ulps are inherent. The grads stay bitwise-checked through the dumps.

### `scenario_trainer_no_failure`

```
Type: comparison (baseline=normal DP, target=indep_dp)
Steps: 2 (NUM_STEPS)
Compare: dumps rel <= 0.0085; metrics rtol=1e-2, atol=1e-8

1. Baseline: normal DP on debug rollout data (real engines in a real-rollout mode)
2. Target: the same arguments plus --use-fault-tolerance
3. Compare:
   - Tensor-level: compare_dumps (weights, grads via dumper & sglang comparator)
   - Metric-level: compare_metrics (MetricEvent, requires train/grad_norm and train/loss)
   - Rank matching: grouping_skip_keys=["rank", "dp", "edp"], the two sides differing in
     world size and DP layout

Roughly equal, not bitwise - allreduce kernel ordering differs across topologies.
```

### `scenario_trainer_with_failure`

```
Type: comparison, multi-phase (phase_a + phase_b)
Steps: phase_a 1 rollout (id 0), phase_b 3 rollouts (ids 1..3)
  --num-rollout 4: exclusive global end id, not a per-run count
Compare: phase_b dumps per rollout, rel <= 0.0085 plus the max_abs floors below;
         metrics rtol=5e-2, atol=1e-7

Phase A (both sides):
  1. Run 1 rollout
  2. Save checkpoint (--save-interval 1), exit

Phase B - baseline:
  1. Resume from the phase_a checkpoint
  2. Run 3 normal rollouts (1..3)

Phase B - target:
  1. Resume from the phase_a checkpoint
  2. Rollout 1: N cells normal
  3. Rollout 2, attempt 0: crash_before_allreduce on last cell rank 0
     -> os._exit(1) -> allreduce timeout -> should_commit=false -> retry
  4. Rollout 2, attempt 1: reconfigure to N-1 cells, commit on the degraded quorum
  5. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last)
  6. Rollout 3: heal back to N cells, train with the healed cell

Fault injection: --ci-ft-test-actions, JSON list of {at_rollout, action, cell_id, rank, attempt}
  at_rollout: rollout id; attempt: retry attempt, actor-level actions only
  stop_cell_at_end / start_cell_at_end: trainer controller, suspend/resume via cell_operations
  crash_before_allreduce: inside the targeted actor

Healing witness: target phase_b event dir, exactly two CellReconfigureEvents
  rollout 2: shrink, alive N -> N-1
  rollout 3: heal, healed = last cell, ckpt src = cell 0, alive back to N
  baseline and phase_a dirs: zero
Dump-leaf witness: {fwd_bwd/rollout_<id> leaf dirs} == {rollouts the comparison loop walks}
```

- **Why the healing witness**: without it the comparison degenerates into two fault-free runs that trivially agree; the shrink proves the injection fired.
- **Why the dump-leaf witness**: a newly added leaf dir would otherwise skip comparison unnoticed.

Grad families with a `max_abs` floor (cancellation-dominated near-zero grads; real grads sit around `1e-2`):

| Rollouts | Families | Floor |
| --- | --- | --- |
| all | MoE expert grads, QK-norm (`q_layernorm` / `k_layernorm`) grads | `max_abs <= 1e-3` |
| injected ones, real-rollout mode only | QK-norms, folded `layer_norm_weight`s, `linear_qkv` / `linear_proj` / `mlp.linear_fc[12]` weights | `max_abs <= 3e-3` |

- **Where `3e-3` comes from**: the degraded commit's ulp drift lands as <= 2.8e-3 absolute noise in those near-zero grads (40 tensors, 2026-06-12), against real grads around `1e-2`. Embedding, output, final-norm grads, every activation and every pre-fault rollout keep the strict set.

#### `kill_train__dp2_cp2` mode

`scenario_trainer_with_failure` against live generation: real sglang engines, deterministic inference, temperature 0.8.

- **Post-fault rollouts are injected**: `--ci-inject-rollout-data-path` replays the baseline's `--save-debug-rollout-data` recording from rollout 3 on (crash rollout + 1).
- **Why inject**: the degraded-quorum commit brackets microbatch accumulation differently, and under live sampling that ulp diff flips tokens until the two runs' rollout data diverges wholesale. It is fault-inherent -- no collective ordering removes it. Injecting makes training inputs identical by construction, keeping the comparison strict.
- **The target stays real**: engines and generation still run (samples discarded), `update_weights` fires after the degraded commit and after healing, the health monitor pauses and resumes — the whole crash → retry → heal → weight-sync path. Engine checksums are not compared here; only `scenario_trainer_deterministic` does that.
- **Generation is still asserted**: `RolloutDataInjectionUtil.assert_matches_generated` requires bitwise-identical prompt tokens per sample, plus a mean response-token match ratio above `--ci-inject-rollout-data-min-match-ratio`, set to 0.5 here (the flag's own default is 0.9). A broken `update_weights` drops that ratio by ~2 orders.
- **Not asserted**: exact post-fault sampled content beyond the ratio; pre-fault rollouts are compared for real.

Guard calibration (2026-06-12, first post-fault rollout, 256 samples, correct weights; a response counts as mismatched from its first flipped token on):

| Model | Mean response-token match | Min |
| --- | --- | --- |
| dense Qwen3-0.6B | **0.63** | 0.035 |
| 5-layer MoE | **0.19** | 0.005 |

- **Why dense**: on the truncated MoE, uncalibrated logits plus router near-ties amplify the drift to 0.19, indistinguishable from unrelated content; dense's 0.63 sits 2 orders above that, so 0.5 separates them.

### `scenario_trainer_deterministic`

```
Type: comparison, multi-phase (phase_a + phase_b)
Steps: 3 rollouts per phase - phase_a 0..2, phase_b 3..5
  --num-rollout 6: exclusive global end id
  --debug-exit-after-rollout 3: counts within the run, fires after that rollout's ckpt save
  --save-interval 3 (NUM_ROLLOUTS_PER_PHASE): one ckpt at each phase's last rollout
Compare: BOTH phases' dumps rel <= 0 (bitwise); metrics rtol=0 / atol=0, except
         train/grad_norm at rtol=1e-6

One shared builder parameterized by the phase's start rollout id P; only the start regime differs:
  phase_a: cold start (no --load, so no_load_optim/no_load_rng/finetune) - rollouts 0..2 (P=0)
  phase_b: resumes from phase_a's post-healing rollout-2 ckpt (start_rollout_id = loaded + 1
           = 3) - rollouts 3..5 (P=3)

Per-phase baseline: rollouts P..P+2 all normal, no stop/start, no healing

Per-phase target:
  1. Rollout P, P+1: all N cells normal
  2. After rollout P+1: stop_cell_at_end(last) + start_cell_at_end(last)
  3. Rollout P+2: heal at the start (recv_ckpt from cell 0), then normal execution

Determinism: --deterministic-mode, plus NCCL_ALGO=Ring, NVTE_ALLOW_NONDETERMINISTIC_ALGO=0,
  CUBLAS_WORKSPACE_CONFIG=:4096:8, SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=8192
  --debug-deterministic-collective: fixed-tree SUM folds, making normal DP's and indep_dp's
    reduction topologies bitwise-comparable

Cross-cell check: --use-fault-tolerance --ft-components train auto-enables
  --save-local-weight-checksum and --enable-event-analyzer
  cross_replica_weight_checksum: cell-to-cell bitwise equality, every rollout attempt,
    post-healing included
Engine checksum (real-rollout modes only): one InferenceEngineWeightChecksumEvent per
  update_weights, carrying every engine's checksum
  _compare, per phase: baseline and target pushed identical weights per (rollout, engine)
  inference_engine_weight_checksum_consistency: all engines of one rollout agree

Healing witness: one heal per target phase, at P+2 (healed = last cell, ckpt src = cell 0,
  alive back to N); no standalone shrink - one _refresh_cells absorbs the stop+start pair
  the event dir is snapshotted into the ckpt and restored on --load, hence:
    target phase_a: heal at rollout 2
    target phase_b: heal at rollout 2 (restored with the ckpt) + heal at rollout 5
    both baselines: zero reconfigure events
```

- **Why P+2 must exist**: healing runs at its start, so a shorter phase never executes the path under test.
- **Why zero tolerance**: a state-copy bug in healing is easy to make and an approximate check would miss it.
- **What phase_b adds**: reproducing the baseline bit-for-bit also proves the ckpt round-trips bitwise.
- **Why the healing witness**: it gates the off-by-one bug where healing never runs and the comparison passes on two fault-free runs.

### `scenario_random_crash`

```
Type: soak (no baseline, no compare); passes if training completes without hanging and the
      witnesses hold
Steps: 30 (default)
CLI: --mode, --seed (42), --num-steps (30), --crash-interval-seconds (120)

Targeting and assertions follow the mode's ft_components:
  ("train",)          -> inject into "actor" cells only, assert healing CellReconfigureEvents
  ("rollout",)        -> inject into "rollout" cells only, assert the recovery cycle
  ("train","rollout") -> inject into every kind, assert both

Architecture (external fault injection, not inside the training loop):
  1. Start indep_dp training + api server (port 18080) + --mini-ft-controller-enable
  2. A background daemon thread iterates every 2s:
     a. GET /api/v1/cells, keeping only the targeted cell type
     b. Update the RecoveryGate and the recovery witness from that snapshot
     c. Stop here unless the next injection time has arrived
     d. Compute the genuinely-alive cells - reported Healthy, minus injected cells that have
        not completed a down -> up cycle - and defer unless a spare replica remains
     e. POST /api/v1/cells/{name}/inject-fault with a mode drawn from SIGKILL / EXIT /
        SEGFAULT, then draw the next injection time (exponential, mean
        crash_interval_seconds divided by the number of ft components)
  3. inject_fault() runs on the actor's own ray concurrency group thread and kills the process
  4. The health checker notices by heartbeat timeout
  5. The mini FT controller recovers it (suspend -> resume)
  6. Verify: training completes, no hangs, prod assertions pass

Witnesses:
  train ft   -> >= 2 accepted injections, >= 2 healing CellReconfigureEvents
  rollout ft -> every accepted rollout injection paired, per cell and in order, with one
                completed Serving -> (Suspended|Pending) -> Serving cycle; an injection still
                unpaired when training ends fails the soak

Faults are random, so neither an exact sequence nor the end-state membership is asserted.
```

- **Why only ft-enabled components are targeted**: injecting into a component whose ft is off would crash something nothing is watching.
- **Why a floor of two**: a single heal could be a fluke. The default `--crash-interval-seconds` is short enough that the soak reliably clears the floor.
- **Why still-recovering cells are excluded**: the api server reports a just-killed cell Healthy for ~95s, far longer than the poll interval, and indep_dp cannot heal from zero survivors, so a naive Healthy count would eventually kill the last replica.
- **Why `Serving`, not `Running`**: `Running` also covers a replacement that got weights but never re-entered the router. `Suspended` is not required in between — it lasts only `--mini-ft-controller-resume-delay` (10s), which a 2s poll can miss.
- **Why poll faster than injections**: a crash → detect → heal cycle completing between two sparse injections must be seen, or its cell stays excluded from the live set forever.

### `scenario_realistic_gsm8k`

```
Type: soak (no baseline run; reference = the baseline test's wandb curves)
Entry: test_realistic_gsm8k__kill_train.py, no mode variants
CLI: --seed (42), --num-rollout (250), --crash-interval-seconds (600), --metric-threshold (0.55);
     no --mode

Recipe: Qwen2.5-0.5B-Instruct, GRPO, 250 rollouts, over the gsm8k RL recipe of
        tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py, whose regular CI runs are the no-fault
        reference wandb curves
Layout: mirrors kill_train__dp2_cp2__moe_5layer - 2 cells x CP2 on 4 train GPUs + 4 rollout engines
        x 1 GPU, disaggregated
Faults: scenario_random_crash's external injection loop (shared conftest_ft/fault_injection/),
        against "actor" cells

Assertions:
  1. --ci-metric-checker-key eval/gsm8k against a threshold that must stay identical to the
     no-fault baseline's (0.55); passes if ANY eval reaches it
  2. assert_soak_reconfigure_events, shared with scenario_random_crash: >= 2 accepted injections
     and >= 2 healings

Fault recovery must not cost end-to-end learning, which the comparison scenarios cannot observe.
```
