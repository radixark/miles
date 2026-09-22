# Fault Tolerance E2E Tests

## Overview Table

### CI Entries

- **CI entry files**: `test_<TEST_NAME>__<mode>.py`, or `test_<TEST_NAME>__<kill>.py` when the scenario pins its own topology and takes no mode; split on the first `__` to read the scenario and the rest back out, which is why no scenario name contains one.
- **The segment after the scenario is always the kill segment**: a mode name starts with it, and an entry with no mode carries it alone, so every entry says what its run crashes without anyone opening the file.
- **Entry file content**: `register_cuda_ci(est_time=..., suite=..., labels=[...], hardware=[...])` plus `run_ci(_MODE)` under `__main__`, no test logic.
- **Execution model**: bare `python3 <file>` from the repo root, exit code = pass/fail (`tests/ci/ci_utils.py` `run_unittest_files`).

| Scenario | Modes with an entry file |
| --- | --- |
| `scenario_trainer_no_failure` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer` |
| `scenario_trainer_deterministic` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer` |
| `scenario_trainer_with_failure` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer`, `kill_train__dp4_cp2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2` |
| `scenario_random_crash` | `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer`, `kill_train__dp2_cp2__moe_5layer`, `kill_train_rollout__dp2_cp2`, `kill_rollout__dp4` |

- **Forced absences**, one reason each:
    - `kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full` is multi-node, and no multi-node CI lane exists.
    - `kill_rollout__dp4__colocate` fits only the scenarios that crash engines.
    - `kill_train__dp2_cp2` supersedes `kill_train__dp2_cp2__moe_5layer` in `scenario_trainer_with_failure`.
- **Every other absence is an unclaimed cell**, not a decision — adding an entry file is all it takes.

### Scenarios

- **CI**: Ray random, deterministic-rollout and precise/mixed entries are enabled on `stage-c-8-gpu-h200` under `ft-long`. The workflow sets `MILES_TEST_DUMPS_ROOT=/data/miles_ci/dumps`, which must be an absolute path.
- **Kubernetes**: the same random FT scenario accepts a Kubernetes backend through the launch configuration. Kubernetes execution requires shared storage, worker images and release-management credentials; the Ray CI lane does not provide those resources. Hot restart remains Kubernetes-only.
- **Validation status**: registration is not execution evidence. This implementation has not run the scenarios, calibrated durations, or verified convergence on the CI machines.


- **Scenario logic**: `conftest_ft/scenario_<name>.py` — a typer app plus a `run_ci(mode)` runner.

| Scenario (`conftest_ft/scenario_*.py`) | Type | What it verifies |
| --- | --- | --- |
| `scenario_trainer_no_failure` | comparison | indep_dp matches normal DP when no faults |
| `scenario_trainer_with_failure` | comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `scenario_trainer_deterministic` | comparison, multi-phase | healing state transfer is bitwise-correct, on cold start and on resume from a post-healing ckpt |
| `scenario_random_crash` | soak | system survives random crashes without hanging |

### Modes

- **Selection**: `--mode`, defined in `conftest_ft/modes.py`.
- **Mode names**: `<kill>__<parallelism>[__fake_rollout][__moe_5layer|__moe_full][__colocate]`, segments separated by `__` and joined by `_` inside a segment.
- **What a name carries**: the `kill` segment always, then only the axes that differ from the naming defaults — real sglang engines, the dense `Qwen3-0.6B`, disaggregated placement. Node counts, engine counts and cell counts are never in the name; read them from the table below.
- **Why `kill` leads**: what a run crashes is the subject of this suite, so it is the first thing the name answers, and it is a property of the mode alone — no scenario widens it at runtime.
- **The scheme is enforced, not remembered**: `compute_mode_name` derives a mode's name from its fields against an explicit naming-default table, and `tests/fast/e2e/ft/test_naming_scheme.py` fails when a name drifts from it.
- **Declared per mode**: cell count, parallelism, model, train/rollout GPU split, `colocate` (default disaggregated, i.e. training and rollout on separate nodes), `ft_components` (default `("train",)`).
- **No rollout engines**: modes with `rollout_num_engines == 0` train on pre-recorded debug rollout data.

| Mode | Nodes | GPUs (train + rollout) | DP cells | Parallelism | Rollout | Model | `ft_components` | Why it exists |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 2 | CP2 TP2 EP2 | debug data | 5-layer MoE | `("train",)` | TP + EP coverage |
| `kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 2 | CP2 PP2 | debug data | 5-layer MoE | `("train",)` | PP coverage, via `--decoder-first-pipeline-num-layers 3 --decoder-last-pipeline-num-layers 2` |
| `kill_train__dp4_cp2__fake_rollout__moe_5layer` | 1 | 8 + 0 | 4 | CP2 | debug data | 5-layer MoE | `("train",)` | multi-replica coverage (>= 4 cells); uneven split after a fault |
| `kill_train__dp2_cp2__moe_5layer` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | 5-layer MoE | `("train",)` | real engines + the weight-update path |
| `kill_train__dp2_cp2` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | `("train",)` | `scenario_trainer_with_failure` under real generation; needs the dense model (see below) |
| `kill_rollout__dp4__colocate` | 1 | 4 shared | 4 | — | 4 engines × 1 GPU, colocated | dense Qwen3-0.6B | `("rollout",)` | colocated comparison mode |
| `kill_rollout__dp4` | 1 | 8 total | 4 | — | 4 engines × 1 GPU, disaggregated | dense Qwen3-0.6B | `("rollout",)` | rollout-only random P2P faults |
| `kill_train_rollout__dp2_cp2` | 1 | 4 + 4 | 2 | CP2 | 4 engines × 1 GPU | dense Qwen3-0.6B | `("train", "rollout")` | both kinds crash in the same run, synchronous training; disaggregated, since colocation makes the two crashes contend for the same gpus |
| `kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full` | 4 train + 2 rollout | 32 + 16 | 4 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | full MoE | `("train",)` | full model, all parallelism; multi-node, so no CI entry |

- **Batch shape**: `--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256` everywhere — 256 samples per rollout, divisible by both 2 and 4 cells. `scenario_trainer_with_failure` x `kill_train__dp4_cp2__fake_rollout__moe_5layer` trains the fault rollout on the 3 surviving cells, so the uneven 256-over-3 split is exercised there.
- **Model**: 1-node modes use the 5-layer MoE `Qwen3-30B-A3B-5layer`, except the three dense modes.

## Running the code

### In CI

- **Gating labels**: `run-ci-ft-short` for the comparison scenarios (minutes each), `run-ci-ft-long` for the soaks (tens of minutes to hours). Nothing here runs on an unlabelled PR.
- **Broad scopes**: `run-ci-all` includes both; the nightly cadence includes `ft-short` but not `ft-long`; `run-ci-image` excludes both.
- **Suite**: `suite="stage-c-8-gpu-h200"`, run by the job of the same name in `.github/workflows/pr-test.yml`.
- **Hardware**: every entry declares `hardware=["hopper", "blackwell"]`.
- **ft-long is enabled**: the registered soak entries run on `stage-c-8-gpu-h200` when selected by `run-ci-ft-long` or `run-ci-all`.
- **Validation boundary**: these scenarios have not been run for this implementation; duration estimates and convergence remain unverified.
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
- **`scenario_random_crash`**: only `run`, with `--mode` / `--seed` / `--num-steps` / `--trainer-crash-interval-seconds` / `--rollout-crash-interval-seconds`.
- **Dumps**: `resolve_dump_dir` in `tests/utils/soak/core/utils.py` puts them under `$MILES_TEST_DUMPS_ROOT/<run_id>/<test_name>/`, falling back to `/node_public/dumps` when the cluster sets no root. A comparison scenario's `run` deletes them when it ends; the random soak rejects a nonempty dump directory, so a finished soak leaves its dumps behind for inspection. The run id is what stops two agents running the same test from deleting each other's dumps.

### Cluster Backend

- **Selection**: `command_utils.default_config()`, off `MILES_SCRIPT_CLUSTER_BACKEND` / `MILES_SCRIPT_NAMESPACE` / `MILES_SCRIPT_RUN_ID`, already set in the miles-workbench pod.
- **Scenarios stay backend-agnostic**: no mode declares one; the backend changes only the set of fault forms.
- **One config throughout**: the same `ExecuteTrainConfig` threads through `prepare()`, `run_training()` and `api_server_host()`; on kubernetes the api server lives on a pod named after its `run_id`, so a second config would aim the injector at a release that does not exist.
- **Side-specific releases**: a comparison may provide `config_for_side`; the pipeline applies it once before the target context and launch, which both receive that same transformed config.
- **Bounded side handoff**: after each kubernetes comparison side, including a failed one, the pipeline uninstalls its Helm release and waits at most five minutes for both the release and every release-labelled pod to disappear. The next side and the CPU comparison start only after that completes, so asynchronous chart cleanup cannot overlap their GPU reservations. Ray comparisons never call Helm.
- **Unreachable is a failure, not a skip**: `create_backend_for_run()` asserts before handing back a backend, since exiting 0 would report green for a test that never ran.
- **Namespaced probes only**: never a cluster-scoped CRD read, which the workbench's Role cannot do.

### Generate Debug Rollout Data

- **Who uses it**: modes with `has_real_rollout == False`, through `--load-debug-rollout-data --debug-train-only`.
- **Where it comes from**: `prepare()` in `conftest_ft/execution.py`, via `U.hf_download_dataset()` on `fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer`.
- **Soak reuse**: `materialize_cyclic_debug_rollout_data()` symlinks the recorded files cyclically under the shared data dir, so a soak can run more steps than were recorded and the run pod can read the links.
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

### Fault Forms and Receivers

| Backend | Cell type | Forms, drawn from uniformly |
| --- | --- | --- |
| ray | actor | `inject_fault:sigkill` |
| ray | rollout | `inject_fault:sigkill` |
| kubernetes | actor | `inject_fault:sigkill`, plus `delete_pod` |
| kubernetes | rollout | `exec_sigkill`, `exec_sigstop`, `delete_pod` |

- **Each `FailureMode` is its own form**: pod deletion has the same weight as each individual failure mode. `FAILURE_MODES` currently enables SIGKILL only.
- **The actor class decides what a kill means**, since an injection carries only a mode and a `sub_index`: `TrainRayActor` and `ServeActor` crash their own process, the only thing that costs torchft a member, while `CommandActor` SIGKILLs the isolated process group rooted at the engine subprocess. That includes the launch shell and every engine child it spawned, so a dead cell cannot leave an orphaned scheduler holding GPU memory while its replacement starts; the Ray actor observes the subprocess exit and reports the death as production sees it.
- **Engine modes**: external SIGKILL and SIGSTOP have separate exit and stopped-process receipts. In-process exit, segfault and deadlock are not approximated with external signals.
- **How a kubernetes engine takes a kill**: its pod runs sglang as the entrypoint (`CommandWorkerSpec`), so no actor and no rpc server exist to receive `inject_fault`. The kill is delivered from outside instead, as a `kubectl exec` SIGKILL of the sglang processes in the engine container, and deleting the pod is the second, coarser form — the engine *is* the pod.
- **Deletion**: the async Kubernetes client deletes the observed pod with UID and resource-version preconditions, then confirms that UID is absent. A pre-existing deletion or a failed read cannot prove an applied fault.

### `scenario_trainer_no_failure`

```
Type: comparison (baseline=normal DP, target=indep_dp)
Steps: 2 (NUM_STEPS)
Compare: dumps rel <= 0.0085; metrics rtol=1e-2, atol=1e-8

1. Baseline: normal DP on debug rollout data (real engines in a real-rollout mode)
2. Target: the same arguments plus get_ft_args(mode), which is --use-fault-tolerance
   --ft-components <the mode's ft_components> --api-server-port 0
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

- **Pre-fault rollouts need bitwise weights on both sides**: the fault rollout trains the target's own live samples, and one bf16 ulp in a weight flips temperature-0.8 samples so the fault rollout trains different data (observed as a 6% `train/grad_norm` gap once the sglang v0.5.18 bump changed the sampled content). Both sides therefore run `--debug-deterministic-collective` (the same fixed fold for the normal-DP 4-rank reduce and the indep_dp CP-then-cross-cell reduce, as in `scenario_trainer_deterministic`) and `--clip-grad 10.0` (clipping inactive: the dense grad norm is ~1.3, and `train/grad_norm` differs by a few fp32 ulps across shardings, which an active clip would multiply into every update). The other FT modes have grad norms below 1.0, so clipping is inactive there without the override.
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
Steps: 60 (default)
CLI: --mode, --seed (42), --num-steps (60), --trainer-crash-interval-seconds (120),
     --rollout-crash-interval-seconds (240)

Targeting and assertions follow the mode's ft_components:
  ("train",)          -> inject into "actor" cells, assert trainer healing
  ("rollout",)        -> inject into "rollout" cells, assert the recovery cycle
  ("train","rollout") -> inject into both kinds, assert both
  A mode declaring rollout ft without real engines would schedule injections into a cell kind
    that does not exist, so FTTestMode refuses to be constructed at all

Architecture (external fault injection, not inside the training loop):
  1. Launch training with its own control endpoint and --mini-ft-controller-enable
  2. SoakRunner owns training, observation and actions on one asyncio loop
  3. SoakActionScheduler derives eligibility and deadlines from events and policy:
     one action at a time, including recovery and subsequent normal training progress;
     healthy topology and an unreserved surviving replica constrain cell targets
  4. Record the incarnation-bound request before starting its async action
  5. Record effect evidence separately from command return; keep observing concurrently
  6. Close admission, observe the recovery tail, collect tasks and take a final observation
  7. Tear down the owned run, archive evidence, then check archived events

Per-kind schedules: exponential, mean that kind's --*-crash-interval-seconds

Witnesses, counted per kind:
  forms   -> every enabled form has a confirmed effect, not merely a successful RPC
  train   -> >= 2 actor effects, matched incarnation replacement/reconfiguration and normal progress
  rollout -> >= 2 rollout effects, matched new incarnation observed Serving
  tail    -> every action resolved, every recovery complete, then normal training progress

Faults are random, so beyond the witnesses neither an exact sequence nor the end-state
membership is asserted.
```

- **Why per-kind schedules and counting**: each kind's cadence stays what it would be in a single-kind soak, and the trainer assertion reads only `actor` injections while the rollout one reads only `rollout` — a mixed soak cannot let one kind's crashes pay for the other's missing heal.
- **Why rollout gets the longer interval**: the replacement pays a full sglang launch plus a weight sync before it can serve again.
- **No per-kind quota**: when the trainer has no spare replica for a long stretch every injection lands on rollout, and the failure form is a loud "too few trainer injections" rather than a silent pass.
- **Sequential recovery**: each form supplies its recovery predicate; the scheduler admits no new action until every prior action has recovered. FT requires replacement plus normal training; deployment takeover requires new checkpoint and training progress. Unknown outcomes and failed observations never release this gate.
- **A form that leaves its target running**: `BaseSoakActionForm.harms_target=False` creates no recovery obligation on a surviving replica. Per-kind deadlines and action limits still apply; quiescence gates apply only where the policy requires them.
- **Quiescence fleet size**: the streak counts replicas against the kind's declared `SoakTargetPolicy.expected_count`, so a deleted pod cannot disappear from the listing and leave a smaller fleet looking complete.
- **Why every enabled form has to land**: the floors count injections, not forms, so `inject_fault:sigkill` alone could clear them while `delete_pod` is never tried. This witness makes the draw's preference for an untried form binding.
- **Why the per-cell pairing**: a floor of ">= 2 healings" passes whenever the last crash never recovered. The default intervals are short enough that a soak reliably clears the floors.
- **Why the step budget is 60**: the run needs time for multiple faults drawn with a mean-240s rollout interval, replacement and recovery, and a completed fault-free tail. The quiescence and recovery gates can extend the wait between faults; this budget has not been calibrated by a run.
- **Why the rollout witness is one-sided**: sampled polls may miss the down transition. Recovery instead requires a different incarnation observed Serving after the request; elapsed time or a stale Healthy reading of the victim cannot satisfy it.
- **Recovery identity**: the same cell name or a long delay cannot prove replacement. Recovery uses the requested incarnation, a new incarnation, and the corresponding Serving or trainer reconfiguration evidence.
- **Session ownership**: `SoakRunner.run` owns training and observation tasks; failures propagate through their task group. Training runs `asyncio.to_thread` in the test process, so it cannot be cancelled: the run timeout closes admission, stops observing and still waits for the training thread before raising. Training completion stops the runner explicitly. Cancellation still collects action tasks before final observation, teardown and evidence collection; cleanup failures do not erase the original failure.
- **Fresh random-run evidence**: the random-crash entry requires an empty dump directory before starting observation. Reusing a nonempty directory fails with its path instead of reading stale progress or deleting previous evidence; select a fresh run ID.
- **Action projection**: `project_actions(events)` associates Requested, Applied and Result by request ID without caching derived state. Missing Applied or Result represents an incomplete action; each action binds one victim; launcher and tail checks retain that action boundary.
- **Independent evidence**: random FT and rollout-deterministic runs write ordered typed events to `<dump_dir>-soak/<session_id>/events.jsonl`. Requests are flushed before dispatch. After task collection, training-event files and discarded generations are copied under `sources/`; checks use those paths. A terminal marker and per-file SHA-256 digests distinguish a complete collection from a truncated or changed archive.

- **Shared soak engine**: the neutral soak engine lives in `tests/utils/soak/core/`, with the FT action forms, observers and checkers under `tests/utils/soak/ft/` and the deployment adapters under `tests/utils/soak/deploy/`. Each kind must independently meet its effect and recovery requirements.
