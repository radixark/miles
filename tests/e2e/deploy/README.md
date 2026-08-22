# Deployment E2E Tests

## Running

Needs `PYTHONPATH=.` and a miles-workbench pod (`MILES_SCRIPT_*` env preset). Kubernetes only:
entries register via `register_cuda_ci` and fail with a reason on any other backend.

```bash
PYTHONPATH=. python tests/e2e/deploy/test_split_deterministic.py                          # as CI
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/split/scenario_split_deterministic.py run  # via app
# hot restart is two levels: the mode is a subcommand of its own
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/hot_restart/scenario_hot_restart_deterministic.py \
    checkpointed run
```

- **Subcommands**: comparison scenarios expose `run` / `baseline` / `target` / `compare` (no GPU) /
  `generate-data`; the multi policy one exposes `run` / `verify`; the realistic soak exposes `run`
  only; hot restart deterministic nests these under one subcommand per mode.
- **Dump dirs**: `$MILES_TEST_DUMPS_ROOT/<run_id>/<TEST_NAME>/`, defaulting to `/node_public/dumps` when
  the cluster sets no root (only `run` deletes it, and only its own run id's subtree; `--dump-dir`
  overrides for `baseline` / `target` / `compare`); multi policy:
  `<output_dir>/multi_policy_solver_verifier/<run_id>/`.

## Test Specifications

### `scenario_split_deterministic`

```
Type: comparison (baseline=one release, target=one release per deployment)
Steps: 3 rollouts

1. Baseline: the whole run in one release
2. Target, installed in order: TRAINER; INFERENCE e0, e1 (one engine each); PRIMARY last
   - installing PRIMARY blocks until the run ends
   - addresses from the example's address_book; ordering, shared run uuid and uninstall from
     conftest_deploy/split/split_deployment.py
3. Compare: dumps and metrics bitwise; engine checksums identical per (rollout, engine); engine
   count; weights moved; nonzero gradients >= 2 rollouts
```

### `scenario_split_multi_policy`

```
Type: single run (multi trainer is not bitwise-reproducible)
Steps: 3 rollouts
Releases: TRAINER solver-actor / verifier-actor, INFERENCE solver / verifier, PRIMARY last

1. Install the five releases via the example, one command per part
2. Assert: every rank trained with its own policy's args; every policy reported a training reward
   inside TRAIN_REWARD_BOUNDS; the leader reported every rollout; finite nonzero grad_norm/loss
3. Assert per policy: train_rollout_logprob_abs_diff <= 0.1 (the cheapest wiring bug - a
   trainer scoring another engine's tokens - shows up here)
```

### `scenario_hot_restart_deterministic`

```
Type: comparison (baseline=untouched, target=same command, orchestration script replaced mid-run)
Steps: 6 rollouts
Releases: baseline and target derive separate releases from the parent run id; every target
          take-over upgrades the target release in place
Timing: exact - the run parks at the scheduled step boundary (sleep-forever action) and the
        driver relaunches it there, so a take-over's landing is pinned, not raced
Plan: a file under the base dump dir, not under either side's, which each run deletes (argv
        stays byte-identical across relaunches; a pod's command carries it)
Gate: the parked run writes a sentinel beside the plan; the driver waits for it
Modes: checkpointed  - --save-interval 2 (saves after 1, 3, 5), 2 restarts: restart 1 frozen
                       between steps 2 and 3 (resumes save 1), restart 2 frozen between steps
                       4 and 5 (resumes save 3)
       no_checkpoint - --save-interval 4 (saves after 3 and 5), 1 restart frozen between steps
                       1 and 2, before anything was saved
Entries: test_hot_restart_checkpointed.py, test_hot_restart_no_checkpoint.py

1. Relaunch the same command + --hot-restart orchestration,rollout_executor per the mode
2. Assert workloads: only orchestrator + rollout-executor rolled (pod uid / restartCount / stamps);
   compare canonical PodTemplate fingerprints for every workload because a controller may advance the generation of
   an unchanged custom resource
3. Assert process: one trainer rpc boot uuid throughout, answering the take-over's fresh client
4. Assert redo, measured off the logs, per mode:
   - checkpointed: one .trash_* per restart; resume point == the pinned save (the snapshot
     beside that checkpoint), so the run resumed there, not at step 0; the redone steps are
     exactly the pinned (save, frozen step] windows; per-step attempts all 1 or 2
   - no_checkpoint: record carries no saved iteration; NO .trash_* (the run's --load resolves to
     --ref-load, which holds no snapshot to restore); the abandoned event stream is discarded and
     restored steps 0..5 each appear exactly once; the restart record and the exact +2 weight-version
     delta at every rollout witness the work that was redone; the run still saves after the restart
5. Compare every metric exactly as in scenario_split_deterministic. Ordinary metrics require a zero
   target-minus-baseline delta. The four rollout weight-version statistics require the exact cumulative number
   of updates redone after each recorded restore point, because the trainer and engines survive and their
   publication version remains monotonic. Each formula is consumed by that metric key's ordered occurrences in
   the already aligned event stream; the sequence length must exactly equal the occurrence count because tracker
   MetricEvents do not carry a rollout id. No metric key is dropped; dumps and engine checksums still compare
   bitwise.

checkpointed lands every take-over on a non-save step, so unsaved steps are rolled back and
redone; no_checkpoint has nothing to resume from and starts over at rollout 0.
```

### `scenario_hot_restart_realistic_gsm8k`

```
Type: single run, ft's scenario_realistic_gsm8k with hot restarts instead of kills
Steps: as scenario_realistic_gsm8k
Injection: HotRestartFaultForm at random intervals via the ft fault-injection plan, seed logged
Eligibility: none - every draw fires; a draw before the first save starts the run over from
        --ref-load, a product path this soak covers rather than skips
Landing signal: both replaced workloads (orchestrator, rollout-executor) carry a stamp other
        than the one they carried at the draw - rewritten, not added
Load-bearing: adds --save/--load and --save-interval 3 (bounds one take-over's cost); mean draw
        interval 600s (--hot-restart-interval-seconds)

1. Run the realistic gsm8k recipe while the plan injects hot restarts
2. Assert: gsm8k reward improves as in scenario_realistic_gsm8k
3. Assert: >= MIN_HOT_RESTARTS take-overs landed; no injection attempt failed; every relaunch
   thread finished without raising (where the run's own metric verdict surfaces); each landed
   take-over stamped orchestrator and rollout-executor once; no other workload rolled or lost a
   pod; one trainer boot uuid throughout; no take-over threw away more than one save interval
4. Artifact: per take-over cost (index, checkpoint held, step reached) in
   <dump_dir>/hot_restart/evidence.json

Hot restart rides the ft injection machinery so a future soak can mix it with pod kills.
```
