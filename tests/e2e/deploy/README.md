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
   - whatever installed the releases removes them all when the run ends, PRIMARY included, and
     waits for their pods, rather than leaving PRIMARY to the teardown it schedules for itself
3. Compare: dumps and metrics bitwise; engine checksums identical per (rollout, engine); engine
   count; weights moved; nonzero gradients >= 2 rollouts
```

### `scenario_split_multi_policy`

```
Type: single run (multi trainer is not bitwise-reproducible)
Steps: 3 rollouts
Releases: TRAINER solver-actor / verifier-actor, INFERENCE solver / verifier, PRIMARY last

1. Install the five releases via the example, one command per part
2. Assert: every rank trained with its own policy's args; the leader reported every rollout;
   finite nonzero grad_norm/loss. Three rollouts say nothing about learning, so whether a policy
   improves is gated by tests/e2e/long/test_multi_policy_solver_verifier_gsm8k.py instead
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
Observation: the cluster observer records a snapshot only once two consecutive reads agree on the
        release's workloads and pods, and afterwards drops any listing missing a settled
        workload, so a topology still being installed is never taken for a run losing pods
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
3. Assert process: one trainer rpc boot uuid throughout, answering the take-over's fresh client, and
   read once off a whole-release snapshot taken before the first take-over stamped anything
4. Assert redo, measured off the logs, per mode:
   - checkpointed: one .trash_* per restart; resume point == the pinned save (the snapshot
     beside that checkpoint), so the run resumed there, not at step 0; the redone steps are
     exactly the pinned (save, frozen step] windows; per-step attempts all 1 or 2
   - no_checkpoint: record carries no saved iteration; exactly one .trash_*, holding the log
     thrown away (steps 0..1, each once) and sharing no step with the log that replaced it; the
     surviving log describes each of the 6 steps exactly once; the run still saves after the
     restart, past the step it was frozen at
5. Compare: bitwise as in scenario_split_deterministic, engine checksums included, with one
   exemption - rollout/weight_version mean/median/max/min. The trainer outlives a take-over, so
   its weight update counter keeps counting through the steps the target redoes and stands
   ahead of the baseline's at the same step

checkpointed lands every take-over on a non-save step, so unsaved steps are rolled back and
redone; no_checkpoint has nothing to resume from, so its event log is moved aside and it starts
over at rollout 0 with the run.
```

### `scenario_hot_restart_realistic_gsm8k`

- **Entry**: the E2E module forwards to `tests.utils.soak.deploy.scenario`; injection, observation and recovery checks live in `tests/utils/soak/`.
- **Training**: synchronous GSM8K training recipe, 250 rollouts by default, disaggregated P2P weight transfer; no accuracy threshold or tail evaluation requirement.
- **Takeovers**: exponential mean interval 600 seconds; at least two applied takeovers; each must preserve non-orchestration workloads and resume from checkpoints within `SAVE_INTERVAL + 1` steps.
- **Recovery**: a takeover is eligible only with a checkpoint; the next action waits for a new checkpoint and training beyond the pre-takeover rollout. Launcher lifetime is tracked separately from recovery.
- **Mixed mode**: `run --mix-ft` adds trainer and rollout faults to the same runner, with mean intervals 120 and 240 seconds. Each FT kind needs its own effects, replacement/reconfiguration and recovery evidence; deployment success cannot satisfy FT coverage.
- **Mixed takeover checks**: compare snapshots from immediately before each takeover through its recovery. FT-driven Pod replacement outside that window is not attributed to deployment takeover.
- **Target refresh**: every action selects identities from a fresh observation; takeover does not reuse a previously prepared cell/process target.
- **Tail**: shared admission closure leaves the final 20 percent of rollouts free of new faults; final observation must prove all actions recovered and all owned launchers finished.
- **Weight evidence**: explicitly enable bounded checksum collection; check same-version consistency across archived active/discarded event generations. Require exact update, epoch and incarnation coverage for at least two publications in the final uninterrupted tail; interrupted earlier publications remain a coverage gap.
- **Artifacts**: `<dump_dir>-soak/<session_id>/events.jsonl`, launcher specifications/logs, `hot_restart/evidence.json`, and independent active/discarded training-event copies under `sources/`; terminal closure and SHA-256 digests detect incomplete or changed evidence.
- **CI boundary**: the entry remains disabled until a Kubernetes lane supplies shared storage, worker images and release-management credentials. The H200 Ray lane cannot execute this contract.
- **Execution status**: neither standalone nor mixed takeover has been run or calibrated in this implementation task.
