# E2B / agentic rollout failure reproduction

## Chosen-token reply pilot

This branch changes score-centering session replies in both v1 and v2. The
server retains the complete backend response for training, including all 128
candidate IDs and logprobs. Non-streaming replies preserve chosen-token logprobs,
token IDs, messages, and usage, but omit internal `output_top_logprobs` and return
empty OpenAI `top_logprobs` lists unless the original client explicitly requested
candidates. Explicit requests retain that many candidates in the standard OpenAI
field. Other loss types and the existing streaming behavior are unchanged.

The filtering copies response containers; it must not mutate session records.
This corrects the earlier experimental `strip` arm, which also removed chosen
logprobs and was unsuitable for Harbor rollout-detail collection.

`chosen_logprobs_gate.py` checks a captured response with the pinned Harbor
chosen-logprob extractor, then exercises the real session server and native
sample decoder. It compares both training and top-128 candidate hashes against
an unfiltered reference. Pass `--root`, `--parent`, `--fixture`,
`--reference_results`, and `--harbor` in the original pinned environment. The gate
creates no E2B sandboxes and starts no optimizer. It is a compatibility check,
not a full agentic test or evidence that the original native crash is fixed.

`chosen-logprobs-pilot.json` specifies the next bounded agentic pilot: one wave,
16 prompt groups with eight samples each, 128 concurrent sandboxes, the same eight
GPU nodes, 64k total tokens and 16k per turn. Merge its argument/environment
overrides into the archived recipe when preparing a fresh run. Keep top-128
recording, prebuilt task images, linear history, and the existing acceptance of
truncated and agent-timeout trajectories. Do not reuse an old corpus for this
pilot; observe fresh multi-turn replies and cleanup. Infrastructure failures
remain invalid training data. The settings file does not launch a job.

## Controlled payload experiment

For the original unfiltered performance comparison, use the
`shi/session-payload-control` branch. On this pilot branch, the production filter
also applies to the historical `full` arm; use the dedicated compatibility gate
above to validate the new behavior.

`payload_control.py` isolates candidate-metadata overhead using the real pinned
Miles session server and its sample assembler. It captures one live task response
from an existing SGLang engine, then replays identical generated tokens through
fresh session-server processes. Arms are plain probabilities, top-128 metadata,
and top-128 metadata retained for training but removed from agent-facing replies.
The default order is plain/full/strip/strip/full/plain, with eight server processes
and 128 concurrent single-turn sessions per block. Owned prebuilt E2B sandboxes
execute the same short command after each reply; no benchmark agent or verifier
runs. This is a synchronized payload stress test, not a full agentic rollout or
a measurement of model inference speed. It measures server/driver event-loop lag,
server CPU, sampled peak RSS, reply/sample bytes, and command errors/latencies.

Run in the original pinned Python environment with the archived runtime environment
and administrator credentials loaded privately. Supply `--root` (a new scratch
directory), `--parent` (the archived run directory), and `--engine` (an existing
OpenAI-compatible engine URL). Start with `--workers 1 --concurrency 1
--sandbox_count 0 --repetitions 1` to verify fixture/session compatibility, then
use the defaults for the stress comparison. Only this experiment's child processes
and sandbox IDs are terminated. Logs and result JSON files remain under `--root`.

The fixture includes exact request/response data and a SHA256 digest. Raw task
contents and credentials are not committed. Read both per-trial errors and cleanup
receipts before interpreting timing numbers.

This is an archived workload reproduction, **not a deterministic minimal reproducer**.
It preserves the collector used for the failure; it does not implement the proposed
standalone-SGLang redesign or disable uvloop.

## Observed failures

Two separate symptoms were observed:

1. E2B command RPCs failed with:
   - `agentenv proxy: upstream body-idle failed ... no request body progress for 30000 ms`
     on `POST /process.Process/Start`.
   - A client timeout while opening the command response stream (60 seconds in examined cases).
2. After approximately seven hours, the Miles `RolloutExecutor` process aborted:
   ```text
   2026-09-25 08:47:59.875 PT (15:47:59.875 UTC)
   SIGABRT
   pthread_kill -> raise -> abort -> uv__epoll_ctl_prep.cold
   Fatal Python error: Aborted
   miles/utils/async_utils.py:28 in _start_loop (loop.run_forever)
   ```
   Ray then reported `ActorDiedError`, `SYSTEM_ERROR`, connection error 2 / EOF.
   Ray's subsequent SIGTERM was cleanup, after the abort. Container memory-event
   counters were `oom=0`, `oom_kill=0`. No core dump was available.
   The installed uvloop binary routes unexpected epoll_ctl failures to abort;
   the failing file descriptor and errno were not recorded. This does not prove
   that E2B or a cancellation race caused the native crash. A simple closed-socket
   probe did not reproduce it.

## Source and runtime

This branch includes the exact Miles code used, plus the four collection/adaptor
modules previously supplied outside the repository. Their behavior is unchanged.
The configuration renderer only relocates paths/endpoints, omits W&B credentials/
project identifiers, omits an unused launcher import path, uses NO_PROXY=* for an
administrator-owned isolated cluster, and starts fresh unless resume roots are supplied.
Task prompts are rebuilt from the pinned TB task checkout, so no benchmark content
or organization-specific E2B template IDs are redistributed here.

- Miles failing base: `5168b9471ff108c01441dcc26eacc129a9a7625f`.
- SGLang: `c9660a7e112c07fc2003e90d5fa3330323acf2f9`.
- Megatron-LM: `73b54618f7e58e0f25f619bcfecbe2640765475a`.
- Harbor: `43e944d6578aca70274315a53121d2147515c400`.
- Terminal Bench 2.1: `7131e4375048a0e408a8fb404b5f499d726b695b`.
- Original image: `docker.io/radixark/miles@sha256:65b6e72dff830fc6823b8b10c31b2d006c9c19adac61aa2299d9f1a253b90351`.
- Python 3.12, uvloop 0.22.1, Ray 2.58.0, E2B 2.50.0, pyqwest 0.10.0,
  connectrpc 0.11.1. Full relevant version inventory: `dependency-versions.json`.

The historical source objects are also archived in
`git@github.com:Shi-Dong/llm-training-snapshots.git` under
`refs/training-runs/260925-252f1eb9/{miles,sglang,megatron-lm,harbor,terminal-bench-2-1}`.
For a pin not reachable in the upstream repository, fetch its archived ref into
the corresponding checkout, then checkout FETCH_HEAD. These objects may require
repository access. Do not silently substitute current main.

The original container had Harbor source and a dependency overlay on PYTHONPATH.
The base image packages take precedence over overlay duplicates. Preserve that
order as encoded in recipe.json. Install Harbor's dependencies into a separate
Python 3.12 environment using uv with overlay-requirements.txt, and install the
pinned Harbor checkout with --no-deps. Some pinned packages may require the
original package mirror or wheels; do not silently substitute versions. Do not replace
the image's torch, Ray, or uvloop to "fix" the environment before reproducing.
After installation, verify imported versions and module paths in the same runtime
environment used by the worker.

## Workload

- 8 nodes, 8 H200 GPUs per node, 64 independent TP1 SGLang engines.
- Frozen Qwen3.8-27B FP8 inference checkpoint, same tokenizer/template.
  The checkpoint may require access; provide an existing local copy.
- Miles `--debug-rollout-only`; no optimizer updates.
- 128 concurrent E2B sandboxes; 8 session workers; 2 client requests per engine;
  engine max-running-requests=8.
- 60 task names in train-task-names.json; 30 waves of 16 groups x 8 samples.
  There are 3,840 planned slots, with at most one replacement attempt per slot.
- 65,536 total tokens, 16,384 generated tokens per turn.
- Temperature=1, top_p=1, top_k=-1; top-128 candidate probability recording.
- Terminus-2; summarization=false; linear_history=true; response-length policy=abort.
- Task timeouts remain task-defined (timeout multiplier 1); the outer trial
  timeout is 7,200 seconds. Model request timeout is 10,800 seconds.
- Truncation gets reward 0; agent time-budget exhaustion uses verifier reward or 0.
  Infrastructure errors and incomplete token records remain invalid.
- Keep `MILES_ASYNC_WARNING_POLICY=log` and `MILES_ASYNC_DIAGNOSTICS=1`.
  Preserve the failing loop behavior for this reproduction; do not switch to the
  standard event loop until collecting a baseline.

## Prepare on an existing Ray cluster

All paths below must exist at the same absolute location on every node.
Use your platform's supported launcher to provision/join the 8-node cluster.
No cluster creation, training launch, sandbox creation, or cleanup is performed by
prepare.py.

1. Clone this Miles branch and the pinned dependencies. Put the pinned SGLang
   checkout in the original image at its original import location, or install it
   there. Verify that it imports the recorded commit, not a different bundled copy.
2. Obtain the administrator's existing E2B template manifest and key file. The
   manifest must contain one entry per task:
   ```json
   {"results": [{"task": "TASK_NAME", "template_id": "EXISTING_TEMPLATE_ID",
                 "source_docker_image": "IMAGE_FROM_TASK_TOML",
                 "cpus": 2, "memory_mb": 4096}]}
   ```
   CPU/memory values above are illustrative; use the exact task resources.
   Templates must already contain the task environment plus tmux and asciinema.
   The adapter never builds an image. It rejects mismatching source-image/resources.
3. Run the following in Fish, substituting administrator-owned paths/endpoints:

```fish
git clone --branch shi/e2b-rollout-crash-repro git@github.com:radixark/miles.git /work/miles
set repro /work/miles/examples/debug/e2b_rollout_repro
/opt/sglang/bin/python3 $repro/prepare.py \
    --output /scratch/e2b-repro \
    --model /models/Qwen3.8-27B-FP8 \
    --reference /models/Qwen3.8-27B_torch_dist \
    --tasks /work/terminal-bench-2-1/tasks \
    --harbor /work/harbor \
    --megatron /work/Megatron-LM \
    --dependencies /work/harbor-venv/lib/python3.12/site-packages \
    --templates /work/e2b-template-manifest.json \
    --key_file /run/secrets/e2b-key \
    --api_url https://YOUR_E2B_API \
    --sandbox_url http://YOUR_E2B_SANDBOX_PROXY \
    --submission_id e2b-rollout-repro
```

Read the generated command.txt and runtime-env.json. Copy the generated directory
to the same path on all nodes before submission. The key file must be accessible
to the workers; only its path is forwarded, never its contents.

To submit to an **already running, dedicated** Ray cluster, from its head:
```fish
set command_line (string trim -- (cat /scratch/e2b-repro/command.txt))
ray job submit \
    --address http://127.0.0.1:8265 \
    --submission-id e2b-rollout-repro \
    --runtime-env-json (string join "" -- (cat /scratch/e2b-repro/runtime-env.json)) \
    -- $command_line
```

Alternatively POST the generated ray-job-request.json to the existing Ray Jobs
API; its entrypoint is already shell-quoted. Do not submit both ways.
Use a new output directory and submission ID on each independent reproduction.
Use a tmux session to tail
`/tmp/ray/session_latest/logs/job-driver-e2b-rollout-repro.log`.

### Exact old resume versus fresh reproduction

The crashed run restored two earlier waves (165 valid samples, 91 exhausted slots)
before collecting new work. Fresh mode is intentional for admins without those
artifacts; it is not the identical request schedule. For the original resume,
obtain both preserved corpus directories out of band and pass
`--resume_roots /path/to/newer-parent:/path/to/older-parent`.
Their attempt manifests contain absolute artifact paths and checksums: mount them
at the recorded paths, or relocate manifest paths with an explicit audited copy.
Do not invent completed slots or reset exhausted retries.
The prior run's artifacts and credentials are not bundled in this public branch.

## Tests and failure evidence

Run the included test_reward_policy.py and test_frozen_collect.py in the pinned
image with the same PYTHONPATH from runtime-env.json (they create only temporary
local fixtures, no sandboxes). The former covers timeout/truncation reward handling;
the latter exercises bounded retries, parent recovery, persistence, and empty waves.
test_prepare.py checks rendering and argument quoting without GPUs.
These tests check packaging/collector behavior; they do not reproduce the native abort.

During reproduction, retain:
- worker-*.err, python-core-worker-*.log, raylet.out and driver log;
- attempts.jsonl, batch-*.json, pilot-verdicts.jsonl and Harbor trial result.json;
- sandbox IDs and exact request IDs from timeout messages;
- /sys/fs/cgroup/memory.events, process RSS, open FD count, and kernel OOM logs;
- native core/backtrace or an epoll_ctl trace around the abort, where permitted.

Use the recorded UTC timestamps when searching service logs; the crash timestamp
above includes its PT conversion. A 30-second body-idle error or 60-second
stream-opening timeout reproduces the RPC symptom, not necessarily the SIGABRT.
Do not call the native crash reproduced unless the worker actually aborts with
the matching stack.

After an attempt, clean up only sandboxes whose session/trial IDs belong to that
attempt. A worker abort can skip normal sandbox teardown. Do not bulk-delete the
administrator's whole E2B inventory.

## Validation scope

The reward-policy tests (2 tests) and collector tests passed in the original
devbox environment. The configuration-renderer test passed locally, including
paths containing spaces. The full 8-node workload was not
relaunched solely to publish this package. Reproduction time and success rate are
unknown; lowering concurrency or changing the loop is an isolation experiment,
not an equivalent reproduction.

## Continuous frozen collection

Select `continuous_collect.RolloutFn` to backfill individual trajectory slots across
export batches. It reuses Miles sample-admission accounting with singleton units,
bounded by `SANDBOX_CONCURRENCY`. Each slot gets at most two attempts. Group
membership and export order remain unchanged. Completed slots retain only file
references in memory; token arrays are loaded for one export batch at a time.
Use only `--debug-rollout-only --start-rollout-id 0`.

Set `FROZEN_RESUME_ROOTS` to colon-separated immutable parent directories. Checksums,
task/group/slot identity and sampling seeds are verified. Valid slots are reused;
two failed attempts exhaust a slot; one failed attempt leaves one retry.
Interrupted attempts without a persisted result may be rerun. Stop parent writers
first. Preserve dataset order, model, seeds, sampling, token budgets and rewards.
Keep parent archives available. The saved data-source cursor is not a resume
checkpoint for this collector: restart from ID zero using attempt manifests.

`progress.json` reports settled/usable/active slots; batch reports describe exports.
Run `test_continuous_collect.py` in the pinned environment. No new dependencies.
