---
paths:
  - "tests/*.py"
  - "tests/fast/**/*.py"
  - "tests/fast-gpu/**/*.py"
  - "tests/e2e/**/*.py"
  - "tests/ci/test/**/*.py"
---

# CI Test Registration

Rules for adding a test file, moving one, or changing a `register_*_ci(...)`
call. CI selection is driven by that declaration, never by the workflow YAML:
the runner discovers a test from where it lives and what it declares. What to
do when a check goes red is in `ci-failure-triage.md`.

## Placement

- A `test_*.py` lives under `tests/fast`, `tests/fast-gpu`, `tests/e2e`, or
  `tests/ci`; no CI job collects anything else, and
  `tests/ci/test/test_ci_discovery_coverage.py` rejects a tracked test file
  elsewhere. `tests/manual/` is only for tests no CI job should run.
- A GPU test file pays for a model download, Ray startup, and an engine launch
  on every run. Add cases to an existing file with the same launch; a new file
  is for a new fixture, model, or parallel layout.

## Stage

A Hopper test holds its stage allocation for its whole run. B200 allocates the
test's explicit `num_gpus` from a shared pool; preserve all cases when choosing it. Walk the table from the top and stop at the first stage that can
run the test; copying a neighbouring test's `suite=` is not a reason.

| Order | Where | Runs on | Use for |
|---|---|---|---|
| 1 | `tests/fast/` (no declaration; suite `stage-a-cpu`) | GitHub-hosted CPU | CPU tests that finish in seconds; it gates every CUDA stage |
| 2 | `register_cpu_ci(..., suite="stage-b-cpu")` | GitHub-hosted CPU | CPU tests that take minutes |
| 3 | `tests/fast-gpu/`, suite `stage-b-2-gpu-h200` | 2× H200 | GPU tests that finish in a few minutes |
| 4 | `stage-c-2-gpu-h200` | 2× H200 | 2-GPU tests |
| 5 | `stage-c-4-gpu-h200` | 4× H200 | 4-GPU tests |
| 6 | `stage-c-8-gpu-h200` or `stage-c-8-gpu-h100` | 8× H200 / 8× H100 | 8-GPU tests |
| — | `stage-c-4-gpu-b200` | 4× B200 | Blackwell tests needing up to 4 GPUs |
| — | `stage-c-8-gpu-b200` | 8× B200 | Blackwell tests needing 8 GPUs |

The stage's GPU count equals the count the test requests (`ray start
--num-gpus`, `--actor-num-gpus-per-node`, `torchrun --nproc-per-node`): a 4-GPU
test on an 8-GPU stage idles four GPUs for its whole run. Blackwell
uses the 4-GPU suite to select small tests, but allocates the declared
`num_gpus=1`, `2`, or `4`; the suite no longer fixes its allocation.
Every new CUDA test must declare `num_gpus` explicitly. Never lower it by
introducing skips or reducing the parallel topology that the test covers.

A new test needs no ROCm registration. A file that already has
`register_rocm_ci(..., suite="nightly-stage-c-<N>-gpu-*")` keeps `<N>` equal to
its CUDA stage's GPU count: a change that moves the CUDA registration to a
stage with a different GPU count updates `<N>` in the same commit. The external
nightly has only `stage-c` suites, so `stage-b-2-gpu-h200` pairs with
`nightly-stage-c-2-gpu-*`. `tests/ci/test/test_ci_rocm_nightly_gpu_count.py`
rejects a mismatch.

## Declaration

```python
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=500,                      # measured; see est_time below
    suite="stage-c-2-gpu-h200",        # home stage, from the table above
    num_gpus=2,                       # minimum preserving every test case
    labels=["megatron"],               # domain labels that select it on a PR
    hardware=["hopper", "blackwell"],  # supported CUDA generations
)
```

- A GPU file has one top-level `register_cuda_ci(...)`, one
  `register_rocm_ci(...)` per ROCm suite it runs in, or both. The calls are
  parsed from the AST: top-level, literal arguments, no alias. `tests/fast/`
  files need none.
- `labels`: reuse a domain label from `tests/ci/labels.py`. A new label also
  needs a `run-ci-<key>` repository label, which only a maintainer can create,
  and an entry in `.github/workflows/policies/comment-command-access.json` to
  be addable from PR comments.
- `hardware`: list a GPU generation only when the code paths the test runs
  support it (`--sglang-attention-backend fa3` is Hopper-only; nvfp4 / mxfp8
  kernels are Blackwell-only). The generation of the home `suite` comes first.

## Which PRs run it

- `labels` decide which PRs run the test within its cadence:
  `labels=["megatron"]` runs only when the PR carries `run-ci-megatron`. If a
  gated test does not run, add the matching `run-ci-<label>`; a maintainer can
  add `run-ci-all` to select every label.
- Cadence is separate: `nightly=True` keeps a registration out of regular runs,
  while nightly, weekly, and release runs include both kinds.
- Until a contributor's first PR merges, GitHub holds every CI run of a fork PR
  for a maintainer's "Approve and run", after every push. Any `run-ci-*` label
  a maintainer adds also approves the held runs, for that push and later ones.

## `est_time`

For a `register_cuda_ci` test, `est_time` balances shards and sets the per-file
timeout, `max(1800 s, 1.25 × est_time)`, in stage runs and in `/rerun-test`
alike. It is measured, never copied from a neighbouring test. CPU and ROCm
registrations need no timing run.

An enabled `register_cuda_ci` test with neither `nightly=True` nor a `long` or
`ft-long` label keeps `est_time` at or below 2400 s; cut its workload until it
fits.

1. Before the first run, use an upper bound that safely exceeds the expected
   runtime, so the timeout does not kill it. Up to 1440 s the 1800 s floor
   applies anyway.
2. Before the PR merges, comment `/rerun-test <path/to/test_file.py>` on it. It
   runs only that file, on its registered suite's runner, at the PR head; it
   needs no domain label and writes no performance baseline. Posting it takes a
   merged commit in `radixark/miles` (a first-time contributor asks a
   maintainer); a fork head gets no `WANDB_API_KEY` or `HF_TOKEN`; a `disabled`
   registration cannot run this way. The command is specified under "Manage CI
   from PR comments" in `docs/developer/ci/01-label.md`.
3. Read the runtime from the job's **Run tests** step, never from the status
   comment, which counts from the start of the workflow and so includes
   queueing, image pull, and dependency setup (one run: 2352 s in the comment,
   362 s in **Run tests**). In a stage run, the file's `End (...)` log line
   carries the same number as `elapsed=`.
   ```bash
   gh run view <run-id> --repo radixark/miles --json jobs --jq '.jobs[].steps[] | select(.name == "Run tests") | (.completedAt | fromdate) - (.startedAt | fromdate)'
   ```
4. Set `est_time` to `ceil(1.25 × measured seconds)`, rounded up to the next
   10 s at or below 200 s and to the next 100 s above it (362 s → 453 → 500),
   the rule the `ci-e2e-time-tune` skill applies to nightly runs.
5. Add one line per new or moved CUDA test to the PR description:
   ```text
   CI timing: tests/e2e/short/test_yours.py on stage-c-2-gpu-h200, Run tests 362 s, est_time=500, https://github.com/radixark/miles/actions/runs/<run-id>
   ```

Repeat steps 2–5 when a test moves to another stage or its runtime clearly
changes. A new `register_ci_gate(...)` stays inactive until nightly runs seed
its history; `/rerun-test` never writes it.

## `disabled=`

- Set it to unblock other PRs on a maintainer's call, never to land your own
  feature PR. The registration stays in the plan and is reported as skipped.
- The reason says what must change before the test is re-enabled and cites the
  tracking issue or PR, as `#123` or its GitHub URL;
  `tests/ci/test/test_ci_disabled_reasons.py` rejects a new one that does not.

## For coding agents

- Never present a guessed `est_time` as measured. Until a `/rerun-test` run
  exists for a CUDA test, use the upper bound and say in the PR description
  that it is unmeasured.
- `/rerun-test`, `/rerun-failed-ci`, and `run-ci-*` labels spend shared GPU
  runners and show up on the PR: propose the exact comment or label and wait
  for the user.
