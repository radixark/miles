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
call. They are written only here; `docs/developer/ci/contributor-guide.md`
explains the mechanics around them (discovery, labels, cadences, reading a red
check).

## Placement

- A `test_*.py` lives under `tests/fast`, `tests/fast-gpu`, `tests/e2e`, or
  `tests/ci`; no CI job collects anything else, and
  `tests/ci/test/test_ci_discovery_coverage.py` rejects a tracked test file
  elsewhere. `tests/manual/` is only for tests no CI job should run.
- A GPU test file pays for a model download, Ray startup, and an engine launch
  on every run. Add cases to an existing file with the same launch; a new file
  is for a new fixture, model, or parallel layout.

## Stage

A test holds every GPU of its stage for its whole run, and the wider stages have
fewer runners. Walk the table from the top and stop at the first stage that can
run the test; copying a neighbouring test's `suite=` is not a reason.

| Order | Where | Runs on | Use for |
|---|---|---|---|
| 1 | `tests/fast/` (no declaration; suite `stage-a-cpu`) | GitHub-hosted CPU | pure-Python / CPU-only tests that finish in seconds |
| 2 | `register_cpu_ci(..., suite="stage-b-cpu")` | GitHub-hosted CPU | CPU tests that take minutes; `stage-a-cpu` gates every GPU stage, so it stays fast |
| 3 | `tests/fast-gpu/`, suite `stage-b-2-gpu-h200` | 2× H200 | short GPU checks (kernels, quantizers, worker entry points) that finish in a few minutes |
| 4 | `stage-c-2-gpu-h200` | 2× H200 | end-to-end runs that fit on two GPUs |
| 5 | `stage-c-4-gpu-h200` | 4× H200 | layouts that need four ranks; the busiest stage, so confirm two GPUs cannot express the case |
| 6 | `stage-c-8-gpu-h200` or `stage-c-8-gpu-h100` | 8× H200 / 8× H100 | layouts that need eight ranks |
| — | `stage-c-8-gpu-b200` | 8× B200 | only tests that cannot run on Hopper (`hardware=["blackwell"]`), at any GPU count |

The stage's GPU count equals the count the test requests (`ray start
--num-gpus`, `--actor-num-gpus-per-node`, `torchrun --nproc-per-node`): a 4-GPU
test on an 8-GPU stage idles four GPUs for its whole run. `stage-c-8-gpu-b200`
is the exception, because the Blackwell fleet is a single unpartitioned host.

## Declaration

```python
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=500,                      # measured; see est_time below
    suite="stage-c-2-gpu-h200",        # home stage, from the table above
    labels=["megatron"],               # domain labels that select it on a PR
    hardware=["hopper", "blackwell"],  # supported CUDA generations
)
```

- A GPU file has one top-level `register_cuda_ci(...)`, plus a
  `register_rocm_ci(...)` when it also covers the MI350 lane. The calls are
  parsed from the AST: top-level, literal arguments, no alias. `tests/fast/`
  files need none.
- `labels`: reuse a domain label from `tests/ci/labels.py`. A new label also
  needs a `run-ci-<key>` repository label, which only a maintainer can create,
  and an entry in `.github/workflows/policies/comment-command-access.json` to
  be addable from PR comments.
- `hardware`: list a GPU generation only when the code paths the test runs
  support it (`--sglang-attention-backend fa3` is Hopper-only; nvfp4 / mxfp8
  kernels are Blackwell-only). The generation of the home `suite` comes first.

## `est_time`

`est_time` balances shards and sets the per-file timeout,
`max(1800 s, 1.25 × est_time)`, in stage runs and in `/rerun-test` alike. It is
measured, never copied from a neighbouring test:

1. Before the first run, use an upper bound that safely exceeds the expected
   runtime, so the timeout does not kill it. Up to 1440 s the 1800 s floor
   applies anyway.
2. Before the PR merges, comment `/rerun-test <path/to/test_file.py>` on it. It
   runs only that file, on its registered suite's runner, at the PR head; it
   needs no domain label and writes no performance baseline. Posting it takes a
   merged commit in `radixark/miles` (a first-time contributor asks a
   maintainer); a fork head gets no `WANDB_API_KEY` or `HF_TOKEN`; a `disabled`
   registration cannot run this way.
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
5. Add one line per new or moved test to the PR description:
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

## Before finishing

From the repo root, run
`python3 tests/ci/run_suite.py --hw <cpu|cuda> --suite <suite> --match-all-labels --list-only`
and confirm the file is listed under `Enabled N test(s)`.

## For coding agents

- Never present a guessed `est_time` as measured. Until a `/rerun-test` run
  exists, use the upper bound and say in the PR description that it is
  unmeasured.
- `/rerun-test`, `/rerun-failed-ci`, and `run-ci-*` labels spend shared GPU
  runners and show up on the PR: propose the exact comment or label and wait
  for the user.
