# CI Failure Triage

What to do when a CI check on a PR goes red. Read the failing job's log first;
the `ci-fetch-log` skill pulls complete logs. Most failures fall into one of two
buckets.

## Likely the PR's change

Fix it locally before re-running.

| Signal | Meaning |
|---|---|
| `ImportError` / `ModuleNotFoundError` / `SyntaxError` / `NameError` | the code does not load |
| `pre-commit` job red | formatting or lint; run `pre-commit run --all-files`, or comment `/run-lint` |
| the new test's own assertion fails | reproduce with `python3 tests/e2e/.../test_x.py` |

## Likely infrastructure

Re-run the job once first (Actions UI or `/rerun-failed-ci`); transient issues
clear on retry. If it reproduces, report it.

| Signal in the log | Cause |
|---|---|
| Job stuck `Queued`, never starts | runner pool busy or a runner is offline |
| `nvidia-smi did not become ready after 120s` / CUDA error 802 | GPU subsystem not ready on the runner |
| `ENOSPC` / model or dataset download failure | runner disk full |
| Job needed N GPUs but ran with fewer | scheduling landed it on a smaller runner |
| A test unrelated to the PR fails an accuracy, score, or latency assertion, then passes on re-run | flaky test |

If the failure is in code or tests the PR did not touch, and a re-run behaves
differently, it is infra or a flake, not the PR.

## Report an infra failure

When a re-run still shows an infra signal, open a GitHub issue with:

- the failing job URL;
- the runner name (`runner_name`, at the top of the job log);
- the suite / stage (e.g. `stage-c-4-gpu-h200`) and the step that failed;
- the log line carrying the infra signal;
- what was already tried (e.g. "re-ran twice, same `ENOSPC`").

A maintainer maps the runner to its host; no runner access is needed. The
`#miles-rl` channel of the SGLang Slack is fine for a quick sanity check, but
the issue is the tracked record.

## Report a flaky test

A flaky test fails non-deterministically, usually on a numeric, accuracy, or
timing assertion, and passes on a re-run with no code change. CUDA PR CI runs
each test once, so a flake fails the check. Report a test that flakes
repeatedly in a GitHub issue with:

- the test file path (e.g. `tests/e2e/megatron/test_x.py`);
- the failing `AssertionError` line;
- run URLs for a passing and a failing run, when available.

Quarantining it with `disabled=` follows `ci-test-registration.md`.

## For coding agents

- Classify a failure only with its log line in hand; never call it infra or a
  flake without a signal from the tables above.
- Re-running jobs, `/rerun-failed-ci`, and filing issues spend shared runners or
  show up on GitHub: propose the exact action and wait for the user.
