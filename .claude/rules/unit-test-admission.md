---
paths:
  - "tests/**/*.py"
---

# Unit Test Admission Criteria

Every test case must have a concrete answer to: "what future diff would turn
this case red?" If the only answer is "editing the test itself", delete it.

A new test case must fall into one of these categories:

1. **Bug regression.** Guards a bug that actually happened (CI failure, issue,
   incident). Before committing, verify the case fails on the pre-fix code and
   passes on the fix. Describe the bug mechanism in the docstring in black-box
   terms. For concurrency bugs, reproduce the exact interleaving
   deterministically (`create_task` + `sleep(0)` scheduling, or a fake that
   yields at the racy point); do not rely on probabilistic stress -- a stress
   loop that cannot hit the bug even on the buggy code has zero guard value.
   Example: `test_concurrent_fill_does_not_duplicate_records`
   (`tests/fast/dashboard/test_partitions.py`).

2. **Derived property.** Pins down a conclusion that required reasoning to
   establish -- boundary/alignment math, CP split and reassembly, offsets and
   dtypes, invariants, protocol semantics (FIFO fairness, idempotency,
   round-trip). Protects against "looks equivalent" rewrites that silently
   break the derivation.

3. **Critical-path bookkeeping.** Defends conventions that are easy to break by
   forgetting to sync -- registry completeness, field lifecycle, serialization
   compatibility, a table that could drift from the code. Enumerating assertions
   are fine here; the guarded failure mode is "someone extended X without
   updating Y". Examples: the launcher hygiene tests under
   `tests/fast/launch_scripts/` and `tests/fast/utils/test_function_registry.py`.

Not admissible:

- Happy-path tautologies that re-assert what the implementation trivially does:
  that a field is stored on a dataclass, that `extra_env_vars["X"] == "1"`,
  that a flag appears in an argv block.
- Mirror tests that restate the implementation logic as assertions, or whose
  only evidence is `assert_called*` on a mock.
- A condition `main` already asserts -- the assert is the check; a test
  asserting the same thing moves the same statement into a second file.
- Probabilistic stress that cannot reproduce the failure it claims to guard.
- A unit test beside a launcher snapshot. The recording under
  `tests/snapshots/launch_scripts/` already covers a launcher's argv and
  runtime env once; cover the consequence instead, where one exists -- not that
  `hardware` is stored, but that the rollout profile follows it
  (`test_the_rollout_profile_follows_the_hardware`).

**Distinguishing test -- does deletion leave a silent-failure path?** A case
that *looks* like a tautology/mirror is still admissible when it guards a
failure mode no other case covers. The criterion is not "is the code under
test simple?" but "would some regression pass every remaining test if this
case were deleted?"

Keep (bookkeeping, not mirror) when the assertion guards one of:

- An **external-source literal** -- a value copied from an outside contract (a
  Megatron parameter name the weight-sync map is keyed by, an SGLang
  `ServerArgs` field, a HF config key, a logged metric name the docs quote).
  Deleting it removes the only guard against silently copying the contract
  wrong.
- A **completeness / negative-branch contract** -- "all builtins are
  registered", "a partial group is *not* accepted", "the default is applied
  when the input is absent". Even if the code is a one-liner, the failure mode
  is "someone added X without updating Y" or "a predicate degraded to
  always-true". Example: `test_partial_group_is_rejected`
  (`tests/fast/backends/training_utils/test_cp_log_prob_assembly.py`) stays
  because no positive-match test covers the rejection branch.

Delete (true mirror/tautology) when the assertion merely echoes an
**isolated** implementation output -- changing it breaks nothing outside the
line itself, so the test has no independent guard value.

One strong case beats several weak ones: each additional case must guard a
distinct failure mode. Ask "which bug escapes if I delete this case?" -- no
answer means delete it. A test is not owed to every behavior change; plenty of
correct changes ship none. What is owed is honesty in the PR body about what
was exercised and what is unvalidated.

Test name = the property (`test_partial_group_is_rejected`); docstring = the
failure it prevents, in one or two sentences; an assertion message where the
failure would otherwise be opaque.

New cases join an existing file in the same subsystem by default, at
`tests/fast/<mirror of source path>/test_<topic>.py`. Create a new file only
when it needs a different fixture, dependency, owner, or CI contract; every
file under `tests/fast/` pays a separate interpreter-import cost in
`stage-a-cpu`, which runs on every PR.

Suite cadence is part of admission: if a failing run cannot be attributed to a
single PR's diff, the test registers with `nightly=True` rather than in a
per-PR lane.

Test mechanics (placement, `register_*_ci` registration, labels, running a
suite) live in the "Registering a test" section of
[`docs/developer/contributor-guide.md`](../../docs/developer/contributor-guide.md).
