# TOP family ledger: <Family>

Copy this file into the run dir at step 0 and fill it as you go. It is the run's evidence: decisions with their
reasons, every question to the user, every run.

## Header

| Field | Value |
|---|---|
| Model, revision | |
| Checkpoint path | |
| `model_type` (trainer / serving) | |
| Miles / SGLang / Megatron SHAs at start | |
| Run dir | |
| Started (UTC) | |

## Time and cost

| Step | Started (UTC) | Ended (UTC) | GPU-hours | Notes |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 3b | | | | |
| 4 | | | | |
| 5 | | | | |

## Questions to the user

| # | When | Question | Answer | What it changed |
|---|---|---|---|---|

## Step 1: op table

Nearest family: 

Layer schedule:

| Module / surface | Verdict | Existing op or new name | Evidence (file:line on both sides) |
|---|---|---|---|

Backward reference route (1 vendor code / 2 HF / 3 op-level) and why:

Gate vehicle (layers kept, how cut, what each layer covers):

## Step 2: ops

One block per new op.

### <op name>
- Commits (Miles / SGLang):
- Program (kernel, dtypes, rounding, what is pinned):
- Kernel rule applied (1-4) and why; for rule 4, the stock kernel and which of (a)-(c) it fails:
- Backward case (i / ii / iii) and why:
- Contract table:

  | Tensor | Trainer dtype | Serving dtype | Conversion (where) |
  |---|---|---|---|

- Cells (serving / trainer), with reasons; the run qualifying each open cell:
- Refusal tests:

## Step 3: family and CPU proof

- Commits (Miles / SGLang):
- Admitted layouts:

| Suite | Commit | Pass | Fail | Notes |
|---|---|---|---|---|

Admission check (layout → admitted / refused message):

## Step 3b: Outside TOP

| File:hunk | Gate with TOP off | Effect with TOP off |
|---|---|---|

Kernel choices:

| New kernel, or stock path turned off (file:line) | Stock kernel it replaces | Reason in the code |
|---|---|---|

PR section text:

## Step 4: gate

- Gate file:
- Slice (HF repo) and upload request:

| Run | Phase | Command / args | Node | Result | Path |
|---|---|---|---|---|---|

Backward phase (layout and contract results, quality numbers, floors with their captures, mutations that failed; or route-3
results with the tolerance fixed before the run):

Before calling it green:
- [ ] every step scored the expected samples and tokens
- [ ] layout and spec recipe as the phase names
- [ ] build stamp lists the new ops' bindings

Self-written checks seen red (check, planted change, run):

Nonzero-diff investigations (first differing op, cause, fix commit):

## Step 5: perf sanity

| Arm | Layout | Rollout tok/s | Train step (s) | Notes |
|---|---|---|---|---|
| TOP | | | | |
| native | | | | |

Top kernels compared (flagged unfused / fallback paths and their share of step time):
