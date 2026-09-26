---
paths:
  - "**/*.py"
  - "**/*.cu"
---

# Comment Style

Applies to `#` and `//` comments and to Python docstrings. Comments are reviewed
like code, with the burden of proof reversed: the author justifies a comment's
existence, not the reviewer its removal.

## Two modes

Writing a comment and editing someone else's are different decisions.

**Writing.** You still have the context that made the comment necessary, so the
judgment is reliable. Everything below `## Cleanup` is written for this case.

**Editing.** The judgment is unreliable and the error is one-way. A one-line
comment carries too little text to tell a restatement apart from the last anchor
for a cross-file fact -- deciding takes the whole function and its callers, more
context than the line costs. And the payoff is asymmetric: keeping a useless
one-liner costs a line of scroll, while deleting a load-bearing one deletes a
fact silently, inside a diff of two hundred deletions where no reviewer will
catch it. So: a bright line, not a judgment call.

> **A comment-only diff does not touch one-line comments.** It removes or
> condenses multi-line prose blocks. A one-liner is rewritten only for a reason
> of its own -- it is wrong, it is stale, or it is commented-out code -- never
> because the line below it says the same thing.

## Cleanup: what a comment-only diff may remove

- **Multi-line prose restating the code** -- the function name, the next line,
  the branch condition, or the loop body, written out as a paragraph.
- **Python `Args:` / `Returns:` blocks** anywhere except a launcher's module
  docstring -- see `## Documentation blocks`.
- **Multi-line history and rationale** -- what the code used to do, why the
  change was made, which approach was abandoned, how the bug was found. Each of
  these has a home elsewhere (see `## Where other explanations live`).
- **Commented-out code**, at any length.

Condense rather than delete when a block carries one fact that is not
recoverable: keep that sentence, drop the enumeration around it.

## What a comment is for

State the fact a reader cannot see from here. Delete the comment and ask what it
costs to recover: a fact that lives in another file costs everything, a grouping
the names only half-encode costs a tedious reconstruction, a restated line costs
nothing. The test for a new comment: would the line look wrong or arbitrary
without it? If not, do not write it; if yes, write the one fact that makes the
line look right.

- **Cross-boundary constraints.** A name, layout, or ordering shared with the
  SGLang engine, a Megatron checkpoint, a Ray actor on another node, or a
  recorded launcher snapshot. Nothing in this file shows the other side, so
  nothing else can warn the next editor.

  ```python
  # a leaked knob would make later recordings depend on which launcher ran first
  ```
  (`tests/fast/launch_scripts/py_harness.py`)

- **Units and layout the name cannot carry.** Tokens vs samples vs groups vs
  bytes, and tensor shape/dtype/layout. Encode it in the name first; comment
  only when the name is fixed by an existing interface.

  ```python
  # [num_tokens, num_kv_heads, head_dim], bf16, THD-packed.
  ```

- **Where a magic number came from** -- not what it means. A hardware
  constraint, a measurement, or an admission that it was picked arbitrarily.
  The last one is the most valuable: it tells the next person the value is safe
  to change.

  ```python
  # Hardware constraint: the fused kernel requires 16B-aligned rows.
  # Measured on the 4-layer CI recipe; re-tune when the recipe changes.
  # Arbitrary; no evidence this is the right threshold.
  ```

- **Workarounds, anchored to a verifiable reference and a retirement
  condition.** An unanchored workaround is immortal -- nobody can prove it is
  safe to delete. Delete the pin and its comment in the PR that bumps past it.

  ```python
  # Colocated multi-engine init can deadlock in the multimem all-gather rendezvous (sgl-project/sglang#36110).
  # Workaround for pytorch/pytorch#12345; drop once we require torch >= 2.9.
  ```
  (first line: `scripts/run_deepseek_v4.py`)

- **Contracts that are a decision, not a mechanism** -- a sentinel's meaning, a
  deliberate omission from a list, an ordering that looks incidental, an
  invariant two code paths rely on.

  ```python
  # setdefault: TP peers share a cp_rank; keep the replica choice order-independent
  # read+parse release the GIL; two fills from one stale offset append the bytes twice
  ```
  (`miles/dashboard/dump_reader.py`, `miles/dashboard/store.py`)

- **Structure the names only partly encode** -- a group boundary in a long flat
  block, a section split in a long body. `# ===== Helpers =====` above two
  functions costs nothing to see past; the same banner splitting a genuinely
  long flat module is the only statement of where one group ends.

Not worth writing, at any length: prose that restates the line below it,
`# Step 1:` / `# Step 2:` numbering over straight-line code (extract named
helpers if the flow needs numbers), the names of the callers (that is what grep
is for), a design decision the PR body already argues, a fact the adjacent log
line or docstring already states, and hedging -- "this should probably be
revisited" is either a fact to establish and state, or a `TODO`.

## Form and tags

- **One line by default, two at most.** A comment that wants a third line is
  usually two facts, or one fact with its history attached; keep the constraint,
  cut the story. A genuinely intricate invariant may run longer; that is a rare
  exception, not a licence. Documentation blocks are the separate case below.
- **ASCII and English only.** No Unicode arrows, math symbols, or CJK. Spell a
  dash `--`.
- **Fragment or sentence, not both.** A lowercase fragment takes no period; a
  full sentence takes sentence case and a period. No first person, no hedging.
- **Break at a clause boundary.** If a comment needs a second line, wrap after a
  semicolon or a comma -- never mid-phrase. A sentence that will not split
  cleanly is a sentence that should be shortened instead.
- **Attach the comment to the block it constrains** -- above the `if`, not
  beside one statement inside it, and not in a preamble at the top of the
  function. Comments collected into a preamble are the ones that go stale.
- **You change the line, you own its comment.** Update it or delete it -- never
  leave it orphaned. A stale comment is worse than no comment.
- **`# copied from <source>` is one line, directly above the copied body.**

Two tag spellings only. New code does not use `FIXME`, `XXX`, or `HACK`; existing
occurrences are grandfathered and get folded into these when the line is touched.

- `# NOTE:` -- a constraint or trap. Most of the time the prefix adds nothing;
  drop it and just state the fact.
- `# TODO:` -- planned work the change deliberately leaves open, stated as the
  gap in one line, with the issue when one exists (`# TODO(#2591): ...`) or an
  owner (`# TODO(<gh-handle>): ...`). `# TODO: fix this` is rejected in review:
  a TODO nobody can act on is never retired. A TODO the change itself resolves
  is deleted with the code it marked; open ones stay.

## Where other explanations live

Every explanation has exactly one home. A second copy in a comment drifts from
the first.

- **What the code used to do -> `git log`.** Comments describe the current
  state: not what was tried first, not which approach was abandoned, not what
  the old name was. The exception is a past failure that is still a live
  constraint -- write it as the constraint, not as the story.

  ```python
  # Bad:  New Megatron renamed wq_a -> linear_q_down_proj, so the plugin maps it here.
  # Good: Megatron's name; the weight-sync map is keyed by it.
  ```

- **Why the change was made -> the PR body.** Why now, what else was tried,
  which benchmark moved, how the bug was found. "Now we also handle the case
  where ..." argues for a diff, and is written for a reviewer who is long gone.
- **Design rationale -> `docs/` or a module docstring. CLI semantics -> the
  `help=` string in `miles/utils/arguments.py`. Test intent -> the test.**
- **Superseded code -> `git show`.**

Also out: review attribution ("as suggested in review") and our own PR numbers
used as a changelog. An upstream issue URL is different -- it is not a changelog
entry but a workaround's retirement condition, and the section above requires it.

What stays next to the line is why *this line* exists, written for a reader who
has neither the PR nor the discussion.

## Documentation blocks

Python docstrings are part of the product on the surfaces users, integrators,
and tooling read, and noise everywhere else. A docstring is a summary line and at
most one short paragraph; identifiers go in backticks; numbers belong in the PR
body, not the docstring. Where warranted a block may run past two lines; every
other rule in this file still applies.

- **Yes:** the extension points a plugin or a user's own code implements -- the
  base types and protocols under `miles/rollout/` (`base_types.py`,
  `data_source.py`) and `miles/backends/training_utils/`, and the hooks
  documented in `docs/user-guide/customization.md` -- where the docstring is
  the contract a third party codes against.
- **Yes:** the module docstring of a launcher under `scripts/` or `examples/`,
  in the shape `.claude/rules/launch-and-model-scripts.md` prescribes: what it
  trains, what must already exist, an `Args:` block for its flags, one runnable
  invocation. A launcher is a CLI and this docstring is its `--help`; it is the
  one place an `Args:` block belongs.
- **Yes:** a bug-regression test, stating in one or two lines which black-box
  behavior must not come back -- the live constraint, not the incident. Root
  cause and repro belong in the issue and the PR
  (see `.claude/rules/unit-test-admission.md`). A test module docstring is one
  line.
- **No:** internal helpers, overrides, private methods, trivial `_helpers`.
  Never hand-write or generate `Args:` / `Returns:` blocks outside the launcher
  docstring -- the signature and its type hints already carry the names and
  types. Cut design rationale, alternative designs, and "which is how X uses
  it" tours; those live in `docs/` or the PR body.
