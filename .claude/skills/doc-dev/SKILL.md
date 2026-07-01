---
name: doc-dev
description: "Keep a file consistent with its governing doc(s). A file opts in via a `# doc-dev:` sentinel comment or by explicit invocation. Any `# doc-dev:` marker binds the file's own header doc; a `# doc-dev: <path>` marker adds a central requirement / reference doc — code and doc must agree on what it states, and which side moves is set by the mode. Co-evolution is the default (build the code, then update the doc); pass `--docfirst` for doc-first (the doc is the authority and the code conforms, stopping to ask before rewriting a contract). Opt-in only. Header format: one point per line, caveats last. Use when editing a `# doc-dev:`-flagged file or a doc a `doc-dev:` sentinel names."
user_invocable: true
---

# Doc Dev

Keep a file **consistent** with its governing doc(s): once a change touches documented behavior, the code and every governing doc agree by the time the change lands. A file is governed only when it **opts in** — either passively, by carrying a `# doc-dev:` sentinel comment, or actively, when you invoke this skill on it. A file that does neither is untouched.

A governing doc is one of two kinds, bound in layers:

- **The file's own header doc** — the block at the top of the file stating its function, responsibility boundary, inputs/outputs, and general design. Any marker — and any active invocation — puts this header under the contract: it must keep accurately describing the code below it. Consistency here means **descriptive accuracy**.
- **A central doc** — a separate requirement or reference doc, named by a repo-root-relative path (`# doc-dev: <path>`) or named directly when you invoke the skill. Consistency here means **the code and the doc agree on what it states**; which side moves to restore agreement is set by the mode.

## The flag and the invocation — the only things that turn this on

Governs a file **only** if it opts in, one of two ways:

- **Sentinel (passive).** A single-line comment in the file's own comment syntax:
  - `# doc-dev:` — bare, no path. Opts in; only the file's own header doc governs.
  - `# doc-dev: docs/design/scheduler.md` — additionally binds that central doc (the header still governs). The path is repo-root-relative; a file may carry several path lines (one per central doc).
  - `// doc-dev: docs/design/scheduler.md` for `//`-comment languages (C / C++ / Go / Rust / JS / TS).
- **Invocation (active).** You run this skill on a specific file. An invocation binds the file's header doc exactly as a marker does; naming a doc additionally binds that central doc, naming none binds the header only. This is a one-time opt-in for that change; it does not add a sentinel.

Never add a sentinel just to pull a file under this standard.

## Detection — two entry points

**Editing code (primary).** Before editing a code file, grep it for the `doc-dev:` marker.

- No sentinel and no active invocation → this standard does not apply; edit as usual.
- Sentinel present, or the file is the target of an invocation → keep the file's **header** consistent, and keep the code and any bound **central doc** in agreement.

**Editing a doc (reverse).** When you edit any doc and the change alters documented behavior or a contract, grep the codebase for `doc-dev: <this doc's repo-root path>` (match any comment syntax — `#`, `//`, …). Any hit makes it a central doc: conform the flagged code in the same pass — never land the doc change and leave flagged code stale. (Editing a header doc is inherently local — you are already in the governed file, so there is nothing to grep.)

A sentinel path that no longer resolves is a **stop-and-ask**, never silently ignored. When moving or renaming a central doc, grep for the **old** path and retarget every sentinel naming it in the same pass.

## Reaching consistency — two modes

When a change touches **documented behavior, a contract, or the file's described responsibility**, bring every governing doc back into agreement before the change lands, in one of two modes:

**Co-evolution (default) — the shape emerges as you build.** Fits ordinary development and the header doc.

1. Write and iterate the code to discover the working shape; the doc may lag while you do.
2. Once the shape settles, update the governing doc to match — re-describe the header freely; if a **central doc**'s requirement moved, that is a contract change: surface it and get confirmation before landing that doc edit.
3. Verify the doc and code agree, and that the sentinel still names the right central doc(s).

**Doc-first (opt in with `--docfirst`) — the doc is the authority, the code conforms.** Fits a central requirement doc, or any time you declare a doc authoritative and want the code made to match; not for exploratory work where the shape is not yet known.

1. If the spec itself is changing, land the spec change in the governing doc first and do not touch the code yet. If the doc already states the target and only the code drifted, there is no doc delta — skip straight to step 2.
2. Conform the code to the doc, scaled to the blast radius: a contained edit for a small change, a full refactor for cross-module, contract, or state-flow work.
3. If conforming would change a public contract, break other callers, the doc itself looks stale or ambiguous, or two governing docs disagree with each other, **stop and ask which side is authoritative** rather than conforming blindly.
4. You may remove code the doc never mentions — surface such code and ask; never delete it silently.
5. Verify the code matches the doc, and that the sentinel still names the right central doc(s).

A **strictly behavior-preserving** edit (rename, extract, reflow with no change to documented behavior or the described responsibility) needs no doc edit. Before landing it, re-confirm the header still describes the code and any central doc still holds, and fix a sentinel path if a file or doc moved.

## Header-doc format

Write the header point-by-point:

- State the main logic first — what the file does, its boundary, IO, and design — one point per bullet or single-sentence line, at most two sentences per point.
- Usage / example command blocks stay verbatim; they are not re-flowed into bullets.
- Collect caveats, warnings, and do-not notes at the end of the header block; keep them few, each guarding a real, current mistake.
- Apply this format when writing a new header or updating one the change already touches; a header that already reads point-by-point is not re-flowed.
- A governed file with no header block is left without one — govern accuracy only where a header exists; adding a header is the owner's choice, not a retro-documentation duty.
- This format governs the header block only; inline comments below it are ordinary comments, out of scope here.

## Core discipline

- **Judge staleness before conforming.** A code↔doc mismatch alone does not prove the code wrong — a reference doc lagging an intentional code change is doc-side staleness. Decide which side is stale first; when unsure, ask.
- **Existing drift is surfaced, not absorbed or ignored.** The conform obligation is scoped to the current change. On stumbling over prior drift in a governed file, surface it to the user; do not silently expand a small change into a conform sweep, and do not leave the drift unmentioned. Drift the user already declared in scope (a `--docfirst` conform target) is the current change itself, not stumbled-on drift.
- **Do not use co-evolution to dodge specifying a requirement you already know** — pass `--docfirst` for that.
- **Edit the delta, conform to scope.** A doc edit is exactly the spec / description delta, not a rewrite of surrounding sections; a code conform implements only what the doc specifies for the area in scope.
- **Never fake conformance.** A conform is a real code change — no fallback, guard, or silent skip added just to make code look conformant.

## When this does NOT apply

- **Neither doc nor code is authoritative** and you are reconciling existing drift — that is a separate reconciliation task: decide a direction per divergence with the user.
- **A brand-new central doc with no code yet** — that is design-doc writing, out of scope here.

## Anti-patterns

- Treating a `# doc-dev: <path>` file as governed by the central doc only, and letting its header drift out of sync with the code.
- Passing `--docfirst` on genuinely exploratory work where the shape is not yet known.
- (For maintainers of this skill, not executors:) reintroducing a manifest or external config as the trigger — the in-code sentinel is the only trigger.
