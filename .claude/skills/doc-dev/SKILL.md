---
name: doc-dev
description: "Keep a file consistent with its governing doc(s). A file opts in via a `# doc-dev:` sentinel or by explicit invocation. Any `# doc-dev:` marker binds the file's own header doc (function / boundary / IO / design; the header still describes the code); a `# doc-dev: <path>` marker adds a central requirement doc (the code conforms to it). Default co-evolution (build code, then update the doc); pass `--docfirst` for doc-first (doc is authority, conform code, ask before rewriting a contract). Opt-in only. Triggers: editing a `# doc-dev:`-flagged file, \"doc dev\", \"doc code consistency\"."
user_invocable: true
---

# Doc Dev

Keep a file **consistent** with its governing doc(s): once a change touches documented behavior, the code and every governing doc agree by the time the change lands. A file is governed only when it **opts in** — either passively, by carrying a `# doc-dev:` sentinel comment, or actively, when you invoke this skill on it and name an authoritative doc. A file that does neither is untouched.

A governing doc is one of two kinds, bound in layers:

- **The file's own header doc** — the block at the top of the file stating its function, responsibility boundary, inputs/outputs, and general design. **Any** `# doc-dev:` marker puts this header under the contract: it must keep accurately describing the code below it. Consistency here means **descriptive accuracy**.
- **A central doc** — a separate requirement / target spec, named by a repo-root-relative path (`# doc-dev: <path>`) or named directly when you invoke the skill. The code must conform to the requirement it states. Consistency here means **requirement conformance**.

A bare `# doc-dev:` binds only the header; `# doc-dev: <path>` binds the header **and** that central doc. A file may carry several path lines (one per central doc); the header is always included once any marker is present.

## Reaching consistency — two modes

The invariant is the endpoint, not the order: once the change lands, the code and every governing doc are consistent. Which side moves first is a mode, and **co-evolution is the default**:

- **Co-evolution (default).** The shape emerges as you build. Write and iterate the code first, then update the doc to match — before the change lands. Fits ordinary development and the header doc, which describes code you just shaped.
- **Doc-first (opt in with `--docfirst`).** The doc is the authority and the code conforms to it. Land the spec in the doc first, then bring the code into conformance. Fits a central requirement doc, or any time you declare a doc authoritative and want the code made to match.

With no flag the skill runs co-evolution; invoke it with `--docfirst` to switch to doc-first.

## The flag and the invocation — the only things that turn this on

Governs a file **only** if it opts in, one of two ways:

- **Sentinel (passive).** A single-line comment in the file's own comment syntax:
  - `# doc-dev:` — bare, no path. Opts in; only the file's own header doc governs.
  - `# doc-dev: docs/design/scheduler.md` — additionally binds that central doc (the header still governs). The path is repo-root-relative.
  - `// doc-dev: docs/design/scheduler.md` for `//`-comment languages (C / C++ / Go / Rust / JS / TS).
  A file may carry more than one path line. Never add a sentinel just to pull a file in — opting in is its owner's choice.
- **Invocation (active).** You run this skill on a specific file and name the authoritative doc for it. This is a one-time opt-in for that change; it does not add a sentinel.

The **header doc** is the description block at the very top of the file (docstring / header comment) covering the file's function, responsibility boundary, inputs/outputs, and general design.

## Detection — two entry points

**Editing code (primary).** Before editing a code file, grep it for the `doc-dev:` marker.

- No sentinel and no active invocation → this standard does not apply; edit as usual.
- Sentinel present → keep the file's **header** consistent (always), and conform the code to any **central doc** the marker names.

**Editing a doc (reverse).** When you edit a **central doc** directly and the change alters documented behavior or a contract, grep the codebase for a `# doc-dev: <this-doc>` sentinel naming it. If any file flags it, conform that flagged code in the same pass — never land the doc change and leave flagged code stale. (Editing a header doc is inherently local — you are already in the governed file, so there is nothing to grep.)

## Reaching consistency (governed file)

When a change touches **documented behavior, a contract, or the file's described responsibility**, bring every governing doc back into agreement before the change lands. Co-evolution is the default; pass `--docfirst` for doc-first.

**Co-evolution (default) — the shape emerges as you build.**

1. Write and iterate the code to discover the working shape; the doc may lag while you do.
2. Once the shape settles, update the governing doc to match — the header to re-describe the code, and any central doc if the requirement itself moved — before the change is done.
3. Verify the doc and code agree, and that the sentinel still names the right central doc(s).

**Doc-first (`--docfirst`) — the doc is the authority, the code conforms.**

1. Land the spec change in the governing doc — for a central doc, the requirement; for the header, the described contract. Do not touch the code yet.
2. Conform the code to the updated doc, scaled to the blast radius: a contained edit for a small change, a full refactor for cross-module, contract, or state-flow work.
3. If conforming would change a public contract, break other callers, or the doc itself looks stale or ambiguous, **stop and ask which side is authoritative** rather than conforming blindly.
4. You may remove code the doc never mentions — surface such code and ask; never delete it silently.
5. Verify the code matches the doc, and that the sentinel still names the right central doc(s).

A **strictly behavior-preserving** edit (rename, extract, reflow with no change to documented behavior or the described responsibility) needs no doc edit. Before landing it, re-confirm the header still describes the code and any central doc still holds, and fix a sentinel path if a file or doc moved.

## Core discipline

- **Opt-in only.** Governs a file only when it carries the `doc-dev:` sentinel or is the explicit target of an invocation. Never apply to a file that opted into neither.
- **The header is always in scope; a central doc is added by a path.** Any marker binds the header; a `# doc-dev: <path>` line binds that central doc on top.
- **Consistency is the endpoint.** Once a change touches documented behavior, code and every governing doc agree — never leave the header stale or a central doc unmet after the code settles, whichever side moved first.
- **Co-evolution is the default; `--docfirst` opts into doc-first.** Run co-evolution unless `--docfirst` is given. Do not use co-evolution to dodge specifying a requirement you already know — pass `--docfirst` for that.
- **Doc-first is bounded.** Conform to the doc up to the point where it would rewrite a public contract or the doc itself looks wrong; past that, ask instead of conforming.
- **Don't over-edit the doc.** A doc change is exactly the spec / description delta, not a rewrite of surrounding sections.
- **Conform, don't over-build.** The code implements only what the doc specifies for the area in scope.

## When this does NOT apply

- **The doc is already correct and the code merely drifted** (no spec change). Just conform the code to the existing doc — the spec did not move.
- **Neither doc nor code is authoritative** and you are reconciling existing drift, deciding a direction per divergence. That is a separate reconciliation task.
- **The file opted into neither** a sentinel nor an invocation. Edit it as usual; this standard stays silent.
- **A brand-new central doc with no code yet.** That is design-doc writing, out of scope here.

## Anti-patterns

- Applying this to a file that carries no sentinel and was not explicitly targeted.
- Adding the `doc-dev:` sentinel just to pull a file under this standard.
- Treating a `# doc-dev: <path>` file as governed by the central doc only, and letting its header drift out of sync with the code.
- Landing a documented-behavior change on one side — doc or code — and leaving the other stale once the change is done.
- Using co-evolution as an excuse to ship code and never update the header or central doc.
- Passing `--docfirst` on genuinely exploratory work where the shape is not yet known.
- Silently rewriting a public contract to match the doc instead of stopping to ask.
- Silently deleting code the doc does not mention instead of surfacing it.
- Over-editing a doc beyond its delta, or bolting on fallbacks / guards / silent skips to fake conformance.
- Reintroducing a manifest or external config as the trigger — the in-code sentinel is the trigger by design.
