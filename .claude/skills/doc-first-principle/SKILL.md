---
name: doc-first-principle
description: "For code carrying a `# doc-dev: <doc>` sentinel comment, a change to documented behavior lands in the governing doc FIRST, then the code is conformed to it. Opt-in only — files without the sentinel are untouched. Use when editing a `# doc-dev:`-flagged file, or when editing a doc that flagged code names. Covers detecting the sentinel, running the doc-first sequence, and keeping doc and code in sync."
user_invocable: true
---

# Doc-First Principle

For code that opts in via a `# doc-dev:` sentinel comment, the governing doc is the spec and **moves first**: when you change documented behavior in a flagged file, you update the doc before touching the code, then conform the code to the updated doc. Files without the sentinel carry no such obligation — this is off by default.

Before you start, be clear on the change you intend and on what "the code now matches the updated doc" will concretely mean (which behavior, signature, or invariant). Keep both edits minimal: the doc change is exactly the spec delta, and the code change implements only what the updated doc specifies — no incidental rewrites of neighboring sections, and no fallbacks, guards, or silent skips bolted on just to make the code look conformant.

## The flag — the only thing that turns this on

This applies to a file **only** if it carries the sentinel.

- Format: a single-line comment in the file's own comment syntax, naming the governing doc by a repo-root-relative path:
  - `# doc-dev: docs/design/scheduler.md` (Python / shell / YAML)
  - `// doc-dev: docs/design/scheduler.md` (C / C++ / Go / Rust / JS / TS)
- Scope is **file-level**: the sentinel sits near the top of the file and governs the whole file. A file may carry more than one sentinel line, one per governing doc.
- The trigger is the in-code comment itself — there is no manifest, no external config, no global registry. Never add the sentinel to a file just to pull it under this standard; whether a file opts in is its owner's choice.

## Detection — two entry points

**Editing code (primary).** Before editing a code file, grep it for the `doc-dev:` marker.

- No sentinel → this standard does not apply; edit as usual.
- Sentinel present → the named doc governs this file; follow the doc-first sequence below.

The trigger lives in the file at the edit site, so there is nothing else to scan.

**Editing a doc (reverse).** When you edit a doc directly and the change alters documented behavior or a contract, grep the codebase for a `# doc-dev: <this-doc>` sentinel naming that doc. If any file flags it, conform that flagged code in the same pass — never land the doc change and leave flagged code stale. This direction is a codebase-wide grep by design, because the flag lives on the code side.

## Doc-first sequence (flagged file)

When a change to a flagged file touches **documented behavior or a contract**:

1. **Doc first.** Land the spec change in the governing doc named by the sentinel — a small wording correction or a larger design revision, whatever the change requires. Do not touch the code yet.
2. **Conform the code** to the now-updated doc, using your normal development process scaled to the blast radius: a contained edit for a small change, a full refactor for cross-module, contract, or state-flow work.
3. **Verify** the flagged code matches the updated doc — observable behavior, signature, or invariant — and that the sentinel still names the right doc.

A **strictly behavior-preserving** edit in a flagged file (rename, extract, reflow with no change to documented behavior) needs no doc edit. But before landing it, re-confirm the governing doc still describes the new code, and fix the sentinel path if the file or the doc moved.

## Core discipline

- **Opt-in only.** Governs only files carrying the `doc-dev:` sentinel. Never apply doc-first to unflagged code.
- **Doc leads.** In a flagged file, a documented-behavior change lands in the doc before the code — never edit flagged code while its doc stays stale.
- **Never desync.** Don't change the doc's behavior without conforming the flagged code in the same pass, and don't change the flagged behavior without first moving the doc.
- **Don't over-edit the doc.** The doc change is exactly the spec delta, not a rewrite of surrounding sections.
- **Conform, don't over-build.** The code implements only what the updated doc specifies for the area in scope.

## When this does NOT apply

- **The doc is already correct and the code merely drifted** (no spec change). Just conform the code to the existing doc — there is no doc-first write to do, because the spec did not move.
- **Neither doc nor code is authoritative** and you are reconciling existing drift, deciding a direction per divergence. That is a separate reconciliation task, not doc-first.
- **The file is unflagged.** Edit it as you normally would; this standard stays silent.
- **A brand-new doc with no code yet.** That is design-doc writing, out of scope here.

## Anti-patterns

- Applying doc-first to a file that carries no sentinel.
- Adding the `doc-dev:` sentinel just to pull a file under this standard.
- Editing flagged code first and updating the doc afterward ("the doc catches up") — the doc leads.
- Landing a documented-behavior change in the code while leaving the governing doc stale.
- Reintroducing a manifest or external config as the trigger — the in-code sentinel is the trigger by design.
