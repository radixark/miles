---
name: doc-dev
description: "Keep opt-in files consistent with their governing documentation. Use when editing a file with a `doc-dev:` sentinel, editing a document named by such a sentinel, or when explicitly asked to keep a file and its documentation in sync. Any marker binds the file's header documentation; a marker with a repo-relative path also binds a central document. Default to co-evolution; use `--docfirst` when the document is authoritative."
---

# Doc Dev

Keep a file consistent with its governing documentation. Once a change touches documented behavior, the code and every governing document must agree before the change lands. A file participates only through a `doc-dev:` sentinel or an explicit invocation; leave every other file outside this workflow.

Before editing, identify the intended behavior, the governed files and documents, and the concrete evidence that will show they agree.

## Governing documentation

A governed file can have two layers of documentation:

- **Header documentation:** The top-of-file block that describes the file's function, responsibility boundary, inputs, outputs, and general design. Any sentinel or explicit invocation binds an existing header, which must continue to describe the code accurately.
- **Central documentation:** A separate requirement or reference document named by a repo-root-relative sentinel path or by the invocation. The code and document must agree on every statement in scope; the selected mode determines which side moves.

## Opt in

Use either form; there is no manifest or external registry:

- **Sentinel:** Put a single-line marker in the file's own comment syntax. `# doc-dev:` binds only the header. `# doc-dev: docs/design/scheduler.md` binds the header and that central document. Use `// doc-dev: docs/design/scheduler.md` in languages with `//` comments. A file may name several central documents, one marker per path.
- **Explicit invocation:** Run this workflow for a specific file. Naming no document binds only its header; naming a document also binds that central document. This opt-in lasts for the current change and does not add a sentinel.

Never add a sentinel merely to bring a file under this workflow; opting in is the file owner's decision.

## Detect both directions

When editing code, search the file for `doc-dev:` before changing it. If no sentinel exists and the file was not explicitly targeted, continue without this workflow. If it is governed, keep its header accurate and keep the code aligned with every named central document.

When editing a document in a way that changes documented behavior or a contract, search the repository for sentinels naming that document's repo-root-relative path. Update every matching file in the same change so the document never lands with governed code left stale.

Stop if a sentinel path does not resolve. When moving or renaming a central document, search for its old path and retarget every matching sentinel in the same change.

## Restore consistency

Choose one mode whenever a change affects documented behavior, a contract, or the responsibility described by the file header.

### Co-evolution (default)

Use co-evolution when the working shape emerges during implementation:

1. Write and iterate on the code; the documentation may temporarily lag while the design is still moving.
2. Once the implementation settles, update the existing header to describe it accurately. If a central requirement also changed, surface that contract change and obtain approval before landing the document update.
3. Verify that the code, header, central documents, and sentinel paths all agree.

### Doc-first (`--docfirst`)

Use doc-first when a central document is authoritative or the target behavior is already known:

1. If the specification is changing, update the governing document before editing code. If the document already states the target behavior and only the code drifted, skip this write.
2. Conform the code to the document, scaling the implementation to the actual blast radius.
3. Stop and ask which side is authoritative if conformance would change a public contract, break callers, expose an ambiguous or stale document, or conflict with another governing document.
4. Surface code that the document never mentions before removing it; never delete it silently.
5. Verify that the resulting code, header, central documents, and sentinel paths all agree.

A strictly behavior-preserving change, such as a rename, extraction, or reflow, needs no document edit. Reconfirm that the header and central documents remain accurate, and update sentinel paths if a file or document moved.

## Maintain header documentation

- State the main logic first: what the file does, its boundary, inputs and outputs, and its design. Use one point per bullet or single-sentence line, with at most two sentences per point.
- Keep usage examples and command blocks verbatim.
- Put a small number of current caveats, warnings, and prohibitions at the end of the header.
- Apply this format only when creating or already updating the header. Do not reflow a header that already reads point by point.
- If a governed file has no header block, do not add one automatically. Header creation remains the owner's choice.
- Treat inline comments below the header as ordinary comments outside this workflow.

## Discipline

- Judge which side is stale before conforming. A mismatch alone does not prove the code is wrong; ask when the intended authority is unclear.
- Mark behavior that is not implemented with an explicit label such as `PLANNED, not implemented`. Do not rely on vague future tense. Planned statements are direction, not current conformance targets.
- Describe current behavior directly, state each fact once in its owning section, and keep rejected alternatives or change history out of governed prose.
- Keep the work scoped to the current change. Surface pre-existing drift instead of silently absorbing it into a broader rewrite or leaving it unmentioned.
- Limit a document edit to the specification or description delta, and limit a code edit to the behavior in scope. A real conformance change must implement that behavior directly; do not add fallbacks, guards, or silent skips solely to appear compliant.
- Do not use co-evolution to avoid writing down a requirement that is already known; use `--docfirst` instead.

## Outside this workflow

- If neither the code nor the document is authoritative, reconcile each divergence with the user before editing.
- If a central document is brand new and no code exists yet, treat it as a design task rather than a consistency task.
- If a file has neither a sentinel nor an explicit invocation, edit it normally.
