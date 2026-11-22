# Single Source of Truth (SSOT)

Applies repository-wide to code and documentation.

- Give each configuration value, state, decision, and documented fact one authoritative owner. Find that owner before adding another representation; consumers should read, derive from, or reference it. Consolidate independently maintained copies instead of adding synchronization logic.
- Derive decisions once at the owner and pass them through instead of repeating the logic at call sites. Do not add per-field fallback or merge layers over an already authoritative input. Cached derived values follow the lifecycle guidance in [General Code Style](general-code-style.md).
- In documentation, keep each specification or procedure in one canonical location and link to it from other pages. Use source-backed summaries or examples when needed for the reader's task; avoid copying full instructions, configuration tables, or defaults that require separate updates. Code-owned behavior and defaults remain authoritative in code.
- Edit generated documentation at its source and regenerate it; never maintain generated output independently. Documentation site conventions and the existing example-documentation generation workflow are defined in [docs/README.md](../../docs/README.md).
