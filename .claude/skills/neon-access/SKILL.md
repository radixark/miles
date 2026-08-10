---
name: neon-access
description: Run arbitrary PostgreSQL SQL against the Miles Neon database through the repository-authorized GitHub workflow and return the encrypted result. Use when a repository writer asks to query, inspect, update, migrate, or otherwise operate on Neon without exposing NEON_DATABASE_URL. The SQL is not classified, rewritten, split, or restricted; the database role remains the authorization boundary.
---

# Neon Access

Use this skill when a writer asks to run SQL, for example: “Use `$neon-access` to run `SELECT now()`.” The workflow is the only route to Neon; never ask for or expose `NEON_DATABASE_URL`.

## Run SQL

1. Treat an explicit request as authorization to generate the exact SQL and a short audit reason, then dispatch without a second confirmation. Ask only when unresolved ambiguity would materially change the SQL. Treat every statement as potentially state-changing because the skill does not inspect or classify it.
2. Save the exact UTF-8 SQL in a temporary `.sql` file. Preserve the text byte-for-byte; do not normalize, split, wrap, retry, or add a transaction.
3. From the repository root, run:

   ```bash
   python3 .claude/skills/neon-access/scripts/run_neon_workflow.py \
     --reason '<short audit reason>' \
     /absolute/path/to/request.sql
   ```

4. Return the decoded JSON result and workflow URL. If the result status is `error`, report the database error and possible partial effects; never retry automatically.

## Boundaries

- The caller and workflow require GitHub `write` or `admin` permission; the workflow checks both the original and triggering actors.
- The workflow reads the existing `secrets.NEON_DATABASE_URL`. Users do not retrieve or provide this secret.
- Any PostgreSQL SQL accepted by the server is allowed, including multiple statements, DDL, DML, functions, and explicit transaction control.
- `psql` backslash commands are client commands, not PostgreSQL SQL, and are unsupported.
- Execution uses autocommit and performs no implicit transaction wrapping. A later error can leave earlier statements committed.
- Returned rows are capped at 4 MiB; this does not limit or cancel SQL execution.
- Results and database errors are encrypted for the one-time local key before artifact upload. Public logs and summaries contain metadata only.
